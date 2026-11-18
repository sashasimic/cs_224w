"""Build time-indexed CORE snapshots and apply signed bipartite attention."""

from contextlib import closing

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from epgnn.experiment_suite.neural import Preprocessor

from .catalog import CORE, connect


def observations(database, times, zones):
    """Read only train/validation CORE observations, with an exact as-of SQL join."""
    hours = pd.to_datetime(times, unit="ns", utc=True)
    with closing(connect(database)) as connection:
        connection.register("requested_hours", pd.DataFrame({"delivery_utc": hours}))
        connection.execute("""
            CREATE TEMP TABLE eligible_domain AS
            SELECT f.delivery_utc, f.source_hour FROM core_forecast_domain f
            JOIN requested_hours USING(delivery_utc) WHERE f.domain_present
        """)
        summaries = connection.execute("""
            SELECT epoch_ns(d.delivery_utc) AS time_ns, s.zone, s.n_presolved,
                   s.min_ram_ratio, 1.0 AS domain_present
            FROM eligible_domain d JOIN jao_zone_summary s ON s.ts_utc = d.source_hour
            WHERE s.regime = 'CORE'
        """).df()
        # Project only the source-side CORE coefficients used by this study.
        coefficients = ", ".join(f"c.ptdf_{zone}" for zone in CORE)
        constraints = connection.execute(f"""
            SELECT epoch_ns(d.delivery_utc) AS time_ns, c.ram_mw, c.fmax_mw, c.ram_fmax,
                   {coefficients}
            FROM eligible_domain d
            JOIN cnec_wide c ON c.ts_utc = d.source_hour AND c.regime = 'CORE'
            ORDER BY time_ns, c.cnec_key
        """).df()
    positions = np.searchsorted(times, constraints.pop("time_ns").to_numpy())
    offsets = np.searchsorted(positions, np.arange(len(times) + 1))
    summary = np.full((len(times), len(zones), 3), np.nan, np.float32)
    zone_index = summaries.zone.map({name: i for i, name in enumerate(zones)})
    priced = zone_index.notna()
    summary[
        np.searchsorted(times, summaries.time_ns[priced]),
        zone_index[priced].astype(int),
    ] = summaries.loc[priced, ["n_presolved", "min_ram_ratio", "domain_present"]]
    return {
        "offsets": offsets,
        "attributes": constraints.iloc[:, :3].to_numpy(np.float32),
        "coefficients": constraints.iloc[:, 3:].to_numpy(np.float32),
        "summary": summary,
    }


class Snapshots:
    """Concatenate independent hourly graphs into one sparse batch."""

    def __init__(self, cache, active, zones, training, shuffle_seed):
        self.offsets = cache["offsets"]
        self.coefficients = cache["coefficients"]
        fit_attributes = cache["attributes"][
            self.offsets[training[0]] : self.offsets[training[-1] + 1]
        ]
        if not len(fit_attributes):
            raise ValueError("No CORE constraint observations in the training window")
        prep = Preprocessor().fit(fit_attributes)
        self.attributes = prep.transform(cache["attributes"])
        self.core = np.asarray([zones.index(z) for z in CORE])
        finite_nonzero = np.isfinite(self.coefficients) & (self.coefficients != 0)
        self.connected = np.zeros_like(active)
        nonempty = np.flatnonzero(np.diff(self.offsets))
        self.connected[np.ix_(nonempty, self.core)] = np.logical_or.reduceat(
            finite_nonzero, self.offsets[nonempty], axis=0
        )
        self.connected &= active
        self.permutations = np.broadcast_to(
            np.arange(len(CORE)), (len(active), len(CORE))
        ).copy()
        rng = np.random.default_rng(shuffle_seed)
        for hour in nonempty:
            present = np.flatnonzero(self.connected[hour, self.core])
            self.permutations[hour, present] = rng.permutation(present)
        self.zone_count = len(zones)

    def batch(self, hours, active, device, shuffled):
        starts = self.offsets[hours]
        sizes = self.offsets[hours + 1] - starts
        batch_hour = np.repeat(np.arange(len(hours)), sizes)
        local = np.arange(len(batch_hour)) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        rows = starts[batch_hour] + local
        values = self.coefficients[rows]
        constraint, column = np.where(np.isfinite(values) & (values != 0))
        hour_index = batch_hour[constraint]
        zones = self.core[
            self.permutations[hours[hour_index], column] if shuffled else column
        ]
        enabled = active[hour_index, zones]
        constraint, column, zones = constraint[enabled], column[enabled], zones[enabled]
        edge_index = np.stack(
            (hour_index[enabled] * self.zone_count + zones, constraint)
        )
        return (
            torch.from_numpy(self.attributes[rows]).to(device),
            torch.from_numpy(edge_index).to(device),
            torch.from_numpy(values[constraint, column, None]).to(device),
        )


class ConstraintGAT(nn.Module):
    def __init__(self, features, hidden=64, heads=4):
        super().__init__()
        self.zone_input = nn.Linear(features, hidden)
        self.constraint_input = nn.Linear(6, hidden)
        self.to_constraint = GATv2Conv(
            (hidden, hidden),
            hidden // heads,
            heads=heads,
            dropout=0.1,
            edge_dim=1,
            add_self_loops=False,
            bias=False,
        )
        self.to_zone = GATv2Conv(
            (hidden, hidden),
            hidden // heads,
            heads=heads,
            dropout=0.1,
            edge_dim=1,
            add_self_loops=False,
            bias=False,
        )
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, graph):
        attributes, edges, coefficients = graph
        zones = torch.nn.functional.elu(self.zone_input(x.flatten(0, 1)))
        if edges.shape[1]:
            constraints = self.constraint_input(attributes)
            messages = self.to_constraint((zones, constraints), edges, coefficients)
            constraints = self.dropout(torch.nn.functional.elu(constraints + messages))
            zones = zones + self.to_zone(
                (constraints, zones), edges.flip(0), coefficients
            )
        return self.head(self.dropout(torch.nn.functional.elu(zones))).reshape(
            x.shape[:2]
        )
