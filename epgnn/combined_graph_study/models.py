"""Directed inter-market attention and factorized typed GraphSAGE."""

import numpy as np
import torch
from torch import nn

from epgnn.experiment_suite.neural import Preprocessor

RELATIONS = ("price", "weather", "load", "jao", "flow")
ADAPTIVE = ("physical", "uniform", "learned", "margin", "shuffled_margin")


def device_tensor(value, device):
    return torch.as_tensor(np.array(value, copy=True), device=device)


def normalize(values, fit):
    flat = fit.reshape(-1, fit.shape[-1])
    return (
        Preprocessor()
        .fit(flat)
        .transform(values.reshape(-1, values.shape[-1]))
        .reshape(*values.shape[:-1], 2 * values.shape[-1])
    )


class TensorInputs:
    """Hold compact normalized observations once on device for each fit phase."""

    def __init__(self, data, training, device, family):
        self.active = device_tensor(data.active, device)
        self.price = device_tensor(np.arcsinh(data.prices / np.float32(50)), device)
        recent = normalize(data.recent, data.recent[training][data.active[training]])
        recent[~data.active] = 0
        self.recent = device_tensor(recent, device)
        self.edges = device_tensor(data.edges, device)
        self.device = device
        self.day = data.day
        self.slots = np.full((len(data.origins), 25), -1, np.int64)
        first = np.searchsorted(data.day, np.arange(len(data.origins)))
        self.slots[data.day, np.arange(len(data.day)) - first[data.day]] = np.arange(
            len(data.day)
        )
        self.incoming = torch.zeros(
            (len(data.zones), len(data.zones)), dtype=torch.bool, device=device
        )
        self.incoming[self.edges[:, 0], self.edges[:, 1]] = True
        self.core = device_tensor(data.core, device)
        if family == "relational":
            self.historical = device_tensor(data.historical, device)
            self.origins = device_tensor(data.origins, device)
            self.day_active = device_tensor(data.day_active, device)
            self.panels = {}
            fit_days = np.unique(data.day[training])
            for name in RELATIONS:
                values, presence, *publication = data.panels[name]
                window = data.historical[fit_days]
                if publication:
                    window = window[:, -72:-24]
                lo, hi = window.min(), window.max() + 1
                sample = values[lo:hi]
                if publication:
                    # For each source hour, the latest fit origin that can use it.
                    cutoff = np.full(len(sample), np.iinfo(np.int64).min, np.int64)
                    np.maximum.at(
                        cutoff,
                        (window - lo).ravel(),
                        np.repeat(data.origins[fit_days], window.shape[1]),
                    )
                    sample = np.where(
                        (publication[0][lo:hi] <= cutoff[:, None])[..., None],
                        sample,
                        np.nan,
                    )
                packed = normalize(values, sample)
                self.panels[name] = (
                    device_tensor(packed, device),
                    device_tensor(presence, device),
                    device_tensor(publication[0], device) if publication else None,
                )
        else:
            rows = np.searchsorted(data.margin_hours, training)
            if not np.array_equal(data.margin_hours[rows], training):
                raise ValueError("Adaptive training hours must fall within CORE")
            norm = normalize(data.margin, data.margin[rows])
            usable = np.isfinite(data.margin[..., 0])
            self.margin = device_tensor(norm, device)
            self.margin_usable = device_tensor(usable, device)
            self.margin_origin = int(data.margin_hours[0])
            rng = np.random.default_rng(901)
            permutation = np.tile(
                np.arange(len(data.core)), (len(data.margin_hours), 1)
            )
            for i, hour in enumerate(data.margin_hours):
                active = np.flatnonzero(data.active[hour, data.core])
                permutation[i, active] = rng.permutation(active)
            self.permutation = device_tensor(permutation, device)

    def relational_batch(self, days):
        positions = self.historical[days]
        observations = {}
        for name, (values, present, publication) in self.panels.items():
            idx = positions if publication is None else positions[:, -72:-24]
            valid = present[idx]
            if publication is not None:
                valid = valid & (publication[idx] <= self.origins[days, None, None])
            active = self.day_active[days]
            if name == "flow":
                source, dest = self.edges.T
                active = active[:, source] & active[:, dest]
            valid &= active[:, None, :]
            observations[name] = values[idx], valid
        hours = self.slots[days]
        active = (
            self.active[hours.clip(min=0)]
            & device_tensor(hours >= 0, self.device)[..., None]
        )
        return hours, active, observations

    def pair_evidence(self, hours, shuffled):
        rows = hours - self.margin_origin
        values, valid = self.margin[rows], self.margin_usable[rows]
        if shuffled:
            p = self.permutation[rows]
            batch = torch.arange(len(hours), device=self.device)[:, None, None]
            values, valid = (
                values[batch, p[:, :, None], p[:, None, :]],
                valid[batch, p[:, :, None], p[:, None, :]],
            )
        return values, valid


class RelationEncoder(nn.Module):
    def __init__(self, fields, width):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(fields + 3, 12), nn.ELU())
        self.pool = nn.Linear(36, width)

    def forward(self, values, valid):
        length = values.shape[1]
        lag = torch.arange(length, device=values.device) - length
        age = torch.stack(
            (
                lag / length,
                torch.sin(lag * (2 * np.pi / 24)),
                torch.cos(lag * (2 * np.pi / 24)),
            ),
            dim=-1,
        )
        age = age[None, :, None, :].expand(*values.shape[:3], -1)
        event = self.project(torch.cat((values, age), dim=-1)) * valid[..., None]
        # Mix three temporal neighborhoods while retaining the original event rows.
        encoded, masks = event.tensor_split(3, dim=1), valid.tensor_split(3, dim=1)
        pooled = [
            e.sum(dim=1) / m.sum(dim=1).clamp(min=1)[..., None]
            for e, m in zip(encoded, masks, strict=True)
        ]
        return self.pool(torch.cat(pooled, dim=-1)) * valid.any(dim=1)[..., None]


class RelationalGNN(nn.Module):
    """Typed history-to-zone, zone-to-border and border-to-zone mean steps."""

    def __init__(self, width, local):
        super().__init__()
        self.local = local
        dims = {"price": 2, "weather": 8, "load": 2, "jao": 6, "flow": 2}
        self.observations = nn.ModuleDict(
            {k: RelationEncoder(dims[k], width) for k in RELATIONS}
        )
        self.zone = nn.Linear(4 * width, width)
        self.from_zone = nn.Linear(width, width, bias=False)
        self.to_zone = nn.Linear(width, width, bias=False)
        self.incoming = nn.Linear(width, width, bias=False)
        self.outgoing = nn.Linear(width, width, bias=False)
        self.head = nn.Sequential(
            nn.Linear(3 * width + 10, width),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(width, 1),
        )

    @staticmethod
    def aggregate(message, ends, valid, zones):
        batch, _, features = message.shape
        indices = ends[None, :, None].expand(batch, -1, features)
        summed = message.new_zeros(batch, zones, features).scatter_add_(
            1, indices, message * valid[..., None]
        )
        count = message.new_zeros(batch, zones).scatter_add_(
            1, ends.expand(batch, -1), valid.to(message.dtype)
        )
        return summed / count.clamp(min=1)[..., None]

    def forward(self, batch, recent, edges):
        obs = {k: self.observations[k](*batch[k]) for k in RELATIONS}
        state = torch.nn.functional.elu(
            self.zone(torch.cat([obs[k] for k in RELATIONS[:-1]], dim=-1))
        )
        source, dest = edges.T
        source_state = self.from_zone(state[:, source])
        dest_state = self.to_zone(state[:, dest])
        # The local control uses endpoint copies of each border observation.
        if self.local:
            out_state, in_state = obs["flow"] + source_state, obs["flow"] + dest_state
        else:
            out_state = in_state = obs["flow"] + source_state + dest_state
        valid = batch["flow"][1].any(dim=1)
        n = state.shape[1]
        outgoing = self.aggregate(
            self.outgoing(torch.tanh(out_state)), source, valid, n
        )
        incoming = self.aggregate(self.incoming(torch.tanh(in_state)), dest, valid, n)
        history = torch.cat((state, outgoing, incoming), dim=-1)
        data = torch.cat(
            (history[:, None].expand(-1, recent.shape[1], -1, -1), recent), dim=-1
        )
        return self.head(data).squeeze(-1)


class AdaptiveGNN(nn.Module):
    def __init__(self, width, zones, variant):
        super().__init__()
        self.variant = variant
        self.local = nn.Sequential(nn.Linear(10, width), nn.ELU())
        self.key = nn.Linear(width, 16, bias=False)
        self.query = nn.Linear(width, 16, bias=False)
        self.sender = nn.Embedding(zones, 16)
        self.receiver = nn.Embedding(zones, 16)
        self.margin = nn.Linear(8, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(2 * width, width), nn.ELU(), nn.Dropout(0.1), nn.Linear(width, 1)
        )

    def forward(self, recent, active, physical, core, evidence=None):
        state = self.local(recent)
        # Axes are source, destination. Normalize incoming scores per destination.
        score = (
            (self.key(state) + self.sender.weight)
            @ (self.query(state) + self.receiver.weight).transpose(-1, -2)
            / 4
        )
        if self.variant == "uniform":
            score = torch.zeros_like(score)
        if evidence is not None:
            margin, valid = evidence
            score[:, core[:, None], core[None, :]] += (
                self.margin(margin).squeeze(-1) * valid
            )
        allowed = active[:, :, None] & active[:, None, :]
        if self.variant == "physical":
            allowed &= physical
        else:
            allowed &= ~torch.eye(state.shape[1], dtype=torch.bool, device=state.device)
        weight = torch.softmax(score.masked_fill(~allowed, -1e9), dim=1) * allowed
        message = weight.transpose(-1, -2) @ state
        return self.head(torch.cat((state, message), dim=-1)).squeeze(-1), weight
