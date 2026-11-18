"""Chronological calibration and resumable refits for the combined study."""

import itertools
import time

import numpy as np
import torch

from epgnn.dataset_build.storage import json_write
from epgnn.experiment_suite.neural import atomic_checkpoint
from epgnn.train import masked_loss

from . import models, sql_features

XGB = ("sql_price", "sql_full")
RELATIONAL = ("relational_local", "relational_cross")
FAMILIES = {"xgb": XGB, "relational": RELATIONAL, "adaptive": models.ADAPTIVE}
VARIANTS = tuple(itertools.chain.from_iterable(FAMILIES.values()))
TRAIN_START = {
    "xgb": "2016-01-01",
    "relational": "2016-01-01",
    "adaptive": "2022-06-09",
}


def split(data, family, smoke):
    days = data.dates.to_numpy()
    fit = np.flatnonzero((TRAIN_START[family] <= days) & (days < "2024-01-01"))
    calibrate = np.flatnonzero(("2024-01-01" <= days) & (days < "2024-04-01"))
    validation = np.flatnonzero(("2024-04-01" <= days) & (days < "2025-04-01"))
    if smoke:
        # Use complete delivery days even around DST.
        fit = fit[np.isin(days[fit], np.unique(days[fit])[-smoke["fit_days"] :])]
        calibrate = calibrate[
            np.isin(
                days[calibrate], np.unique(days[calibrate])[: smoke["calibration_days"]]
            )
        ]
        validation = validation[
            np.isin(
                days[validation],
                np.unique(days[validation])[: smoke["validation_days"]],
            )
        ]
    return fit, calibrate, validation


class NeuralPhase:
    def __init__(self, data, family, training, device, config):
        self.family = family
        self.data = data
        self.config = config
        self.tensors = models.TensorInputs(data, training, device, family)
        self.device = device

    def units(self, indices):
        return (
            np.unique(self.data.day[indices])
            if self.family == "relational"
            else indices
        )

    def batches(self, units):
        n = (
            self.config["relational_days"]
            if self.family == "relational"
            else self.config["adaptive_hours"]
        )
        for start in range(0, len(units), n):
            yield units[start : start + n]

    def forward(self, model, variant, units):
        t = self.tensors
        if self.family == "relational":
            hours, active, observations = t.relational_batch(units)
            keep = hours >= 0
            indices = hours.clip(min=0)
            result = model(observations, t.recent[indices], t.edges)
            return result[keep], t.price[hours[keep]], active[keep], None
        evidence = None
        if variant in {"margin", "shuffled_margin"}:
            evidence = t.pair_evidence(units, variant == "shuffled_margin")
        output, weight = model(
            t.recent[units], t.active[units], t.incoming, t.core, evidence
        )
        return output, t.price[units], t.active[units], weight

    @torch.no_grad()
    def infer(self, model, variant, hours, links=False):
        model.eval()
        forecasts = []
        weights = np.zeros((len(self.data.zones), len(self.data.zones)), np.float64)
        for batch in self.batches(self.units(hours)):
            output, _, _, attention = self.forward(model, variant, batch)
            forecasts.append(output.cpu().numpy())
            if links and attention is not None:
                weights += attention.sum(dim=0).cpu().numpy()
        values = np.concatenate(forecasts)
        if len(values) != len(hours) or not np.isfinite(values).all():
            raise ValueError("Forecast shape or finiteness failed")
        return values, weights / len(hours)

    def fit(self, variant, seed, training, directory, limit, calibrate=None):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        width = self.config["width"]
        if self.family == "relational":
            model = models.RelationalGNN(width, local=variant == "relational_local").to(
                self.device
            )
        else:
            model = models.AdaptiveGNN(width, len(self.data.zones), variant).to(
                self.device
            )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"])
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "latest.pt"
        curves = []
        best, best_epoch, stale = np.inf, 0, 0
        if checkpoint.exists():
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["cpu_rng"].cpu())
            torch.cuda.set_rng_state_all([v.cpu() for v in state["gpu_rng"]])
            curves, best, best_epoch, stale = (
                state["curves"],
                state["best"],
                state["best_epoch"],
                state["stale"],
            )
        for epoch in range(len(curves), limit):
            if stale >= self.config["patience"]:
                break
            start = time.monotonic()
            torch.cuda.reset_peak_memory_stats(self.device)
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(
                self.units(training)
            )
            for batch in self.batches(order):
                optimizer.zero_grad(set_to_none=True)
                prediction, y, active, _ = self.forward(model, variant, batch)
                loss = masked_loss(prediction, y, active)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5, error_if_nonfinite=True
                )
                optimizer.step()
            value = None
            if calibrate is not None:
                pred, _ = self.infer(model, variant, calibrate)
                actual = np.arcsinh(self.data.prices[calibrate] / np.float32(50))
                keep = np.isfinite(actual)
                value = float(np.abs(pred[keep] - actual[keep]).mean())
                improved = value < best
                best, best_epoch, stale = (
                    (value, epoch + 1, 0) if improved else (best, best_epoch, stale + 1)
                )
            curves.append(
                {
                    "epoch": epoch + 1,
                    "calibration_l1": value,
                    "seconds": time.monotonic() - start,
                    "peak_vram_gib": torch.cuda.max_memory_allocated(self.device)
                    / 2**30,
                }
            )
            atomic_checkpoint(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "cpu_rng": torch.get_rng_state(),
                    "gpu_rng": torch.cuda.get_rng_state_all(),
                    "curves": curves,
                    "best": best,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
            )
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == limit:
                print(f"{variant} {seed} {directory.name} {curves[-1]}", flush=True)
        if calibrate is not None and best_epoch < 1:
            raise ValueError("An eligible calibration epoch is required")
        return model, best_epoch, curves


def neural_jobs(data, family, fit, calibration, validation, settings, smoke, device):
    config = settings["neural"]
    cutoff = np.concatenate((fit, calibration))
    calibration_phase = NeuralPhase(data, family, fit, device, config)
    refit_phase = NeuralPhase(data, family, cutoff, device, config)
    limit = smoke["epochs"] if smoke else config["epochs"]

    def job(variant, seed, output):
        _, epochs, calibration_curve = calibration_phase.fit(
            variant, seed, fit, output / "calibration", limit, calibrate=calibration
        )
        model, _, refit_curve = refit_phase.fit(
            variant, seed, cutoff, output / "refit", epochs
        )
        pred, weights = refit_phase.infer(
            model, variant, validation, links=family == "adaptive"
        )
        if family == "adaptive":
            np.save(output / "link_weights.npy", weights)
        return 50 * np.sinh(pred), {
            "selected": epochs,
            "calibration": calibration_curve,
            "refit": refit_curve,
        }

    return job


def xgb_jobs(data, fit, calibration, validation, settings, smoke, cache):
    import xgboost as xgb

    xgb_settings = settings["xgb"]
    max_rounds = smoke["trees"] if smoke else xgb_settings["max_rounds"]
    params = {
        k: v for k, v in xgb_settings.items() if k not in {"max_rounds", "patience"}
    }
    params.update(objective="reg:absoluteerror", eval_metric="mae", tree_method="hist")
    current = None

    def matrix(hourly, aggregates, hours, training=False, reference=None):
        y = np.arcsinh(data.prices[hours].ravel() / np.float32(50))
        keep = np.isfinite(y) if training else np.ones(len(y), dtype=bool)
        local = hourly[hours].reshape(len(y), -1)
        historical = aggregates[data.day[hours]].reshape(len(y), -1)
        x = np.concatenate((local[keep], historical[keep]), axis=-1)
        if not training:
            return x
        return xgb.QuantileDMatrix(
            x,
            label=y[keep],
            ref=reference,
            max_bin=params["max_bin"],
            nthread=params["nthread"],
        )

    def job(variant, seed, output):
        nonlocal current
        if current is None or current[0] != variant:
            current = (
                None  # Release prior histograms before building the next feature set.
            )
            daily = sql_features.features(data, variant == "sql_full", cache)
            hourly = data.recent
            initial = matrix(hourly, daily, fit, training=True)
            stopping = matrix(
                hourly, daily, calibration, training=True, reference=initial
            )
            final = matrix(
                hourly, daily, np.concatenate((fit, calibration)), training=True
            )
            holdout = matrix(hourly, daily, validation)
            current = variant, initial, stopping, final, holdout
        _, initial, stopping, final, holdout = current
        config = params | {"seed": seed}
        checkpoint = output / "selection.json"
        if checkpoint.exists():
            import json

            selection = json.loads(checkpoint.read_text())
        else:
            curve = {}
            started = time.monotonic()
            learner = xgb.train(
                config,
                initial,
                max_rounds,
                evals=[(stopping, "calibration")],
                evals_result=curve,
                early_stopping_rounds=xgb_settings["patience"],
                verbose_eval=20,
            )
            selection = {
                "selected": learner.best_iteration + 1,
                "calibration": curve,
                "calibration_seconds": time.monotonic() - started,
            }
            json_write(checkpoint, selection)
        model_file = output / "refit.ubj"
        started = time.monotonic()
        if model_file.exists():
            trained = xgb.Booster(model_file=model_file)
        else:
            trained = xgb.train(config, final, selection["selected"])
            temporary = model_file.with_name("refit.partial.ubj")
            trained.save_model(temporary)
            temporary.replace(model_file)
        pred = trained.inplace_predict(holdout).reshape(len(validation), -1)
        return 50 * np.sinh(pred), selection | {
            "refit_seconds": time.monotonic() - started
        }

    return job
