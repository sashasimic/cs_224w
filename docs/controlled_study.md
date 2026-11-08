# Classical neural, graph and temporal controls

**Current status:** the static comparisons remain retrospective conditional
evidence. The original GRU training path fails CPU agreement on shuffled
training-shaped inputs; corrected temporal fits are needed. This design and
its code are retained for reproduction. See the [consolidated report](experiments.md).

This separate study implements the predefined 100-job matrix. It preserves the
completed five-model suite, its predictions and its fingerprints. New full
results are not available until the host run finishes; smoke results are pipeline
checks and are not multi-seed scientific conclusions.

## Launch and monitor

Use the ROCm-enabled host terminal from the repository root:

```bash
.venv/bin/python -m epgnn.controlled_study dry-run
.venv/bin/python -m epgnn.controlled_study smoke
.venv/bin/python -m epgnn.controlled_study status

# After the smoke phase is complete, launch separately and detach immediately.
tmux new-session -d -s controlled-full './scripts/experiments/run_controlled_study.sh'
tail -f runs/controlled_study_launch/full.log
```

Ctrl-C exits `tail` without stopping training. Attach with
`tmux attach -t controlled-full`; Ctrl-b then d detaches, while Ctrl-C interrupts
the foreground training process. Relaunch the same command after the old session
exits to resume. Duplicate launches are rejected by both launcher and phase locks.
The runner records a failed/interrupted state and traceback rather than silently
changing data, batch size or hardware. `exit.json` beside the log records the
launcher exit code. A hard kill can leave a stale running status; locks and logs
distinguish it from a live process.

Keep code, configuration and environment unchanged while a study is running.
Models, configuration and source hashes define the generation. New code requires
its own preflight and France checks. Neural checkpoints restore the architecture,
optimizer, early stopping, preprocessing, sampler epoch and torch random states;
an interrupted epoch is repeated. Completed jobs are checksum-verified and reused.

## Matrix

| Group | Variants | Runs |
|---|---|---:|
| Tabular references | Per-zone recent/rich/neighbour XGBoost; pooled rich XGBoost | 20 |
| Feature ablations | Pooled MLP and real GAT, each with five feature groups | 50 |
| Topology | Self-loop GAT; three fixed rewired GAT topologies | 20 |
| Temporal | GRU + MLP; GRU + real GAT | 10 |
| Total | Five training seeds for every variant | 100 |

Training seeds are 224–228, and rewiring seeds are 901–903. Each per-zone XGBoost
job contains 49 fitted zone models; a pooled job is one model. The feature groups
are recent price/calendar, recent plus load only, weather only, JAO only, and all
three additions. No feature set is selected by inspecting test results.

All three classical/per-zone references preserve existing feature definitions.
The pooled XGBoost and neural models have no country/zone identity feature.
Both MLP and GAT share parameters across zones. MLP has two ELU/dropout hidden
layers; width is selected deterministically by matching GAT parameter count,
not by scoring candidate widths. Rich static inputs give 96-unit MLP layers:
12,001 parameters versus GAT's 12,097. Every job records its parameter count and
actual architecture. The temporal heads are matched the same way.

GAT remains two layers, 64 channels, four heads and dropout 0.1. Self-loop GAT
uses the identical architecture with an empty non-self edge list. Real and shuffled
graphs use training-only physical-pair topology and dated node lifetimes. The
three shuffled graphs preserve static in/out degrees without duplicates/self
edges, and are fixed across training seeds. GAT adds its own self-loops. No model
uses same-day realised flows as features. Neural learning uses FP32, Adam 0.001,
maximum 30 epochs, patience five and gradient clipping at five. All models fit
asinh(price/50), use L1 loss and invert to EUR/MWh for scoring.

## Histories and admission

GRU models add a shared single-layer, 32-unit encoder over the last 168 completed
hourly price intervals before the literal Brussels-noon origin. Raw history uses
the canonical price identity, explicit timestamp lookup and original missingness,
never row-offset history or a fixed UTC cutoff. The full canonical calendar starts
at the earliest source price hour; earlier unavailable observations stay missing.
No DE_AT_LU history is attached to DE_LU. Inactive identities have missing history.

History values use asinh(price/50), training-only median/mean/scale and a missing
indicator. Histories are stored once per distinct forecast origin and encoded
once per distinct origin in each batch. The original hourly batch schedule is
retained. Both temporal models use the same batch size: the GPU preflight reduces
the temporal family together if required to fit below 12 GiB, with a minimum of
eight. Static models use their own common preflight batch size.

The data calendar, 49-identity eligibility, splits, target masks and conditional
feature status remain unchanged. Historical publication/revision vintages for
load, reanalysis weather and JAO are unresolved. Past valid time is not proof of
publication availability. This is an exploratory conditional study, not an
operational availability certification.

## Gates, analysis and outputs

The smoke command first verifies all four neural architectures, the three GAT
edge modes, missing-label gradients and interrupted-versus-continuous training
on ROCm. It then runs all 20 seed-224 prototypes on France, checks saved-model
predictions, and requires identical target-key/value/mask signatures. Neural France
checks use the last 90 training days, three epochs and 14 validation days for
stopping, then score full validation/test periods. They do not replace full fits.
The full command refuses to start without the matching smoke generation.

`status` displays the phase and an estimate extrapolated from warm France epochs
and worst-case full-node GPU batches. Temporal computation, not GPU memory, is
likely to dominate. The estimate assumes all 30 epochs and sequential jobs;
early stopping, pooling and system load affect the actual duration.

Each job saves architecture-aware best/latest checkpoints (or XGBoost models),
keyed predictions, learning curves, training-only regime thresholds, per-zone
scores and validation weekly error sums. The full analysis reports MAE/RMSE,
paired persistence comparisons, mean/sample SD across seeds, country/season and
negative/upper/lower price-tail slices. Price-tail thresholds are each zone's
training-only 5th/95th percentiles. Empty slices retain zero counts.

The 2,000-resample paired calendar-week bootstrap resamples weeks jointly across
countries after averaging fixed-model losses across seeds. It uses common finite
weekly-history hours, matching primary paired MAE. This conditions on the fitted
models; seed and rewiring variability are summarized separately. Scores, not
predictions, are averaged, and no best test seed is selected.

Artifacts live under `runs/controlled_study/<fingerprint>/{smoke,full}/`.
Tracked summaries are written as `results/controlled_study_{smoke,full}.json`,
`_scores.csv`, `_by_zone.csv` and `_regimes.csv`. Source/tensor caches remain in
staging. Full completion also generates `docs/controlled_study_results.md` and
rebuilds the two-page milestone with multi-seed controls; an incomplete study
cannot replace the milestone. A report-generation failure is explicit and can
be retried without fitting completed jobs. The launcher never stages or commits
Git changes; full results can be reviewed and committed afterwards.

Implementation is in `epgnn/controlled_study/`: `design` defines the matrix,
`history` and `models` define inputs/architectures, `training` owns shared neural
training, `preflight` checks ROCm and resume, `runner` orchestrates jobs, and
`analysis`/`report` publish results. Use named scripts/module commands throughout.
