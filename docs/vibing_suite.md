# Five-experiment suite on `vibing`

The suite is implemented independently of the legacy graph/sample stubs, using
`python -m epgnn.experiment_suite`. It shares the audited calendar, canonical
geography and target masks across all five methods. The full run has **not**
been launched: the user requested launching it from a separate host `tmux` session.

## Launch, monitor, stop and resume

From the host terminal where `scripts/check_rocm.py` passes:

```bash
cd /srv/cs224w/Project/arsen/cs_224w
# Runs in a separate persistent session and returns immediately.
tmux new-session -d -s vibing-full './scripts/experiments/run_vibing_suite.sh'

# Watch logs without attaching.
tail -f runs/vibing_suite_launch/full.log

# Inspect the current generation and phase status.
.venv/bin/python -m epgnn.experiment_suite status

# Attach; detach with Ctrl-b then d. Ctrl-C while attached stops the job.
tmux attach -t vibing-full
```

The launcher and Python runner both reject concurrent launches of the same job.
After a stop/failure, launch the same command again, once the old tmux session
has exited. Completed experiment outputs are hash-checked and reused. XGBoost
resumes at completed zone models; neural models resume at the last completed
epoch, restoring optimizer, early-stopping, preprocessing and random states.
A partially processed epoch is repeated. GPU loss or unexpected failures stop the
job explicitly; the runner never silently switches to CPU.

`runs/vibing_suite_launch/exit.json` records the exit code (zero means success).
`status` identifies the exact fingerprinted artifact directory. A `running`
status after an ungraceful machine/process termination can be stale; the process
lock and logs distinguish that from a live job. Avoid changing the implementation,
config, Python environment or source data during a run: those define its identity.

## Experiments and fixed settings

| Name | Inputs / model |
|---|---|
| `recent_xgb` | Local hour, weekday, month, guarded price_lag24, price_lag168 |
| `rich_xgb` | Recent inputs + load_fc + temperature/anomaly/wind/solar at lag 48 + three lag-48 JAO summaries |
| `neighbour_xgb` | Rich inputs + incoming neighbours' guarded prior-day/weekly price means and finite-neighbour counts |
| `real_gat` | Rich inputs, two spatial GATv2 layers on training-observed directed physical pairs |
| `shuffled_gat` | Same learner and inputs, directed degree-preserving randomized topology |

The GAT is an hourly spatial model with two layers, 64 total hidden channels,
four heads, ELU, dropout 0.1, and a linear scalar-price head. It does not use a
history GRU, residual target, spread loss or detailed CNECs. It avoids the legacy
history-unit defect. Neural models share parameters across zones; XGBoost fits
separate zone models. Only real-vs-shuffled GAT isolates topology under the same
learner, rather than comparing these different parameter-sharing choices.

All models use asinh(price/50) and L1 loss. The inverse is 50*sinh(prediction),
with metrics in EUR/MWh. XGBoost uses CPU histogram trees, depth 6, maximum
600 rounds, learning rate 0.05, row/column sampling 0.8, child weight/lambda 1,
seed 224, four threads, and 50-round validation stopping. GAT uses ROCm FP32,
Adam 0.001, seed 224, maximum 30 epochs, patience five, gradient clipping at five,
and batch size 64. Validation stopping is on transformed MAE; there is no
parameter search against test. GPU preflight halves batches if allocation fails
or exceeds 12 GiB, down to a minimum of eight.

The full fit uses all finite pretrain/train labels (2016-01-01–2024-03-31),
validation 2024-04-01–2025-03-31, and test 2025-04-01–2026-09-30. Dates are
Brussels delivery dates; October stays unassigned. There are 49 identities with
some valid labels in each of fit/validation/test; the other 13 are explicitly
unscored, including DE_AT_LU and GB. All 62 canonical identities remain in graph
context, with historical lifetime masks. DE_AT_LU is never relabelled DE_LU.

## Admission and missingness

These are **conditional experiments**, not certified historical operational
forecasts. Literal Brussels-noon origins and exact source timestamp joins are
inherited from the audited master. Price_lag24 is guarded against landing in the
target delivery date during autumn DST; past valid-time alone does not certify
historical publication availability. Previous-day auction prices normally precede
the origin. Load, ERA5 weather and JAO historical publication/revision vintages
remain unverified. Weather is reanalysis, not archived forecasts or perfect
future weather; lag 48 is a timing restriction, not proof of availability.

JAO inputs are n_presolved, min_ram_ratio and domain publication presence from
same-zone lag-48 CWE/CORE domains. Missing domains, nonregional absences and
published numeric zeros remain distinct. Original JAO history before the master
window is read when necessary. No MaxBex value is treated as physical capacity.

Topology is the union of directed physical pairs observed in pretrain/train,
never selected using test coverage. This is a static training-topology assumption,
not a reconstruction of every historical commissioning date. Inactive historical
nodes cannot send graph messages or contribute to neighbour means. Shuffling
preserves global in/out degrees and forbids duplicates and self-loops; lifetime
masking can change the effective hourly degrees. GAT adds its own self-loops.

Native NaNs remain for XGBoost. Neural median imputation, mean and standard
 deviation are fitted on active training-period feature rows only and then frozen;
missingness indicators accompany every feature. Entirely absent training features
use an explicit zero fallback with a missingness indicator. Targets are never
imputed, and losses index valid labels **before** subtraction, including spread
loss, fixing NaN-times-zero contamination.

## Completed preflight and France gate

The RX 9070 XT, ROCm 7.2 and PyTorch 2.14.1+rocm7.2 passed both real/shuffled
forward passes, NaN masking, finite gradients, Adam, validation, checkpoint reload,
and resumed optimizer updates. The actual neural epoch loop also passed an
interrupted-versus-continuous two-epoch weight comparison. A representative
62-node/520-edge batch used about **0.23 GiB** of tensor allocation; this excludes
ROCm context and other processes' VRAM.

All five France checks passed, retaining 8,760 validation and 13,152 test targets.
The three XGBoost checks fit the complete France training period. Neural smoke
fits deliberately use only the last 90 training days, three epochs, and the first
14 validation days for stopping, then score full evaluation periods. These short
smoke scores are **not fully trained model comparisons** or evidence of a graph
advantage. Full models start fresh, with all training and validation data.

Warm neural smoke epochs took approximately 0.18–0.20 seconds for 2,160 training
hours. Scaling to 72,310 hours and 30 epochs for both GATs gives roughly seven
minutes of neural training before full validation and overhead. France XGBoost
checks took 0.5–0.9 seconds each; 49 zones add roughly two minutes at that rate.
Allow **10–20 minutes** for the full suite as a rough estimate, not a guarantee;
large training preprocessing, evaluation, disk and other system load affect it.
Early stopping may finish sooner. No long unattended run is required to verify
that the pipeline functions.

Reproduce the gates in the ROCm-enabled host environment:

```bash
.venv/bin/python -m epgnn.experiment_suite gpu-smoke
.venv/bin/python -m epgnn.experiment_suite france
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m unittest discover -s tests/dataset_build -p 'test_*.py'
```

## Results and provenance

- `results/vibing_suite_france.json`, `_by_zone.csv`, `_aggregate.csv`: saved smoke
  evidence; full evaluation will write corresponding `vibing_suite_full` files.
- `runs/vibing_suite/<fingerprint>/{france,full}/`: status, provenance, per-method
  model files, keyed val/test predictions and artifact checksums. Neural `latest.pt`
  and `best.pt` include optimizer/random states and curves. Load these trusted
  project checkpoints with the same implementation and environment.
- `staging/vibing_suite/<cache fingerprint>/`: source-checked, fixed-width
  memory-mapped data. Preparation reads one zone at a time; graph training reads
  bounded batches. The cache records source hashes, geography and timing rules.

Tables report zone, country/market group, overall, and matched JAO-publication
subsets, including counts. Country groups pool bidding-zone error sums; they do
not construct national average-price labels. Combined DE_LU/IE_SEM and unresolved
ENTSOE identifiers remain distinct. All models share valid target masks, with
persistence compared on the common finite-history subset. RMSE pools squared
errors rather than averaging regional RMSEs. Existing repeated test inspection
means these remain exploratory, single-seed experiments.

Full summaries are saved to `results/` automatically, but the training process
never runs Git commands. Implementation and smoke results are committed before
launch; the full results can be reviewed and committed afterwards.
