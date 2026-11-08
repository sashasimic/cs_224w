# Partner-inspired residual GRU on the audited data

This experiment adopts the GRU+MLP approach in partner commit
`757da793c2e2e726cc6e750f2d42f1dd836ca39c`. It is an adaptation, not an exact
reproduction of their legacy-table result. Their published 20.40 MAE uses 43
scored zones; this run retains our 62 identities, 49 scored markets, original
chronological splits and unchanged target validity. It starts with one seed (0),
matching their recipe, rather than mixing this fit into existing five-seed means.

## What carries over

- Predict `asinh(price/50) - asinh(base/50)` and recover price with
  `50*sinh(predicted_residual + asinh(base/50))`.
- Base = weekly price, else guarded previous-day price, else zero. This fallback
  reconstructs predictions only; no target is filled or removed because of it.
- Eight multivariate history channels: past guarded price_lag24, price_lag168,
  load forecast, net imports, temperature, anomaly, wind and solar capacity factors.
- A shared one-layer GRU (64 units), a two-hidden-layer ReLU MLP (128 units),
  dropout 0.1, 16-day batches, Adam 0.001, clip 1, maximum 50 epochs, patience 8,
  minimum validation improvement 1e-5. There are 40,577 model parameters.
- Per-feature train-only z-scores, missing features set to zero after scaling,
  with no extra missingness indicators or zone embeddings.
- L1 residual error plus 0.5 times interconnected-zone residual-contrast L1.
  This is a contrast in transformed residual space, not EUR/MWh spread loss.
  The independently reported physical spread MAE uses original-unit predictions.
- Validation residual-plus-spread loss selects the checkpoint. The test split
  never selects epochs. Batches shuffle delivery days and encode each day's
  history once, rather than repeatedly encoding it for hourly samples.

The weights are shared across zones, as in the partner's implemented model.
There is no inference-time message passing. The spread objective couples zones
only through supervised training losses.

## Audited-data adaptations and availability

Forecast origins are previous calendar-date noon in Europe/Brussels; histories
contain 168 hourly intervals completed by that origin. Exact timestamp joins
preserve gaps and units. Days retain 23, 24 or 25 hourly slots. DE_AT_LU remains
separate, without German-only weather/history substitution; unresolved geographic
compatibility and the DE_LU weather proxy remain conditional.

Day inputs contain guarded price_lag24/168, load forecast, four weather features
at **lag48**, net imports at lag48, hour, weekday and month. This differs from the
partner's lag24 weather, some of which referred to observations after noon.
Even lag48 weather is **not operationally admissible** under documented ERA5
release latency. The model uses our supplied ERA5-lineage reanalysis export,
not weather forecasts. Historical load/flow revisions, weather release identity
and the anomaly climatology remain unverified. No leakage-free claim is made.
See [ECMWF's product description](https://www.ecmwf.int/en/forecasts/datasets/era5-hourly-time-series-data-single-levels-1940-present).

Our node master does not supply an opaque filled net-import column. We derive
net inflow from original directional physical-flow observations over the frozen
training-period pairs. A value is missing unless all incident pairs with active
endpoints have finite values; a genuinely observed zero stays zero. This is
net inflow over that graph, not an asserted total national import measure.
Later unseen edges do not enter the graph. The derivation and coverage counts
are recorded in the cache. Original data, partner source and prior models remain
unchanged. Index masks before arithmetic to avoid NaN-times-zero loss defects.

Scalers use fit days and the scored identities only; padded day slots and missing
features do not contribute. Source histories outside the canonical calendar stay
missing. Training and inference never cross zone identities. Cache construction
processes zone sources and edge batches in bounded memory; histories are gathered
only for a batch. Versioned cache/checkpoint identities include source artifact,
configuration, partner commit and code hashes.

## Run and inspect

```bash
.venv/bin/python -m unittest discover -s tests -p test_partner_gru.py -v
.venv/bin/python -u -m epgnn.partner_gru gpu-smoke
.venv/bin/python -u -m epgnn.partner_gru audit
.venv/bin/python -u -m epgnn.partner_gru smoke
tmux new-session -d -s partner-gru './scripts/experiments/run_partner_gru.sh'
.venv/bin/python -m epgnn.partner_gru status
tail -n 20 runs/partner_gru_launch/full.log
```

Commands accept `--config`. Smoke checks the actual architecture, masked
residual/spread gradients, Adam, checkpoint round trips and resumed-versus-
continuous epochs on ROCm. A three-epoch France-supervised fit then tests actual
inputs, original-unit predictions and scoring, using the last 90 fit days and
first 14 validation days for early stopping. Its full-period scores are pipeline
checks, not fully trained scientific results. Full training requires matching
verified smoke artifacts. Each epoch saves optimizer, scalers, sampler and RNG
states. Relaunch the same command to resume; changed code/config needs new gates.

Outputs live under `runs/partner_gru/<fingerprint>/{smoke,full}`; data caches under
`staging/partner_gru/<fingerprint>`. Completed phases save machine-readable
`results/partner_gru_{smoke,full}.json` and per-zone CSVs. Full completion generates
`docs/partner_gru_results.md`, including MAE/RMSE, weekly rMAE and learning curves.
Saved predictions are checked against canonical raw targets, reconstructed scores
and reloaded-checkpoint predictions. Full target signatures must match the
existing 49-market evaluation. Large models remain ignored by Git.

The host tmux session exits normally when its command ends. Check the launcher's
`exit.json` and final log lines to distinguish completion from failure. This run
has no automatic upload or Git commit.

## ROCm backend correction

The initial fused-RNN run (`881d84...`) completed training but failed the final
replay check. An independent reference then showed a larger CPU/MIOpen GRU
discrepancy on actual 168-step inputs. The native GPU path agrees with CPU;
`results/partner_gru_backend_audit.json` records the same-weight comparison.
The original run is rejected as evidence for the intended GRU and remains
archived. A new run uses native GPU RNN operations (`cudnn.enabled=False` around
the GRU, which disables MIOpen's fused RNN on ROCm). This is still GPU training.
Both preflight and actual-input evaluation now require CPU float64 agreement.
The data, architecture, objective and seed remain the same; source/configuration
fingerprints distinguish the replacement run. Earlier GRU experiments require
separate backend checks; static MLP/GAT results do not use this RNN path.

On two validation batches, the original fused path differed from CPU float64 by
up to 35.39 and 41.68 EUR/MWh; native GPU differed by at most 0.000053 EUR/MWh.
These comparisons use identical checkpoint weights and input tensors. They
establish a discrepancy for this configuration, not its underlying library cause
or a finding about every earlier GRU checkpoint. Reproduce with:

```bash
.venv/bin/python -u scripts/experiments/check_gru_backend.py
```

The replacement generation is
`9b5aaae786ce06f14abf0b82c96882638fa5a306f9a9ec064db16a2b23b235aa`.
Its France smoke and full evaluation passed checkpoint replay and actual-input
CPU checks. The full run finished on 9 November 2026 after 37 epochs, selecting
epoch 29. See the [completed results](partner_gru_results.md) and
[comparison with earlier experiments](partner_gru_comparison.md). The original
rejected generation remains excluded.
