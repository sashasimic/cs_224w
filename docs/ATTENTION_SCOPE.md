# Attention-scope GAT (H1) and JAO-constraint arm (H2): what this branch adds

This branch starts from `c1_xgb_zone`. It adds the feature pipeline, the attention-scope GAT, the constraint arm, their configs, results and figures. **No existing file was changed** except `.gitignore`; everything else is new.

Start with [OVERVIEW.md](OVERVIEW.md): task, data, decisions, models and results. [PIPELINE_REVIEW.md](PIPELINE_REVIEW.md) is the detailed audit, with every fix and every result and its provenance. Both were written in a local layout where this repo sat in a folder called `cs_224w-c1_xgb_zone/`, so paths that start with that name point to the repo root here.

## What is new

| Path | What it is |
|---|---|
| `epgnn/pipeline/` | The feature pipeline: a registry giving every column a timing class, zone features with as-of weather, edge features, JAO constraint nodes, and the zone-to-constraint graph. Run it with `python -m epgnn.pipeline.build`. It writes to `features/v1/`; only the small docs (`feature_dictionary.csv`, `feature_summary_zone.csv`, `graph.json`, `build.json`) are committed. |
| `epgnn/models/gat_scope.py` | GRU + dense GATv2 with three attention scopes (self, interconnects, full), Laplacian positional encoding, and monthly rolling fine-tuning. |
| `epgnn/models/gat_constraints.py` | H2: JAO constraints as nodes, with signed-PTDF attention. |
| `epgnn/models/asof.py`, `epgnn/bootstrap.py` | As-of input variants of the c1/c2/c2.5 baselines, and the paired weekly block bootstrap. |
| `configs/experiments/c3s_*`, `c4s_*`, `*_asof` | Experiment configs. |
| `results/*.json` | Metrics and bootstrap comparisons. |
| `tests/` | Includes a numerical-equivalence test against PyG `GATv2Conv` and parameter-count checks. |
| `figures/` | Figure scripts and outputs: data (`d*`), JAO (`j*`), results (`r*`), model schematics (`m*`, `line/L*`). |
| `scripts/build/` | `build_model_dataset.py`, which builds the node and edge tables from ENTSO-E v1.2, weather and the Core JAO views, and `download_entsoe_ext.py`, which reads the token from `ENTSOE_API_KEY`. |

## Running it

1. **Get the data.** Pull it from the bucket with `python scripts/datasets/datasets.py pull`.
2. **Build the features.** Run `PYTHONPATH=. python -m epgnn.pipeline.build --jao-dir <path to jao_core_hourly_views>`. Add `--skip-jao` if you don't have the JAO views.
3. **Run an experiment.** Run `PYTHONPATH=. python -m epgnn.run configs/experiments/c3s_gat_interconnects_curves_rolling.yaml`.
4. **Run the tests.** Run `PYTHONPATH=. python -m pytest tests`.

The figure scripts read the model tables from `data/model/{nodes,edges}_hourly.parquet` at the repo root. That folder is gitignored.
