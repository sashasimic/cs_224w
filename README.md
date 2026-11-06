# Weather, Interconnections, and Energy Prices

A spatio-temporal graph model of European day-ahead electricity markets. Stanford CS224W, Fall 2026, Project 1.

**Task.** Predict the next day's 24 hourly day-ahead prices for European bidding zones using only information available at noon the day before (gate closure). The graph has two node types: **zones** (price, load-forecast and weather history), connected by **interconnector edges** (physical flows), and **transmission constraints** from the JAO flow-based domain, each connected to the zones that load it with the PTDF as edge weight. Hypotheses: H1 the graph beats neighbours-as-features; H2 published constraints improve cross-border spread forecasts; H3 heat stress raises prices and spreads most under stress.

## Current state (6 Nov 2026)

- **Data is in place.** ENTSO-E prices, load forecasts and physical flows (Jan 2016 – Oct 2026), per-zone PECD/ERA5 weather (to Sep 2026) and JAO flow-based constraints (CWE + Core, May 2015 – Oct 2026) are cleaned, documented and versioned in the bucket (see *Data*).
- **Model tables.** `zone_nodes_hourly` and `zone_edges_hourly` are the per-(zone, hour) and per-(border, hour) inputs the model reads; `jao_cnec_hourly` supplies the constraint nodes. All three share the same zone codes (`DE_LU`, `NO_1`, `IT_NORD`, ...) and UTC hours.
- **Split** ([configs/splits.json](configs/splits.json)): pretrain 2016-01-01 – 2022-06-08 (no Core constraints yet), train 2022-06-09 – 2024-03-31, validation 2024-04-01 – 2025-03-31, test 2025-04-01 – 2026-09-30 (two summers). Tables stay whole; loaders filter by date.
- **Decided:** constraint nodes carry RAM and Fmax from the published JAO domain. Zone ids are the weather/JAO codes. Germany before Oct 2018 is the joint DE-AT-LU zone, reported under `DE_LU`.
- **Open:** ENTSO-E wind/solar day-ahead forecasts and transfer capacities (NTC) are not pulled yet (weather capacity factors and JAO `maxbex` stand in); whether the detailed JAO domain is published before gate closure is unverified (fallback: use day D-1's constraints); whether to use CWE-era constraints in training or treat 2016-22 as constraint-free pretraining.
- **Code:** the basic structure is in place and runs end to end with one super simple, non-trainable model: weekly persistence (`c1_persistence`, forecast = same hour last week). It exercises the whole chain (samples, cutoff, masks, metrics, results file) and sets the reference score (test MAE 29.0 EUR/MWh). No trained model exists yet: XGBoost, GATv2 and the hetero model are stubs with fixed interfaces, one per teammate (see *Experiments*).

## Repository layout

```
cs_224w/
├── configs/
│   ├── datasets.json        # which dataset versions the code expects (tracked)
│   ├── splits.json          # the chronological split: pretrain / train / val / test by delivery day
│   └── experiments/         # one YAML per run: c1_persistence, c1_xgb_zone, c2_xgb_neighbours,
│                            #   c3_gat_interconnects, c3_gat_shuffled, c4a_gat_jao_summaries,
│                            #   c4b_hetero_constraints, c4b_hetero_heat
├── epgnn/                   # the package ("electricity price GNN")
│   ├── paths.py             # dataset_path(name, file): resolves datasets/<stem>_<version>.<ext>
│   ├── splits.py            # reads configs/splits.json; split_of(delivery_day)
│   ├── zones.py, jao.py     # zone codes, ENTSO-E mapping, JAO rules (used by the build scripts)
│   ├── data/                # shared data layer: every model reads through here
│   │   ├── tables.py        #   load the pinned tables, cached
│   │   ├── features.py      #   named feature sets + the gate-closure rule (what day D may see)
│   │   ├── samples.py       #   (zone, day) samples: 168 h history to the cutoff, day-D inputs, 24 targets + mask
│   │   ├── scaling.py       #   asinh / residual-vs-last-week target, train-only z-scores
│   │   └── graph.py         #   zone graph + edge features; constraint nodes + PTDF relations (4b)
│   ├── models/              # one module per model family, same run(cfg, fit_splits, eval_splits) interface
│   │   ├── persistence.py   #   1a  (done)
│   ├── gru_mlp.py       #   2.5 GRU + MLP, no graph (done)
│   │   ├── xgb.py           #   1b, 2
│   │   ├── encoders.py      #   HistoryGRU shared by gat and hetero
│   │   ├── gat.py           #   3, 3', 4a
│   │   └── hetero.py        #   4b
│   ├── train.py             # masked + spread loss, training loop for the torch models
│   ├── metrics.py           # MAE, RMSE, rel-MAE vs persistence, spread MAE / F1, slices
│   └── run.py               # python -m epgnn.run configs/experiments/<name>.yaml
├── scripts/
│   └── datasets/            # dataset tools, run by hand; nothing here is model code
│       ├── datasets.py      #   pull / push / status / clean for versioned datasets
│       ├── build_jao_cnec_hourly.py        #   JAO bundle -> constraint-node table (jao_cnec_hourly)
│       ├── build_agg_zones_hourly.py       #   prices + load + weather -> one row per (zone, hour)
│       └── build_agg_zone_edges_hourly.py  #   physical flows -> one row per (zone_from, zone_to, hour)
├── results/                 # one JSON per run: metrics by split and slice, config, data versions, commit (tracked)
├── runs/                    # predictions and checkpoints per run (gitignored)
├── datasets/                # pulled from GCS: data files AND their docs,
│                            #   e.g. zone_nodes_hourly_v1.0.parquet (gitignored), zone_nodes_hourly_v1.0.md (tracked)
├── staging/                 # drop files here to push a new dataset version (gitignored)
├── tmp/                     # local scratch (gitignored)
└── lectures/                # course notes and the proposal (gitignored, course material)
```

## Setup

```bash
# Python deps
pip install -e ".[model]"            # pandas, pyarrow, GCS client, torch, torch_geometric, xgboost, ...

# Google Cloud SDK, then log in with the Google account that has bucket access
brew install --cask google-cloud-sdk          # macOS; cloud.google.com/sdk for others
gcloud auth login
gcloud auth application-default login         # credentials for the Python client

# Fetch the data
scripts/datasets/datasets.py pull
```

If `pull` fails with a permission error, your account has not been added to the bucket yet; ask Sasha.

## Experiments

The structure is in place; only the non-trainable persistence reference runs today. A run is a config, not a branch. Every YAML in `configs/experiments/` names a model, a feature set and the splits; the checkpoints of the Methods doc are these files:

| Config | Checkpoint | Model | Status |
|---|---|---|---|
| `c1_persistence` | 1a | price one week earlier | done |
| `c1_xgb_zone` | 1b | per-zone XGBoost, own features | done: test MAE 21.2, rel 0.73 |
| `c2_xgb_neighbours` | 2 | + neighbours' features | done: test MAE 21.0, rel 0.73 |
| `c25_gru_mlp` | 2.5 | GRU + MLP per zone, no graph (the no-edge case of 3) | done: test MAE 20.4, rel 0.70, spread 15.1 |
| `c3_gat_interconnects` | 3 | GRU + GATv2 on the interconnector graph | to do |
| `c3_gat_shuffled` | 3' | same, shuffled graph (control) | to do |
| `c4a_gat_jao_summaries` | 4a | 3 + per-zone JAO summaries | to do |
| `c4b_hetero_constraints` | 4b | + constraint nodes and PTDF relations | to do (needs `jao_cnec_hourly` with `cnec_id`) |
| `c4b_hetero_heat` | 4b + H3 | + heat-stress features | to do |

```bash
python -m epgnn.run configs/experiments/c1_persistence.yaml
# -> results/c1_persistence.json, runs/c1_persistence/pred_{val,test}.npy
```

Rules that keep the three model families comparable:

- **Same samples, same score.** `epgnn/data/samples.py` defines the (zone, delivery day) samples per split and the targets; `epgnn/metrics.py` scores every run against weekly persistence on exactly those hours. Model code never builds its own test set.
- **Leakage rule lives in one file.** `epgnn/data/features.py` lists which columns day D may see (lags, load forecast, calendar, JAO domain) and which are outcomes. Models ask for a feature set by name.
- **Cutoff.** History windows end at D-1 10:00 UTC (noon CEST); day-D inputs are only the columns allowed by the feature set.
- **Results are files.** `results/<run>.json` records metrics by split and slice, the config, the pinned dataset versions and the commit; report tables are generated from that folder.

Ownership for the milestone: A owns `data/` + `metrics.py` + `models/xgb.py` (checkpoint 1); B `models/encoders.py` + `models/gat.py` + `train.py` (checkpoint 3); C `data/graph.py` constraint part + `models/hetero.py` (checkpoint 4b).

## Data

The parquet files live in `gs://cs224w-epgnn-datasets` (GCP project `cs224w-epgnn-2026`), versioned by path:

```
gs://cs224w-epgnn-datasets/<dataset>/<version>/<file>
```

A dataset version is its data file(s) **plus its doc**, `<dataset>.md`: schema, quality findings, caveats, version history. They are stored together and pulled together, so the doc you read always describes the data you have.

The repo never commits the data files. It commits [configs/datasets.json](configs/datasets.json), which says which version of each dataset the code expects and the checksum of each file, and the docs (`datasets/*.md`) so they can be read on GitHub. **A commit pins a manifest; the manifest pins the data.** Old versions are never deleted, so checking out any commit and running `pull` reproduces the data that commit was built on.

Locally the files are flat with the version appended, so `ls datasets/` shows which versions you have:

```
datasets/
├── zone_nodes_hourly_v1.0.parquet   zone_nodes_hourly_v1.0.md
└── zone_edges_hourly_v1.0.parquet   zone_edges_hourly_v1.0.md
```

Code never hardcodes these paths. It asks the manifest, using the version-free name:

```python
from epgnn.paths import dataset_path
nodes = pd.read_parquet(dataset_path("zone_nodes_hourly", "zone_nodes_hourly.parquet"))
# -> datasets/zone_nodes_hourly_v1.0.parquet
```

so bumping a version in the manifest is the only change needed. `dataset_path` raises if the file isn't pulled or isn't part of that dataset.

| Dataset | Version | Contents | Doc |
|---|---|---|---|
| `zone_nodes_hourly` | v1.0 | per (zone, hour): day-ahead price (target), price lags, load forecast, weather (+ lags), net import, JAO Core summaries, `split`; 45 zones, Jan 2016 – Sep 2026 | [datasets/zone_nodes_hourly_v1.0.md](datasets/zone_nodes_hourly_v1.0.md) |
| `zone_edges_hourly` | v1.0 | per (from, to, hour): physical flow + lags, JAO `maxbex` / scheduled flow, `split`; 180 directed interconnectors | [datasets/zone_edges_hourly_v1.0.md](datasets/zone_edges_hourly_v1.0.md) |

These two are the model's zone-node and zone-edge tables (Sophie-An's build). Constraint nodes for the full hetero graph come from `jao_cnec_hourly` (below). The split is defined once in [configs/splits.json](configs/splits.json) (read by `epgnn/splits.py`); tables are kept whole and filtered by date, there are no per-split files.

**Other datasets in the bucket, not pinned.** The sources and intermediate tables stay in `gs://cs224w-epgnn-datasets` but are not pulled by default. Pull one explicitly with `scripts/datasets/datasets.py pull NAME@VERSION` (add `--pin` to put it in the manifest so `dataset_path` and the build scripts can use it); `status` lists them.

| Dataset | Latest | Contents |
|---|---|---|
| `entsoe_prices` | v1.2 | hourly day-ahead prices, 62 areas, Jan 2016 – Oct 2026 |
| `entsoe_load_forecast` | v1.2 | hourly day-ahead load forecast, 68 areas, Jan 2016 – Oct 2026 |
| `entsoe_physical_flows` | v1.2 | hourly cross-border flows, 426 directed pairs, Jan 2016 – Oct 2026 |
| `weather_hourly` | v1.1 | PECD/ERA5 hourly weather per bidding zone, 45 zones, 2010 – Sep 2026 |
| `jao_cnec_hourly` | v1.0 | JAO physical constraints per hour with RAM, Fmax and PTDF per zone (constraint nodes), May 2015 – Oct 2026; built from Arsen's raw bundle by [scripts/datasets/build_jao_cnec_hourly.py](scripts/datasets/build_jao_cnec_hourly.py) |
| `agg_zones_hourly` | v1.0 | our own join of prices + load + weather per (zone, hour), 44 zones; built by `scripts/datasets/build_agg_zones_hourly.py` (cross-check for `zone_nodes_hourly`) |
| `agg_zone_edges_hourly` | v1.0 | our own flows per (zone_from, zone_to, hour), 174 directed pairs; built by `scripts/datasets/build_agg_zone_edges_hourly.py` (cross-check for `zone_edges_hourly`) |

Each has a doc next to it in the bucket (`<dataset>.md`), pulled with the data; the docs are the reference for schema, cleaning rules, known gaps and how the tables were derived from each other.

### Pulling data

```bash
scripts/datasets/datasets.py pull                  # every dataset at the version the manifest pins
scripts/datasets/datasets.py pull zone_nodes_hourly   # one dataset, pinned version
scripts/datasets/datasets.py pull entsoe_prices@v1.2  # any dataset in the bucket, exact version; manifest untouched
scripts/datasets/datasets.py pull entsoe_prices@v1.2 --pin  # ...and pin it, so dataset_path() can resolve it
scripts/datasets/datasets.py pull --latest         # newest version in the bucket (may be newer than pinned)
scripts/datasets/datasets.py pull --latest --pin   # ...and write those versions into the manifest
scripts/datasets/datasets.py pull --prune          # also delete local copies of other versions
scripts/datasets/datasets.py clean                 # delete datasets/ files not at a pinned version, and staging dirs already pushed (-n to preview)
scripts/datasets/datasets.py status                # local vs manifest vs bucket
```

`pull` never re-downloads a file that is already on disk with the expected size, so it costs nothing to run after every `git pull`: if someone bumped a version in the manifest you get the new file, otherwise it's a no-op. Pass `--verify` to MD5-check local files too (slow, reads every file). Downloads are always MD5-checked.

Only one version of each dataset is pulled at a time. Pulling a different version adds a second local file (`zone_nodes_hourly_v1.1.parquet` next to `_v1.0`). `clean` deletes every local file that isn't at the manifest's pinned version; `pull --prune` does the same for just the dataset being pulled.

### Pushing a new version

When you have a new build of a dataset, for example a rebuilt node table:

```bash
# 1. stage the data file(s) and the doc, with unversioned names
staging/zone_nodes_hourly/
├── zone_nodes_hourly.parquet
└── zone_nodes_hourly.md  # start from datasets/zone_nodes_hourly_v1.0.md; add a version-history row:
                          #   | v1.1 | <what changed> |
# 2. push
scripts/datasets/datasets.py push zone_nodes_hourly v1.1
# 3. commit the manifest and the versioned doc copy that push wrote
git add configs/datasets.json datasets/zone_nodes_hourly_v1.1.md
git commit -m "zone_nodes_hourly v1.1: <what changed>"
git push
```

What `push` does:

1. **Refuses unless `staging/zone_nodes_hourly/zone_nodes_hourly.md` exists and has a history row for the new version.** This is the only thing you write by hand: one line saying what changed. Without it the push stops before uploading anything. For a brand-new dataset it prints a doc template to start from.
2. Rewrites the mechanical header lines of the doc (version, bucket path, local filenames) so they can't go stale.
3. Uploads everything in `staging/zone_nodes_hourly/` to `zone_nodes_hourly/v1.1/`. It won't overwrite a version that already exists (`--force` to override; object versioning on the bucket means even that is recoverable).
4. Updates `configs/datasets.json` with the new version, checksums and sizes.
5. Copies the staged files into `datasets/` under their versioned names, so you don't re-download what you just uploaded. The data file is gitignored; the doc copy (`datasets/zone_nodes_hourly_v1.1.md`) is tracked, so commit it with the manifest. `scripts/datasets/datasets.py clean` then removes `staging/zone_nodes_hourly/`, since its files now match the pinned version; a staging dir that hasn't been pushed is never deleted.
6. Teammates get the new data and doc on their next `git pull && scripts/datasets/datasets.py pull`.

Conventions:

- Files are named after their dataset: `<dataset>.md` for the doc, `<dataset>.parquet` for the main data file (granularity goes in the dataset id, e.g. `weather_hourly`). Names in the bucket, manifest and staging carry no version; `pull` appends it locally (`weather_hourly_v1.1.parquet`). Code goes through `dataset_path`, so bumping a version never changes a line of code.
- Version strings are `vMAJOR.MINOR`. Bump MINOR for fixes to the same source (masking, patching); bump MAJOR for a change in scope or schema (new zones, new columns, different time range).
- The doc is named `<dataset id>.md` and is part of every version: a header (maintained by `push`), a version-history table (maintained by you, cumulative), then schema, quality findings and caveats.
- Pushing needs `roles/storage.objectAdmin` on the bucket; pulling needs `roles/storage.objectViewer`.

### Giving someone access

Bucket owner, once per person:

```bash
gcloud storage buckets add-iam-policy-binding gs://cs224w-epgnn-datasets \
  --member=user:THEIR_EMAIL --role=roles/storage.objectViewer     # or objectAdmin to allow push
```
