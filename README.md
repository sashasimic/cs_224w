# Weather, Interconnections, and Energy Prices

A spatio-temporal graph model of European day-ahead electricity markets. Stanford CS224W, Fall 2026, Project 1.

**Task.** Predict the next day's 24 hourly day-ahead prices for ~38 European bidding zones using only information available at noon the day before. Zones are nodes in a graph connected by interconnectors (and, later, by flow-based transmission constraints). See [docs/milestone-plan.md](docs/milestone-plan.md) for the current plan.

## Repository layout

```
cs_224w/
├── configs/
│   └── datasets.json        # which dataset versions the code expects (tracked)
├── scripts/
│   ├── datasets.py          # pull / push / status for versioned datasets
│   ├── build_jao_cnec_hourly.py  # JAO bundle -> constraint-node table (jao_cnec_hourly)
│   ├── build_agg_zones_hourly.py       # prices + load + weather -> one row per (zone, hour)
│   └── build_agg_zone_edges_hourly.py  # physical flows -> one row per (zone_from, zone_to, hour)
├── epgnn/
│   ├── paths.py             # dataset_path(name, file): resolves datasets/<stem>_<version>.<ext>
│   ├── jao.py               # shared JAO rules: synthetic-name pattern, zone codes, neighbours
│   ├── zones.py             # canonical zone codes and the ENTSO-E name -> zone mapping (German splice)
│   └── agg.py               # helpers for the agg_* build scripts (provenance table from the manifest)
├── datasets/                # pulled from GCS: data files AND their docs,
│                            #   e.g. weather_hourly_v1.1.parquet (gitignored), weather_hourly_v1.1.md (tracked)
├── staging/                 # drop files here to push a new dataset version (gitignored)
├── docs/
│   ├── milestone-plan.md    # what we build for the 11/10 milestone, and why
│   └── jao.md               # background on the JAO flow-based constraint data
└── lectures/                # course notes and the proposal (gitignored, course material)
```

## Setup

```bash
# Python deps
pip install pandas pyarrow google-cloud-storage torch torch_geometric

# Google Cloud SDK, then log in with the Google account that has bucket access
brew install --cask google-cloud-sdk          # macOS; cloud.google.com/sdk for others
gcloud auth login
gcloud auth application-default login         # credentials for the Python client

# Fetch the data
scripts/datasets.py pull
```

If `pull` fails with a permission error, your account has not been added to the bucket yet; ask Sasha.

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
├── entsoe_prices_v1.1.parquet        entsoe_prices_v1.1.md
├── entsoe_load_forecast_v1.1.parquet entsoe_load_forecast_v1.1.md
├── entsoe_physical_flows_v1.1.parquet entsoe_physical_flows_v1.1.md
└── weather_hourly_v1.1.parquet                   weather_hourly_v1.1.md
```

Code never hardcodes these paths. It asks the manifest, using the version-free name:

```python
from epgnn.paths import dataset_path
prices = pd.read_parquet(dataset_path("entsoe_prices", "entsoe_prices.parquet"))
# -> datasets/entsoe_prices_v1.1.parquet
```

so bumping a version in the manifest is the only change needed. `dataset_path` raises if the file isn't pulled or isn't part of that dataset.

| Dataset | Version | Contents | Doc |
|---|---|---|---|
| `entsoe_prices` | v1.2 | hourly day-ahead prices, 62 areas, Jan 2016 – Oct 2026 | [datasets/entsoe_prices_v1.2.md](datasets/entsoe_prices_v1.2.md) |
| `entsoe_load_forecast` | v1.2 | hourly day-ahead load forecast, 68 areas, Jan 2016 – Oct 2026 | [datasets/entsoe_load_forecast_v1.2.md](datasets/entsoe_load_forecast_v1.2.md) |
| `entsoe_physical_flows` | v1.2 | hourly cross-border flows, 426 directed pairs, Jan 2016 – Oct 2026 | [datasets/entsoe_physical_flows_v1.2.md](datasets/entsoe_physical_flows_v1.2.md) |
| `weather_hourly` | v1.1 | PECD/ERA5 hourly weather per bidding zone, 45 zones, 2010 – Sep 2026 | [datasets/weather_hourly_v1.1.md](datasets/weather_hourly_v1.1.md) |
| `agg_zones_hourly` | v1.0 | **derived**: prices + load forecast + weather per (zone, hour), 44 zones, Jan 2016 – Oct 2026 | [datasets/agg_zones_hourly_v1.0.md](datasets/agg_zones_hourly_v1.0.md) |
| `agg_zone_edges_hourly` | v1.0 | **derived**: physical flows per (zone_from, zone_to, hour), 174 directed pairs = 87 interconnectors | [datasets/agg_zone_edges_hourly_v1.0.md](datasets/agg_zone_edges_hourly_v1.0.md) |
| `jao_cnec_hourly` | v1.0 | JAO physical constraints per hour with RAM, Fmax and PTDF per zone (constraint nodes), May 2015 – Oct 2026 | [datasets/jao_cnec_hourly_v1.0.md](datasets/jao_cnec_hourly_v1.0.md) |

The three tables the model reads are `agg_zones_hourly` (zone nodes), `agg_zone_edges_hourly` (zone-zone edges) and `jao_cnec_hourly` (constraint nodes with PTDF edges), all keyed on the same zone codes and UTC hours. The `agg_*` tables are built from the pinned source datasets by `scripts/build_agg_*.py`; their docs list the source versions and checksums and the rules used. `jao_cnec_hourly` is the constraint-node table the proposal's graph needs. It is derived from Arsen's raw JAO bundle by [scripts/build_jao_cnec_hourly.py](scripts/build_jao_cnec_hourly.py); the rules are in its doc. Background on the raw data: [docs/jao.md](docs/jao.md).

### Pulling data

```bash
scripts/datasets.py pull                  # every dataset at the version the manifest pins
scripts/datasets.py pull weather_hourly   # one dataset, pinned version
scripts/datasets.py pull weather_hourly@v1.1  # one dataset, exact version; manifest untouched
scripts/datasets.py pull --latest         # newest version in the bucket (may be newer than pinned)
scripts/datasets.py pull --latest --pin   # ...and write those versions into the manifest
scripts/datasets.py pull --prune          # also delete local copies of other versions
scripts/datasets.py clean                 # delete datasets/ files not at a pinned version, and staging dirs already pushed (-n to preview)
scripts/datasets.py status                # local vs manifest vs bucket
```

`pull` never re-downloads a file that is already on disk with the expected size, so it costs nothing to run after every `git pull`: if someone bumped a version in the manifest you get the new file, otherwise it's a no-op. Pass `--verify` to MD5-check local files too (slow, reads every file). Downloads are always MD5-checked.

Only one version of each dataset is pulled at a time. Pulling a different version adds a second local file (`weather_hourly_v1.2.parquet` next to `_v1.1`). `clean` deletes every local file that isn't at the manifest's pinned version; `pull --prune` does the same for just the dataset being pulled.

### Pushing a new version

When you have a new build of a dataset, for example weather at per-bidding-zone resolution:

```bash
# 1. stage the data file(s) and the doc, with unversioned names
staging/weather_hourly/
├── weather_hourly.parquet
└── weather_hourly.md     # start from datasets/weather_hourly_v1.1.md; add a version-history row:
                          #   | v1.2 | <what changed> |
# 2. push
scripts/datasets.py push weather_hourly v1.2
# 3. commit the manifest and the versioned doc copy that push wrote
git add configs/datasets.json datasets/weather_hourly_v1.2.md
git commit -m "weather_hourly v1.2: <what changed>"
git push
```

What `push` does:

1. **Refuses unless `staging/weather_hourly/weather_hourly.md` exists and has a history row for the new version.** This is the only thing you write by hand: one line saying what changed. Without it the push stops before uploading anything. For a brand-new dataset it prints a doc template to start from.
2. Rewrites the mechanical header lines of the doc (version, bucket path, local filenames) so they can't go stale.
3. Uploads everything in `staging/weather_hourly/` to `weather_hourly/v1.2/`. It won't overwrite a version that already exists (`--force` to override; object versioning on the bucket means even that is recoverable).
4. Updates `configs/datasets.json` with the new version, checksums and sizes.
5. Copies the staged files into `datasets/` under their versioned names, so you don't re-download what you just uploaded. The data file is gitignored; the doc copy (`datasets/weather_hourly_v1.2.md`) is tracked, so commit it with the manifest. `scripts/datasets.py clean` then removes `staging/weather_hourly/`, since its files now match the pinned version; a staging dir that hasn't been pushed is never deleted.
6. Teammates get the new data and doc on their next `git pull && scripts/datasets.py pull`.

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
