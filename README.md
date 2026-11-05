# Weather, Interconnections, and Energy Prices

A spatio-temporal graph model of European day-ahead electricity markets. Stanford CS224W, Fall 2026, Project 1.

**Task.** Predict the next day's 24 hourly day-ahead prices for ~38 European bidding zones using only information available at noon the day before. Zones are nodes in a graph connected by interconnectors (and, later, by flow-based transmission constraints). See [docs/milestone-plan.md](docs/milestone-plan.md) for the current plan.

## Repository layout

```
cs_224w/
├── configs/
│   └── datasets.json        # which dataset versions the code expects (tracked)
├── scripts/
│   └── datasets.py          # pull / push / status for versioned datasets
├── epgnn/
│   └── paths.py             # dataset_path(name, file): resolves datasets/<stem>_<version>.<ext>
├── datasets/                # pulled from GCS: data files AND their docs,
│                            #   e.g. weather_hourly_v1.0.parquet (gitignored), weather_v1.0.md (tracked)
├── staging/                 # drop files here to push a new dataset version (gitignored)
├── docs/
│   ├── milestone-plan.md    # what we build for the 11/10 milestone, and why
│   └── jao.md               # guide to the JAO constraint data (not yet a dataset)
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
├── electricity_prices_hourly_v1.1.parquet        entsoe_prices_v1.1.md
├── electricity_load_forecasts_hourly_v1.1.parquet entsoe_load_forecast_v1.1.md
├── electricity_physical_flows_hourly_v1.1.parquet entsoe_physical_flows_v1.1.md
├── weather_hourly_v1.0.parquet                   weather_v1.0.md
└── weather_daily_v1.0.parquet
```

Code never hardcodes these paths. It asks the manifest, using the version-free name:

```python
from epgnn.paths import dataset_path
prices = pd.read_parquet(dataset_path("entsoe_prices", "electricity_prices_hourly.parquet"))
# -> datasets/electricity_prices_hourly_v1.1.parquet
```

so bumping a version in the manifest is the only change needed. `dataset_path` raises if the file isn't pulled or isn't part of that dataset.

| Dataset | Version | Contents | Doc |
|---|---|---|---|
| `entsoe_prices` | v1.1 | hourly day-ahead prices, 61 areas, Jan 2016 – Sep 2025 | [datasets/entsoe_prices_v1.1.md](datasets/entsoe_prices_v1.1.md) |
| `entsoe_load_forecast` | v1.1 | hourly day-ahead load forecast, 68 areas | [datasets/entsoe_load_forecast_v1.1.md](datasets/entsoe_load_forecast_v1.1.md) |
| `entsoe_physical_flows` | v1.1 | hourly cross-border flows, 426 directed pairs | [datasets/entsoe_physical_flows_v1.1.md](datasets/entsoe_physical_flows_v1.1.md) |
| `weather` | v1.0 | PECD/ERA5 hourly and daily weather, 38 country codes, 1991 – 2026 | [datasets/weather_v1.0.md](datasets/weather_v1.0.md) |

JAO flow-based constraint data is described in [docs/jao.md](docs/jao.md) but is not yet a dataset in the bucket.

### Pulling data

```bash
scripts/datasets.py pull                  # every dataset at the version the manifest pins
scripts/datasets.py pull weather          # one dataset, pinned version
scripts/datasets.py pull weather@v1.0     # one dataset, exact version; manifest untouched
scripts/datasets.py pull --latest         # newest version in the bucket (may be newer than pinned)
scripts/datasets.py pull --latest --pin   # ...and write those versions into the manifest
scripts/datasets.py pull --prune          # also delete local copies of other versions
scripts/datasets.py status                # local vs manifest vs bucket
```

`pull` never re-downloads a file that is already on disk with the expected size, so it costs nothing to run after every `git pull`: if someone bumped a version in the manifest you get the new file, otherwise it's a no-op. Pass `--verify` to MD5-check local files too (slow, reads every file). Downloads are always MD5-checked.

Only one version of each dataset is pulled at a time. Pulling a different version adds a second local file (`weather_hourly_v1.1.parquet` next to `_v1.0`); `--prune` removes the one the command didn't ask for.

### Pushing a new version

When you have a new build of a dataset, for example weather at per-bidding-zone resolution:

```bash
# 1. stage the data file(s) and the doc, with unversioned names
staging/weather/
├── weather_hourly.parquet
├── weather_daily.parquet
└── weather.md            # start from datasets/weather_v1.0.md; add a version-history row:
                          #   | v1.1 | per-bidding-zone resolution for DK, IT, NO, SE |
# 2. push
scripts/datasets.py push weather v1.1
# 3. commit the manifest and the versioned doc copy that push wrote
git add configs/datasets.json datasets/weather_v1.1.md
git commit -m "weather v1.1: per-bidding-zone resolution"
git push
```

What `push` does:

1. **Refuses unless `staging/weather/weather.md` exists and has a history row for the new version.** This is the only thing you write by hand: one line saying what changed. Without it the push stops before uploading anything. For a brand-new dataset it prints a doc template to start from.
2. Rewrites the mechanical header lines of the doc (version, bucket path, local filenames) so they can't go stale.
3. Uploads everything in `staging/weather/` to `weather/v1.1/`. It won't overwrite a version that already exists (`--force` to override; object versioning on the bucket means even that is recoverable).
4. Updates `configs/datasets.json` with the new version, checksums and sizes.
5. Copies the staged files into `datasets/` under their versioned names, so you don't re-download what you just uploaded. The data file is gitignored; the doc copy (`datasets/weather_v1.1.md`) is tracked, so commit it with the manifest. `staging/weather/` can then be deleted.
6. Teammates get the new data and doc on their next `git pull && scripts/datasets.py pull`.

Conventions:

- Filenames in the bucket, manifest and staging carry no version (`weather_hourly.parquet`); `pull` appends it locally (`weather_hourly_v1.0.parquet`). Code goes through `dataset_path`, so bumping a version never changes a line of code.
- Version strings are `vMAJOR.MINOR`. Bump MINOR for fixes to the same source (masking, patching); bump MAJOR for a change in scope or schema (new zones, new columns, different time range).
- The doc is named `<dataset id>.md` and is part of every version: a header (maintained by `push`), a version-history table (maintained by you, cumulative), then schema, quality findings and caveats.
- Pushing needs `roles/storage.objectAdmin` on the bucket; pulling needs `roles/storage.objectViewer`.

### Giving someone access

Bucket owner, once per person:

```bash
gcloud storage buckets add-iam-policy-binding gs://cs224w-epgnn-datasets \
  --member=user:THEIR_EMAIL --role=roles/storage.objectViewer     # or objectAdmin to allow push
```
