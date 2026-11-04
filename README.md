# Weather, Interconnection, and Prices

A spatio-temporal graph model of European day-ahead electricity markets. Stanford CS224W, Fall 2026, Project 1.

Sophie-An Kingsbury Lee · Sasha Simic · Arsen Shebzukhov

**Task.** Predict the next day's 24 hourly day-ahead prices for ~38 European bidding zones using only information available at noon the day before. Zones are nodes in a graph connected by interconnectors (and, later, by flow-based transmission constraints). See [docs/milestone-plan.md](docs/milestone-plan.md) for the current plan.

## Repository layout

```
cs_224w/
├── configs/
│   └── datasets.json        # which dataset versions the code expects (tracked)
├── scripts/
│   └── datasets.py          # pull / push / status for versioned datasets
├── datasets/                # parquet files, pulled from GCS (gitignored)
├── docs/
│   ├── milestone-plan.md    # what we build for the 11/10 milestone, and why
│   └── datasets/            # one doc per dataset: schema, quality findings, caveats
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

The repo never commits the files. It commits [configs/datasets.json](configs/datasets.json), which says which version of each dataset the code expects and the checksum of each file. **A commit pins a manifest; the manifest pins the data.** Old versions are never deleted, so checking out any commit and running `pull` reproduces the data that commit was built on.

| Dataset | Version | Contents | Doc |
|---|---|---|---|
| `entsoe_prices` | v1.1 | hourly day-ahead prices, 61 areas, Jan 2016 – Sep 2025 | [docs/datasets/data-energy-prices.md](docs/datasets/data-energy-prices.md) |
| `entsoe_load_forecast` | v1.1 | hourly day-ahead load forecast, 68 areas | [docs/datasets/data-load-forecast.md](docs/datasets/data-load-forecast.md) |
| `entsoe_physical_flows` | v1.1 | hourly cross-border flows, 426 directed pairs | [docs/datasets/data-physical-flows.md](docs/datasets/data-physical-flows.md) |
| `weather` | v1.0 | PECD/ERA5 hourly and daily weather, 38 country codes, 1991 – 2026 | [docs/datasets/data-weather-hazards.md](docs/datasets/data-weather-hazards.md) |

JAO flow-based constraint data is documented in [docs/datasets/jao-dataset-guide.md](docs/datasets/jao-dataset-guide.md) but not yet in the bucket.

### Pulling data

```bash
scripts/datasets.py pull              # everything in the manifest → datasets/
scripts/datasets.py pull weather      # one dataset
scripts/datasets.py status            # local vs manifest vs bucket; lists every version in the bucket
```

`pull` skips files already present with the right checksum, so run it after every `git pull`; if someone bumped a version in the manifest, you get the new file, otherwise it's a no-op.

### Pushing a new version

When you have a new build of a dataset, for example weather at per-bidding-zone resolution:

```bash
scripts/datasets.py push weather v1.1 path/to/weather_hourly.parquet path/to/weather_daily.parquet
git add configs/datasets.json
git commit -m "weather v1.1: per-bidding-zone resolution"
git push
```

What happens:

1. Files upload to `weather/v1.1/`. The script refuses to overwrite a version that already exists (`--force` to override; object versioning on the bucket means even that is recoverable).
2. `configs/datasets.json` is updated with the new version and checksums.
3. Teammates get the new data on their next `git pull && scripts/datasets.py pull`.

Conventions:

- Filenames carry no version (`weather_hourly.parquet`, not `weather_hourly_v1.1.parquet`). The version lives in the bucket path and the manifest, so bumping a version never changes a path the code reads.
- Version strings are `vMAJOR.MINOR`. Bump MINOR for fixes to the same source (masking, patching); bump MAJOR for a change in scope or schema (new zones, new columns, different time range).
- Write down what changed in the dataset's doc under `docs/datasets/` in the same commit.
- Pushing needs `roles/storage.objectAdmin` on the bucket; pulling needs `roles/storage.objectViewer`.

### Giving someone access

Bucket owner, once per person:

```bash
gcloud storage buckets add-iam-policy-binding gs://cs224w-epgnn-datasets \
  --member=user:THEIR_EMAIL --role=roles/storage.objectViewer     # or objectAdmin to allow push
```
