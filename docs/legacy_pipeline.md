# Original pipeline and scaffold

The original YAML workflow is separate from the audited suite. Its entry point
is `python -m epgnn.run CONFIG.yaml`; it reads pinned v1.0 model tables through
`epgnn/data/`. Those tables and paths remain available for reproduction.

| Original component | Current state on `vibing` |
|---|---|
| `models/persistence.py` | Implemented original weekly baseline |
| `models/xgb.py` | Original scaffold stub; implemented audited boosting is in `boosting.py` and `experiment_suite/` |
| `models/gat.py`, `models/hetero.py` | Original scaffold stubs; the implemented hourly GAT is in `experiment_suite/neural.py` |
| `models/encoders.py` | Implemented `HistoryGRU`, used by the ROCm check; not used by the hourly suite |
| `train.py` | Shared masked loss is fixed; original `fit` scaffold remains a stub |
| `data/graph.py` | Original graph helpers; detailed constraint graph requires a supported identity definition |

The partner's implemented XGBoost is reproduced from its own pinned worktree,
not by replacing the stub on this branch. See [partner reproduction](partner_pipeline_reproduction.md).

## Data and timing differences

Original v1.0 tables splice historical DE_AT_LU prices under DE_LU and retain
upstream filled inputs. The audited master keeps those identities separate,
retains missingness and records provenance. Original legacy history construction
has timestamp-unit and fixed-UTC-cutoff issues identified in the dataset audit.
The active suite bypasses those history tensors and constructs literal Brussels
noon origins. The shared NaN-times-zero masked-loss defect was fixed when the
suite was implemented.

These differences are why the root README directs new experiments to the audited
suite. Existing source, YAML names and saved metrics remain available; no original
model/data files were deleted or silently substituted during cleanup.

For the earlier repository overview, ownership notes and GCS publishing recipes,
inspect `git show f50dc6e:README.md`. That is historical documentation rather than
the current status of the active suite.
