"""Run one experiment from a YAML config and write its results.

    python -m epgnn.run configs/experiments/c1_persistence.yaml

Config:
  name          run id (defaults to the file stem)
  model         persistence | xgb | gat | hetero
  fit_splits    e.g. [pretrain, train]      (ignored by persistence)
  eval_splits   e.g. [val, test]
  zones         optional subset; default all zones in the table
  ...           model-specific keys, see the model module docstrings

Output: results/<name>.json with metrics per eval split and slice, the config, the pinned dataset
versions and the git commit, plus runs/<name>/pred_<split>.npy with the predictions.
"""
import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from epgnn import metrics
from epgnn.data import samples as S
from epgnn.models import persistence
from epgnn.paths import ROOT, manifest


def main(path: str) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    name = cfg.get("name") or Path(path).stem
    model = importlib.import_module(f"epgnn.models.{cfg['model']}")
    preds = model.run(cfg, cfg.get("fit_splits", []), cfg["eval_splits"])

    out_dir = ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for split, yhat in preds.items():
        s = S.build(split, cfg.get("zones"))
        results[split] = metrics.score(s, yhat, persistence.predict(s))
        np.save(out_dir / f"pred_{split}.npy", yhat)

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    record = {"name": name, "config": cfg, "metrics": results,
              "datasets": {k: v["version"] for k, v in manifest()["datasets"].items()},
              "git": commit, "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    res = ROOT / "results" / f"{name}.json"
    res.parent.mkdir(exist_ok=True)
    res.write_text(json.dumps(record, indent=2))
    for split, r in results.items():
        a = r["all"]
        print(f"{name:28s} {split:5s} mae={a['mae']:.2f} rmse={a['rmse']:.2f} rel_mae={a['rel_mae']:.3f} "
              f"spread_mae={a.get('spread_mae', float('nan')):.2f} n_hours={a['n_hours']:,}")
    print(f"-> {res.relative_to(ROOT)}")
    return record


if __name__ == "__main__":
    main(sys.argv[1])
