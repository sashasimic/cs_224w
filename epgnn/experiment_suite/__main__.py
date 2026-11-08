"""python -m epgnn.experiment_suite {gpu-smoke,france,full,status}"""
import argparse
import json

from epgnn.dataset_build.storage import ROOT
from .runner import DEFAULT_CONFIG, gpu_smoke, run, status

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('command', choices=['gpu-smoke', 'france', 'full', 'status'])
parser.add_argument('--config', default=str(DEFAULT_CONFIG))
args = parser.parse_args()
if args.command == 'status':
    status(args.config)
elif args.command == 'gpu-smoke':
    gpu_smoke(json.loads(open(args.config).read()), ROOT / 'runs/vibing_gpu_smoke')
else:
    run(args.config, smoke=args.command == 'france')
