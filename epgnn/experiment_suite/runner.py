"""Five matched experiments: one data contract, restartable jobs, auditable scores."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from epgnn.boosting import score
from epgnn.dataset_build.storage import ROOT, canonical, digest
from . import data, neural

EXPERIMENTS = ['recent_xgb', 'rich_xgb', 'neighbour_xgb', 'real_gat', 'shuffled_gat']
DEFAULT_CONFIG = ROOT / 'configs/experiments/vibing_suite.json'


def write_json(path, value):
    """Publish status/results atomically; readers never see truncated JSON."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def identity(config):
    code = {str(p.relative_to(ROOT)): digest(p)['sha256']
            for p in sorted((ROOT / 'epgnn/experiment_suite').glob('*.py'))}
    code['epgnn/train.py'] = digest(ROOT / 'epgnn/train.py')['sha256']
    code['epgnn/boosting.py'] = digest(ROOT / 'epgnn/boosting.py')['sha256']
    value = {'config': config, 'code': code, 'versions': {'torch': torch.__version__,
        'hip': torch.version.hip, 'numpy': np.__version__, 'pandas': pd.__version__, 'xgboost': xgb.__version__}}
    return hashlib.sha256(canonical(value)).hexdigest(), value


def checked_cache(config):
    source = ROOT / config['dataset_directory']
    complete = json.loads((source / 'complete.json').read_text())
    if complete['fingerprint'] != config['dataset_fingerprint']:
        raise ValueError('Dataset differs from pinned construction')
    cache_identity = hashlib.sha256(canonical({'source': complete, 'data_code': digest(data.__file__)['sha256'],
        'boosting_code': digest(ROOT / 'epgnn/boosting.py')['sha256']})).hexdigest()
    cache = ROOT / 'staging/vibing_suite' / cache_identity
    if not (cache / 'complete.json').exists():
        temporary = cache.with_name('.' + cache.name + '.building')
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
        data.prepare(source, temporary)
        temporary.rename(cache)
    for name, sha in json.loads((cache / 'complete.json').read_text()).items():
        if digest(cache / name)['sha256'] != sha:
            raise ValueError('Cached input checksum mismatch: ' + name)
    return cache, *data.load(cache)


def gpu_smoke(config, output):
    device = neural.require_gpu()
    torch.set_num_threads(4)
    random.seed(224); np.random.seed(224); torch.manual_seed(224); torch.cuda.manual_seed_all(224)
    report = {'gpu': torch.cuda.get_device_name(), 'hip': torch.version.hip, 'torch': torch.__version__,
              'checks': [], 'peak_vram_gib': 0., 'batch_size': config['neural']['batch_size'],
              'neural_code_sha256': digest(neural.__file__)['sha256'],
              'loss_code_sha256': digest(ROOT / 'epgnn/train.py')['sha256']}
    zones = 62
    pairs = [(a, b) for a in range(zones) for b in range(zones) if a != b]
    chosen = np.random.default_rng(224).choice(len(pairs), 520, replace=False)
    real = np.array([pairs[k] for k in chosen], np.int64).T
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for shuffled, graph in [(False, real), (True, data.shuffle_edges(real))]:
        edges = torch.tensor(graph.copy(), device=device)
        batch_size = report['batch_size']
        while True:
            try:
                torch.cuda.reset_peak_memory_stats()
                prep = neural.Preprocessor().fit(np.random.default_rng(224).normal(size=(100, len(data.RICH))).astype(np.float32))
                model = neural.PriceGAT(len(data.RICH)*2).to(device)
                opt = torch.optim.Adam(model.parameters(), lr=.001)
                x = torch.randn(batch_size, zones, len(data.RICH)*2, device=device)
                active = torch.ones(batch_size, zones, dtype=torch.bool, device=device)
                active[:, -3:] = False
                y = torch.randn(batch_size, zones, device=device)
                y[::2, 0] = float('nan')
                valid = torch.isfinite(y)
                started = time.monotonic()
                for _ in range(3):
                    opt.zero_grad(set_to_none=True)
                    prediction = model(x, edges, active)
                    loss = neural.masked_loss(prediction, y, valid)
                    if not torch.isfinite(loss):
                        raise RuntimeError('Smoke loss is not finite')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise RuntimeError('Smoke gradients are missing/nonfinite')
                    opt.step()
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 2**30
                if peak > 12:
                    raise torch.OutOfMemoryError('Above 12 GiB safety budget')
                break
            except torch.OutOfMemoryError:
                if batch_size <= 8:
                    raise
                batch_size //= 2
                del model, opt, x, y
                torch.cuda.empty_cache()
        checkpoint = output / ('shuffled.pt' if shuffled else 'real.pt')
        model.eval()
        with torch.no_grad():
            before = model(x, edges, active).cpu()
        neural.atomic_checkpoint(checkpoint, {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                                             'prep': prep.state()})
        saved = torch.load(checkpoint, weights_only=False, map_location=device)
        restored = neural.PriceGAT(len(data.RICH)*2).to(device)
        restored.load_state_dict(saved['model']); restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(before, restored(x, edges, active).cpu(), rtol=1e-5, atol=1e-6)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
        restored_optimizer.load_state_dict(saved['optimizer'])
        restored_optimizer.zero_grad(set_to_none=True)
        neural.masked_loss(restored(x, edges, active), y, valid).backward()
        restored_optimizer.step()
        if not all(torch.isfinite(p).all() for p in restored.parameters()):
            raise RuntimeError('Resumed optimizer produced invalid weights')
        report['checks'].append({'graph': 'shuffled' if shuffled else 'real', 'training_steps': 3,
            'checks': ['forward', 'NaN-masked loss', 'backward', 'Adam', 'validation', 'checkpoint reload', 'optimizer resume'],
            'seconds': time.monotonic()-started, 'peak_vram_gib': peak})
        report['peak_vram_gib'] = max(report['peak_vram_gib'], peak)
        report['batch_size'] = min(report['batch_size'], batch_size)
        print('GPU smoke PASS:', report['checks'][-1], flush=True)
    # Exercise the actual epoch/checkpoint loop, not just an isolated optimizer.
    rng = np.random.default_rng(224)
    fixture = {'x': rng.normal(size=(128, 8, len(data.FEATURES))).astype(np.float32),
               'y': rng.normal(50, 10, size=(128, 8)).astype(np.float32),
               'active': np.ones((128, 8), bool),
               'split': np.array([0]*96 + [1]*16 + [2]*16),
               'edges': np.array([[0,1,2,3,4,5,6,7], [1,2,3,4,5,6,7,0]], np.int64)}
    fixture['shuffled_edges'] = data.shuffle_edges(fixture['edges'])
    fixture['y'][0, 0] = np.nan
    fixture_cfg = {**config['neural'], 'epochs': 1, 'identity': 'gpu-resume-fixture'}
    import tempfile
    with tempfile.TemporaryDirectory(dir=output) as temp:
        resumed = Path(temp) / 'resumed'
        continuous = Path(temp) / 'continuous'
        neural.train(fixture, list(range(len(data.RICH))), list(range(8)), resumed, fixture_cfg)
        fixture_cfg['epochs'] = 2
        neural.train(fixture, list(range(len(data.RICH))), list(range(8)), resumed, fixture_cfg)
        neural.train(fixture, list(range(len(data.RICH))), list(range(8)), continuous, fixture_cfg)
        actual = torch.load(resumed / 'latest.pt', map_location='cpu', weights_only=False)
        expected = torch.load(continuous / 'latest.pt', map_location='cpu', weights_only=False)
        if [c['epoch'] for c in actual['curves']] != [1, 2]:
            raise RuntimeError('Epoch resume repeated or skipped training')
        for name in actual['model']:
            torch.testing.assert_close(actual['model'][name], expected['model'][name], rtol=1e-4, atol=1e-5)
    report['epoch_resume_matches_continuous_training'] = True
    report['passed'] = True
    write_json(output / 'report.json', report)
    return report


def evaluate(predictions, arrays, meta, zones, experiment, split, directory):
    indices = np.flatnonzero(arrays['split'] == {'val': 1, 'test': 2}[split])
    if predictions.shape != (len(indices), len(zones)) or not np.isfinite(predictions).all():
        raise ValueError('Evaluation requires finite predictions for every selected zone-hour')
    times = pd.to_datetime(np.asarray(arrays['times'][indices]), utc=True)
    price_column = data.FEATURES.index('price_lag168')
    jao_column = data.FEATURES.index('jao_domain_present_lag48')
    records, keyed = [], []
    for k, j in enumerate(zones):
        y = np.asarray(arrays['y'][indices, j])
        p = np.asarray(arrays['x'][indices, j, price_column])
        domain = np.asarray(arrays['x'][indices, j, jao_column])
        f = pd.DataFrame({'zone': meta['zones'][j], 'ts_utc': times, 'price': y,
            'target_valid': np.isfinite(y), 'prediction': predictions[:, k], 'price_lag168': p,
            'jao_domain_present_lag48': domain})
        result = score(f, predictions[:, k], p)
        for coverage, selection in [('all', np.ones(len(f), bool)), ('jao_published', domain == 1)]:
            if coverage != 'all':
                result = score(f.loc[selection], predictions[selection, k], p[selection])
            common = result['weekly_persistence_conditional_common_hours']
            records.append({'experiment': experiment, 'split': split, 'zone': meta['zones'][j], 'coverage': coverage,
                'hours': result['model']['n_hours'], 'mae': result['model']['mae'], 'rmse': result['model']['rmse'],
                'paired_hours': common['model']['n_hours'], 'paired_mae': common['model']['mae'],
                'persistence_mae': common['persistence']['mae'], 'persistence_rmse': common['persistence']['rmse'],
                'relative_mae': common['relative_mae']})
        keyed.append(f)
    frame = pd.concat(keyed, ignore_index=True)
    if frame.duplicated(['zone', 'ts_utc']).any():
        raise ValueError('Duplicate prediction keys')
    frame.to_parquet(directory / f'predictions_{split}.parquet', index=False)
    return records


def grouped_scores(rows):
    frame = pd.DataFrame(rows)
    # DE_LU/IE_SEM stay combined markets; regional targets are pooled, not averaged prices.
    frame['country'] = frame.zone.map(lambda z: z.split('_')[0] if z.startswith(('IT_', 'NO_', 'SE_', 'DK_')) else z)
    result = []
    for keys in [['experiment', 'split', 'coverage'], ['experiment', 'split', 'coverage', 'country']]:
        for group, f in frame.groupby(keys):
            n, paired = int(f.hours.sum()), int(f.paired_hours.sum())
            def weighted(column, weights):
                ok = f[column].notna() & (f[weights] > 0)
                return float(np.average(f.loc[ok, column], weights=f.loc[ok, weights])) if ok.any() else None
            # Pool squared errors rather than averaging RMSEs.
            squared = f.rmse.fillna(0).to_numpy(float)**2 * f.hours
            mae, pm = weighted('paired_mae', 'paired_hours'), weighted('persistence_mae', 'paired_hours')
            result.append({**dict(zip(keys, group)), 'scope': 'overall' if len(keys) == 3 else 'country',
                'hours': n, 'paired_hours': paired, 'mae': weighted('mae', 'hours'),
                'rmse': float(np.sqrt(squared.sum()/n)) if n else None,
                'paired_mae': mae, 'persistence_mae': pm, 'relative_mae': mae/pm if pm else None})
    return result


def xgb_experiment(arrays, features, zones, directory, config):
    fit = np.flatnonzero(arrays['split'] == 0)
    val = np.flatnonzero(arrays['split'] == 1)
    evaluation = {s: np.flatnonzero(arrays['split'] == k) for s, k in [('val', 1), ('test', 2)]}
    predictions = {s: np.empty((len(v), len(zones)), np.float32) for s, v in evaluation.items()}
    selected = {}
    for k, j in enumerate(zones):
        filename = directory / f'zone_{j}.ubj'
        x = np.asarray(arrays['x'][:, j, :])[:, features]
        y = np.asarray(arrays['y'][:, j])
        if filename.exists():
            model = xgb.XGBRegressor(); model.load_model(filename)
        else:
            model = xgb.XGBRegressor(**config['xgb'], early_stopping_rounds=50)
            valid_fit, valid_val = fit[np.isfinite(y[fit])], val[np.isfinite(y[val])]
            model.fit(x[valid_fit], np.arcsinh(y[valid_fit]/50),
                      eval_set=[(x[valid_val], np.arcsinh(y[valid_val]/50))], verbose=False)
            temporary = filename.with_suffix('.tmp.ubj')
            model.save_model(temporary); temporary.rename(filename)
            reloaded = xgb.XGBRegressor(); reloaded.load_model(filename)
            np.testing.assert_array_equal(model.predict(x[valid_val[:128]]), reloaded.predict(x[valid_val[:128]]))
        selected[str(j)] = model.best_iteration+1
        for s, idx in evaluation.items():
            predictions[s][:, k] = 50 * np.sinh(model.predict(x[idx]))
        print(f'XGBoost zone index {j}: {model.best_iteration+1} rounds', flush=True)
    return predictions, {'selected_rounds': selected}


def run(config_path=DEFAULT_CONFIG, smoke=False):
    """Reject duplicate launches targeting the same experiment generation."""
    config = json.loads(Path(config_path).read_text())
    token, _ = identity(config)
    phase = 'france' if smoke else 'full'
    directory = ROOT / 'runs/vibing_suite' / token / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This suite generation is already running') from error
        return _run(config_path, smoke)


def _run(config_path, smoke):
    config = json.loads(Path(config_path).read_text())
    token, provenance = identity(config)
    root = ROOT / 'runs/vibing_suite' / token
    phase = 'france' if smoke else 'full'
    directory = root / phase; directory.mkdir(parents=True, exist_ok=True)
    status = directory / 'status.json'
    try:
        write_json(status, {'state': 'preparing', 'phase': phase, 'fingerprint': token})
        gpu_report = json.loads((ROOT / 'runs/vibing_gpu_smoke/report.json').read_text())
        if not gpu_report['passed'] or gpu_report['neural_code_sha256'] != digest(neural.__file__)['sha256'] or gpu_report['loss_code_sha256'] != digest(ROOT / 'epgnn/train.py')['sha256']:
            raise ValueError('Run GPU smoke for the current implementation before starting')
        if not smoke:
            gate = json.loads((root / 'france/summary.json').read_text())
            if gate['fingerprint'] != token or len(gate['experiments']) != 5:
                raise ValueError('All France experiments must pass before full execution')
        cache, meta, arrays = checked_cache(config)
        if meta['dataset_fingerprint'] != config['dataset_fingerprint']:
            raise ValueError('Cache refers to wrong dataset')
        zones = [meta['zones'].index('FR')] if smoke else meta['eligible']
        if not all(j in meta['eligible'] for j in zones):
            raise ValueError('Smoke zone is ineligible')
        write_json(directory / 'provenance.json', {**provenance, 'fingerprint': token, 'cache': str(cache),
            'training': {'xgboost': 'all finite pretrain/train labels',
                'neural': 'last 90 training days, 3 epochs, first 14 validation days' if smoke else
                          'all finite pretrain/train labels, full validation period'},
            'limitations': ['Conditional publication availability', 'Single-seed exploratory test evaluation',
                'GAT pools zones whereas XGBoost fits per zone; real-vs-shuffled isolates graph topology'],
            'excluded': meta['excluded'], 'zones': [meta['zones'][j] for j in zones]})
        all_rows, experiments = [], []
        for experiment in EXPERIMENTS:
            work = directory / experiment; work.mkdir(exist_ok=True)
            write_json(status, {'state': 'running', 'phase': phase, 'experiment': experiment, 'fingerprint': token})
            if (work / 'complete.json').exists():
                for name, sha in json.loads((work / 'complete.json').read_text()).items():
                    if digest(work / name)['sha256'] != sha:
                        raise ValueError('Saved experiment checksum mismatch')
                saved = json.loads((work / 'result.json').read_text())
                all_rows.extend(saved['rows']); experiments.append(saved)
                continue
            features = data.RECENT if experiment == 'recent_xgb' else data.FEATURES if experiment == 'neighbour_xgb' else data.RICH
            columns = [data.FEATURES.index(f) for f in features]
            started = time.monotonic()
            if experiment.endswith('xgb'):
                predictions, details = xgb_experiment(arrays, columns, zones, work, config)
            else:
                neural_cfg = {**config['neural'], 'identity': token + ':' + phase + ':' + experiment,
                              'batch_size': gpu_report['batch_size']}
                model, prep, edges, curves, batch_size = neural.train(arrays, columns, zones, work, neural_cfg,
                    shuffled=experiment == 'shuffled_gat', smoke=smoke)
                device = neural.require_gpu()
                # Verify actual checkpoint reload against the selected model on real data.
                reload = neural.PriceGAT(len(columns)*2).to(device)
                state = torch.load(work / 'best.pt', map_location=device, weights_only=False)
                reload.load_state_dict(state['model'])
                idx = np.flatnonzero(arrays['split'] == 1)[:24]
                before = neural.infer(model, arrays, idx, columns, prep, edges, batch_size, device)
                after = neural.infer(reload, arrays, idx, columns, prep, edges, batch_size, device)
                np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
                predictions = {s: neural.infer(model, arrays, np.flatnonzero(arrays['split'] == k), columns,
                    prep, edges, batch_size, device)[:, zones] for s, k in [('val', 1), ('test', 2)]}
                details = {'learning_curve': curves, 'preprocessor': prep.state(),
                           'peak_vram_gib': max(c['peak_vram_gib'] for c in curves), 'batch_size': batch_size}
                del model, reload
                torch.cuda.empty_cache()
            rows = []
            for split, prediction in predictions.items():
                rows.extend(evaluate(prediction, arrays, meta, zones, experiment, split, work))
            saved = {'experiment': experiment, 'fingerprint': token, 'features': features,
                     'seconds': time.monotonic()-started, 'rows': rows, 'details': details}
            write_json(work / 'result.json', saved)
            write_json(work / 'complete.json', {p.name: digest(p)['sha256'] for p in work.iterdir()
                                               if p.is_file() and p.name != 'complete.json'})
            all_rows.extend(rows); experiments.append(saved)
            print(f'Completed {phase}/{experiment}: {saved["seconds"]:.1f}s', flush=True)
        # Every method must score exactly the same target population.
        counts = pd.DataFrame(all_rows).pivot(index=['zone', 'split', 'coverage'], columns='experiment', values='hours')
        if counts.isna().any().any() or counts.nunique(axis=1).max() != 1:
            raise ValueError('Experiment target masks/counts differ')
        summary = {'fingerprint': token, 'phase': phase, 'experiments': experiments,
                   'aggregate': grouped_scores(all_rows), 'excluded_zones': meta['excluded']}
        write_json(directory / 'summary.json', summary)
        pd.DataFrame(all_rows).to_csv(directory / 'by_zone.csv', index=False)
        pd.DataFrame(summary['aggregate']).to_csv(directory / 'aggregate.csv', index=False)
        destination = ROOT / f'results/vibing_suite_{phase}.json'
        write_json(destination, summary)
        pd.DataFrame(all_rows).to_csv(ROOT / f'results/vibing_suite_{phase}_by_zone.csv', index=False)
        pd.DataFrame(summary['aggregate']).to_csv(ROOT / f'results/vibing_suite_{phase}_aggregate.csv', index=False)
        write_json(status, {'state': 'complete', 'phase': phase, 'fingerprint': token})
        return summary
    except BaseException as error:
        state = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        write_json(status, {'state': state, 'phase': phase, 'fingerprint': token, 'traceback': traceback.format_exc()})
        raise


def status(config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text())
    token, _ = identity(config)
    root = ROOT / 'runs/vibing_suite' / token
    result = {'fingerprint': token, 'artifact_directory': str(root)}
    for phase in ['france', 'full']:
        path = root / phase / 'status.json'
        result[phase] = json.loads(path.read_text()) if path.exists() else {'state': 'not_started'}
    print(json.dumps(result, indent=2))
