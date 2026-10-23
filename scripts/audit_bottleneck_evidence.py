"""Recompute the repair design's paired metrics from pinned archived results.

No model inference, threshold selection for deployment, or training is performed.
Run from the project root with the editable environment; output is an artifact.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np


REVISION = '959e69aba30878d8e09954b555fe1c916843a9d1'
BASE = 'docs/experiment_results_20261023/server/analysis/'
RUNS = {'C0': 'krsv2_baseline_20261022/run/',
        'A': 'krsv2_weight_controls_20261023/A_dmin025/run/',
        'B': 'krsv2_weight_controls_20261023/B_areq3/run/'}


def audit():
    sources = {}
    root = Path(__file__).resolve().parents[1]

    def rows(path):
        blob = subprocess.check_output(['git', 'show', REVISION + ':' + BASE + path], cwd=root)
        sources[BASE + path] = hashlib.sha256(blob).hexdigest()
        if path.endswith('.gz'):
            blob = gzip.decompress(blob)
        return [json.loads(line) for line in blob.splitlines()]

    result = dict(source_commit=REVISION, scope='paired archive analysis; not a new experiment',
                  validation_epoch=20, risk={}, motion={}, train_order={})
    cohorts, objects = [], {}
    for arm, path in RUNS.items():
        data = rows(path + 'epoch_020_val_samples.jsonl.gz')
        cohorts.append([(r['sample_id'], r['target'], r['mask']) for r in data])
        pred, target, mask = (np.asarray([r[key] for r in data]) for key in ('risk', 'target', 'mask'))
        windows = {}
        for h, name in enumerate(('0.5s', '1s', '2s')):
            valid = mask[:, h].astype(bool)
            score, y = pred[valid, h], target[valid, h] > .5
            order = np.argsort(-score, kind='stable')
            endpoints = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(order) - 1]
            tp = np.cumsum(y[order])[endpoints]
            fp = (np.arange(1, len(order) + 1) - np.cumsum(y[order]))[endpoints]
            at_recall = np.flatnonzero(tp >= .8 * y.sum())
            windows[name] = dict(known=int(valid.sum()), positives=int(y.sum()),
                positive_episodes=len({r['episode_id'] for r in data if r['mask'][h] and r['target'][h] > .5}),
                tp_at_0p5=int(((score >= .5) & y).sum()),
                fp_at_0p5=int(((score >= .5) & ~y).sum()),
                exploratory_min_fp_at_80pct_recall=int(fp[at_recall[0]]) if len(at_recall) else None,
                positive_dmin_min_m=float(pred[valid, 3][y].min()))
        brake = mask[:, 4].astype(bool) & (target[:, 4] > 0)
        result['risk'][arm] = dict(samples=len(data), windows=windows,
            positive_brake_count=int(brake.sum()),
            positive_brake_pred_median=float(np.median(pred[brake, 4])),
            positive_brake_target_median=float(np.median(target[brake, 4])))
        objects[arm] = {(r['sample_id'], str(o['id'])): o for r in data for o in r['objects']}
    assert cohorts[0] == cohorts[1] == cohorts[2], 'Validation cohort/labels differ'
    assert objects['C0'].keys() == objects['A'].keys() == objects['B'].keys(), 'GT objects differ'
    common = {key for key in objects['C0'] if all(arm[key]['matched'] for arm in objects.values())}
    for arm, values in objects.items():
        errors = [values[key]['future_ade_m'] for key in common]
        assert all(value is not None for value in errors), 'Common target lacks motion label'
        result['motion'][arm] = dict(common_targets=len(common), ade_m=statistics.mean(errors))
    for epoch in (1, 20):
        orders = {}
        for arm, path in RUNS.items():
            data = rows(path + f'epoch_{epoch:03d}_train_attempt_01.jsonl.gz')
            orders[arm] = [sample for r in data for sample in r['sample_ids']]
        result['train_order'][str(epoch)] = {
            arm: dict(samples=len(order), unique_samples=len(set(order)),
                      same_cohort_as_C0=set(order) == set(orders['C0']),
                      same_order_as_C0=order == orders['C0'],
                      sha256=hashlib.sha256('\n'.join(order).encode()).hexdigest())
            for arm, order in orders.items()}

    data = rows('full_val_diagnosis_20261018/e5_complete_validation/samples.jsonl.gz')
    errors = {key: [] for key in ('free', 'teacher_forced_numeric', 'codec', 'cv')}
    for row in data:
        scores = row['scores']
        maps = {key: dict(zip(scores[key].get('indices', []), scores[key].get('errors_m', [])))
                for key in ('free', 'teacher_forced_numeric', 'codec')}
        maps['cv'] = dict(zip(scores['free'].get('indices', []), scores['free'].get('paired_cv_errors_m', [])))
        required = set(range(2 if row['teacher_mode'] == 'short' else 6))
        shared = required.intersection(*maps.values())
        for key in errors:
            errors[key].extend(maps[key][i] for i in sorted(shared))
    result['historical_sft'] = dict(samples=len(data), common_points=len(errors['free']),
                                   ade_m={key: statistics.mean(value) for key, value in errors.items()})
    result['source_blob_sha256'] = sources
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    print(f"Wrote paired evidence from {REVISION[:7]} to {args.output}")
