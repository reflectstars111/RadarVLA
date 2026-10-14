#!/usr/bin/env python3
"""Select lag and QC margin for dense unfolding A using first-half frames only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import dense_abstain_support as support
from build_dense_abstain_unfolding import ANCHORS, SOURCE

LAGS = (1, 2, 3)
MARGINS = (0.35, 0.45, 0.5166545689105988, 0.63, 0.79, 0.98, 1.24)
TRAIN = tuple(range(3, 25))
REPORT = tuple(range(25, 47))
MIN_STABLE_ANCHORS = 500
MIN_TRAIN_AGREEMENT = 0.90
MAX_DENSE_REGRESSION = 0.02


def score_lag(folder: Path) -> dict:
    result = json.loads((folder / 'results.json').read_text())
    lag = int(result['match_lag'])
    if lag not in LAGS or result['shape'] != [50, 2, 256, 107]:
        raise ValueError(f'bad candidate: {folder}')
    dense = np.load(folder / 'version_a_50x2x256x107.npy', mmap_mode='r')
    if not np.isfinite(dense).all():
        raise ValueError(f'nonfinite dense candidate: {folder}')
    peak = float(np.max(dense[:, 0]))
    anchors = support.read_anchors(ANCHORS)
    by_frame = [[] for _ in range(50)]
    for (i, r, a), row in anchors.items():
        by_frame[i].append((r, a, float(row['static_reference_mps'])))
    collected = {}
    for name, indices in (('train', TRAIN), ('report', REPORT)):
        direct, margin, strong = [], [], []
        a_direct, a_margin, a_correct, a_stable = [], [], [], []
        for i in indices:
            stem = result['frames'][i]['radar_stem']
            with np.load(folder / 'version_b_qc' / f'{stem}.npz') as z:
                src = z['source_code']; agree = z['direct_agreed']; mar = z['score_margin']
                k = z['k_dense']; vf = z['corrected_folded_ra_mps']
                d = (src == SOURCE['direct_two_sided']) & agree
                db = 10 * np.log10(np.maximum(dense[i, 0], 1) / peak)
                direct.append(d.ravel()); margin.append(mar.ravel()); strong.append((db >= -48).ravel())
                for r, a, ref in by_frame[i]:
                    kr = int(np.rint((ref - float(vf[r, a])) / result['period_mps']))
                    residual = ref - (float(vf[r, a]) + kr * result['period_mps'])
                    a_direct.append(bool(d[r, a])); a_margin.append(float(mar[r, a]))
                    a_correct.append(bool(k[r, a] == kr))
                    a_stable.append(bool(result['period_mps'] / 2 - abs(residual) >= .5))
        direct = np.concatenate(direct); margin = np.concatenate(margin); strong = np.concatenate(strong)
        a_direct = np.array(a_direct, bool); a_margin = np.array(a_margin)
        a_correct = np.array(a_correct, bool); a_stable = np.array(a_stable, bool)
        rows = []
        for threshold in MARGINS:
            chosen = a_direct & a_stable & (a_margin >= threshold)
            strong_n = int(strong.sum())
            rows.append(dict(threshold=threshold, stable_n=int(chosen.sum()),
                stable_pseudo_agreement=float(a_correct[chosen].mean()) if chosen.any() else None,
                strong_reliable=int((direct & strong & (margin >= threshold)).sum()),
                strong_n=strong_n,
                strong_reliable_coverage=float((direct & strong & (margin >= threshold)).sum() / strong_n) if strong_n else None))
        collected[name] = dict(anchor_n=int(len(a_correct)),
            dense_pseudo_agreement=float(a_correct.mean()) if len(a_correct) else None,
            margins=rows)
    return dict(lag=lag, **collected)


def select(root: Path) -> dict:
    candidates = [score_lag(root / f'lag{lag}') for lag in LAGS]
    baseline = candidates[1]['train']['dense_pseudo_agreement']
    eligible = []
    for c in candidates:
        train = c['train']
        for row in train['margins']:
            if (train['dense_pseudo_agreement'] >= baseline - MAX_DENSE_REGRESSION and
                row['stable_n'] >= MIN_STABLE_ANCHORS and
                row['stable_pseudo_agreement'] is not None and
                row['stable_pseudo_agreement'] >= MIN_TRAIN_AGREEMENT):
                eligible.append((row['strong_reliable_coverage'], -c['lag'], -row['threshold'], c['lag'], row['threshold']))
    if not eligible:
        raise RuntimeError('no eligible lag and margin; current A remains untouched')
    _, _, _, lag, threshold = max(eligible)
    return dict(selected=dict(lag=lag, margin_threshold=threshold),
        selection_rule='train frames 3..24 only: stable pseudo agreement >=90%, stable n>=500, dense agreement no more than 2 pp below lag-2 baseline; maximize -48 dB strong reliable coverage, then shorter lag, then lower margin',
        comparison_frames=dict(train=[3,24], report=[25,46]),
        baseline_lag2_train_dense_pseudo_agreement=baseline, candidates=candidates)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate_root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = select(args.candidate_root)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['selected']))
