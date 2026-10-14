#!/usr/bin/env python3
"""Summarize dense hypotheses and abstention without implying k ground truth."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import dense_abstain_support as trial
from build_dense_abstain_unfolding import OUTPUT, ANCHORS, CUTS_DB, SOURCE


def _rate(n: int, total: int) -> float | None:
    return float(n / total) if total else None


def _bootstrap_frame_ratio(counts: np.ndarray, total: np.ndarray, seed: int = 281) -> list[float] | None:
    if total.sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(total), (2000, len(total)))
    den = total[ix].sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        rates = counts[ix].sum(axis=1) / den
    rates = rates[np.isfinite(rates)]
    return np.quantile(rates, [.025, .975]).tolist() if len(rates) else None


def summarize(output: Path = OUTPUT) -> dict:
    result = json.loads((output / 'results.json').read_text())
    dense = np.load(output / 'version_a_50x2x256x107.npy', mmap_mode='r')
    frames = result['frames']
    if dense.shape != (50, 2, 256, 107) or not np.isfinite(dense).all():
        raise ValueError('Version A must contain 50 fully finite [2,256,107] frames')
    global_peak = float(np.max(dense[:, 0]))
    k_hist = Counter()
    source_counts = Counter()
    anchor_map = trial.read_anchors(ANCHORS)
    anchor_by_frame = [[] for _ in range(50)]
    for (i, r, a) in anchor_map:
        anchor_by_frame[i].append((r, a))
    split = {'first_half': np.zeros((25, 6), dtype=int),
             'second_half': np.zeros((25, 6), dtype=int)}
    reliable_counts = []
    for i, frame in enumerate(frames):
        stem = frame['radar_stem']
        with np.load(output / 'version_b_qc' / f'{stem}.npz') as qc:
            mask = qc['reliable_mask']; reliable = qc['doppler_reliable_ra_mps']
            source = qc['source_code']; k = qc['k_dense']
            vf = qc['corrected_folded_ra_mps']
            consensus = qc['static_consensus_mask']
            if mask.shape != (256, 107) or source.shape != mask.shape:
                raise ValueError(f'bad QC shape: {stem}')
            if not np.array_equal(reliable[mask], dense[i, 1][mask]) or not np.isnan(reliable[~mask]).all():
                raise ValueError(f'B mismatch: {stem}')
            if np.any(mask & (source != SOURCE['direct_two_sided'])) or np.any(consensus & ~mask):
                raise ValueError(f'fallback leaked into reliable mask: {stem}')
            if not np.allclose(vf + k * result['period_mps'], dense[i, 1], atol=3e-5):
                raise ValueError(f'foldback reconstruction mismatch: {stem}')
            for code, n in zip(*np.unique(source, return_counts=True)):
                source_counts[int(code)] += int(n)
            for order, n in zip(*np.unique(k, return_counts=True)):
                k_hist[int(order)] += int(n)
            reliable_counts.append(int(mask.sum()))
            rows = anchor_by_frame[i]
            arr = split['first_half' if i < 25 else 'second_half'][i if i < 25 else i-25]
            for r, a in rows:
                row = anchor_map[(i, r, a)]
                kref = int(np.rint((float(row['static_reference_mps']) - float(vf[r, a])) / result['period_mps']))
                arr[0] += 1
                arr[4] += int(k[r, a] == kref)
                if mask[r, a]:
                    arr[1] += 1
                    arr[2] += int(k[r, a] == kref)
                resid = float(row['static_reference_mps']) - (float(vf[r,a]) + kref*result['period_mps'])
                if result['period_mps']/2 - abs(resid) >= .5 and mask[r,a]:
                    arr[3] += int(k[r,a] == kref)
                    arr[5] += 1
    if sum(source_counts.values()) != 50*256*107:
        raise ValueError('source counts do not cover all cells')
    # Original cached Power must be byte-for-byte identical, checked independently per frame.
    import dense_abstain_support as base
    for i, f in enumerate(frames):
        with np.load(base.CACHE / 'radar_maps' / f"{f['radar_stem']}.npz") as cache:
            if not np.array_equal(dense[i, 0], cache['power_peak_ra']):
                raise ValueError(f'Power channel mismatch: {f["radar_stem"]}')
    split_summary = {}
    for name, arr in split.items():
        n, selected, correct, stable_correct, dense_correct, stable_selected = arr.sum(axis=0).tolist()
        split_summary[name] = dict(anchor_count=n, reliable_anchor_count=selected,
            anchor_coverage=_rate(selected, n),
            anchor_coverage_frame_ci95=_bootstrap_frame_ratio(arr[:, 1], arr[:, 0]),
            dense_pseudo_agreement=_rate(dense_correct, n),
            pseudo_agreement=_rate(correct, selected),
            pseudo_agreement_frame_ci95=_bootstrap_frame_ratio(arr[:, 2], arr[:, 1]),
            stable_reliable_anchor_count=stable_selected,
            stable_pseudo_agreement=_rate(stable_correct, stable_selected))
    velocities = np.asarray(dense[:, 1]).ravel()
    velocity_quantiles = {str(q):float(np.quantile(velocities,q)) for q in (0,.001,.01,.5,.99,.999,1)}
    strata = {}
    for cut in CUTS_DB:
        x = result['power_strata'][str(cut)]
        strata[str(cut)] = dict(**x, reliable_coverage=_rate(x['reliable'], x['n']),
                                 static_consensus_coverage=_rate(x['static_consensus'], x['n']))
    selection_path = output / 'parameter_selection.json'
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    summary = dict(all_cell_count=result['all_cell_count'], dense_coverage=1.0,
        match_lag=result.get('match_lag', 2),
        reliable_count=result['reliable_count'], reliable_coverage=result['reliable_coverage'],
        static_consensus_coverage=result['static_consensus_coverage'],
        source_counts={name:source_counts[code] for name,code in SOURCE.items()},
        source_fractions={name:source_counts[code]/result['all_cell_count'] for name,code in SOURCE.items()},
        k_histogram={str(k):n for k,n in sorted(k_hist.items())},
        velocity_quantiles_mps=velocity_quantiles,
        power_strata=strata, split_anchor_metrics=split_summary,
        score_margin_threshold=result['calibration']['threshold'],
        parameter_selection=selection['selected'] if selection else None)
    (output / 'summary.json').write_text(json.dumps(summary,indent=2))
    fig, ax=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    names=list(SOURCE)
    ax[0,0].bar(names,[100*summary['source_fractions'][n] for n in names])
    ax[0,0].set_ylabel('% all RA cells');ax[0,0].set_title('Direct and fallback provenance')
    ax[0,0].tick_params(axis='x',rotation=20)
    orders=list(sorted(k_hist));ax[0,1].bar(orders,[k_hist[k] for k in orders])
    ax[0,1].set_xlabel('Selected fold order k');ax[0,1].set_ylabel('Cells');ax[0,1].set_title('Dense k distribution')
    ax[1,0].plot(np.arange(1,51),np.array(reliable_counts)/(256*107),marker='.',ms=3)
    ax[1,0].axvline(25.5,color='gray',ls='--')
    ax[1,0].set_ylim(0,1);ax[1,0].set_xlabel('Frame index');ax[1,0].set_ylabel('Reliable fraction')
    ax[1,0].set_title('Reliable coverage by frame')
    cuts=[str(c) for c in CUTS_DB]
    ax[1,1].bar(cuts,[100*strata[c]['reliable_coverage'] for c in cuts])
    ax[1,1].set_ylabel('% strong cells');ax[1,1].set_xlabel('Power threshold (dB)')
    ax[1,1].set_title('Reliable coverage at three Power thresholds')
    fig.savefig(output/'coverage_and_provenance.png',dpi=150);plt.close(fig)
    lines=['# K-Radar sequence 1 dense and abstain Doppler diagnostic','',
      '50 native RA frames, each [256 range, 107 azimuth]. Version A is [Power, dense unfolded Doppler] with every cell filled. Version B is a separate QC artifact; its mask is not a VLA input. Matching is pose assisted, not the CARLA algorithm.','',
      f"- All-cell dense coverage: 100% ({result['all_cell_count']:,} cells).",
      f"- Bidirectionally reliable QC coverage: {100*summary['reliable_coverage']:.2f}% ({result['reliable_count']:,} cells).",
      f"- Static pose-prior consensus QC coverage: {100*summary['static_consensus_coverage']:.2f}%.",
      f"- Selected matching lag: {summary['match_lag']} frames on each available side.",
      f"- Selected score-margin threshold: {summary['score_margin_threshold']}.",'']
    if selection:
        chosen = next(x for x in selection['candidates'] if x['lag'] == selection['selected']['lag'])
        for split, title in (('train', 'Training frames 3–24'), ('report', 'Held-out report frames 25–46')):
            row = next(x for x in chosen[split]['margins'] if x['threshold'] == selection['selected']['margin_threshold'])
            lines += [f"- {title}: dense pseudo agreement {100*chosen[split]['dense_pseudo_agreement']:.2f}%; stable reliable anchors {row['stable_n']}, agreement {100*row['stable_pseudo_agreement']:.2f}%; −48 dB strong-cell reliable coverage {100*row['strong_reliable_coverage']:.2f}%."]
        lines += ['- Parameter selection used training frames only; held-out numbers are descriptive.', '']
    lines += ['## Decision sources','', '| Source | Cells | Share |','|---|---:|---:|']
    for name in names:lines.append(f"| {name} | {source_counts[SOURCE[name]]:,} | {100*summary['source_fractions'][name]:.2f}% |")
    lines += ['', '## Power threshold sensitivity','', '| Threshold | Cells | Reliable | Coverage |','|---|---:|---:|---:|']
    for cut in CUTS_DB:
        x=strata[str(cut)]
        lines.append(f"| {cut} dB | {x['n']:,} | {x['reliable']:,} | {100*x['reliable_coverage']:.2f}% |")
    lines += ['', '## Static candidate pseudo-reference','', '| Split | Anchors | Reliable | Coverage | Agreement among reliable |','|---|---:|---:|---:|---:|']
    for name,x in split_summary.items():
        a='n/a' if x['pseudo_agreement'] is None else f"{100*x['pseudo_agreement']:.2f}%"
        lines.append(f"| {name} | {x['anchor_count']:,} | {x['reliable_anchor_count']:,} | {100*x['anchor_coverage']:.2f}% | {a} |")
    lines += ['', f"Dense pseudo-reference agreement on all static anchors: first half {100*split_summary['first_half']['dense_pseudo_agreement']:.2f}%, second half {100*split_summary['second_half']['dense_pseudo_agreement']:.2f}%.", '', 'The margin threshold was calibrated only on first-half static candidates with at least 0.5 m/s fold-boundary margin. Second-half agreement uses the same pose-derived pseudo-reference, so it is not independent validation. Frame-level bootstrap intervals are in summary.json.','',
      '## Interpretation','',
      f"The dense field has {100*split_summary['first_half']['dense_pseudo_agreement']:.2f}% / {100*split_summary['second_half']['dense_pseudo_agreement']:.2f}% pseudo-reference agreement on first / second-half static anchors, and the reliable mask retains {100*strata['-48']['reliable_coverage']:.2f}% of -48 dB strong cells. These results do not support treating the dense candidate as a validated VLA input yet.",'',
      'Dense speeds, fallback order, and static pseudo-reference agreement do not establish true fold-order accuracy. Low-power cells and dynamic targets have no independent k truth. The LiDAR label time serves as a radar-frame proxy; independent radar timestamps are unavailable. No oracle labels or formal model inputs were generated.','',
      '## Files','',
      '- `version_a_50x2x256x107.npy` and `version_a/*.npy`: finite dense two-channel candidates.',
      '- `version_b_qc/*.npz`: reliable Doppler and mask, static consensus mask, k, source code, and score margin.',
      '- `ra_frames/*.png` and `index.html`: native RA plots and frame switcher.',
      '- `coverage_and_provenance.png`, `summary.json`, and `results.json`: aggregate diagnostics.','']
    (output/'README.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT)
    args=p.parse_args();summarize(args.output)
