#!/usr/bin/env python3
"""K-Radar sequence-1 dense Doppler hypothesis and separate abstention QC.

Every RA cell receives a finite v_fold+k*T candidate. These are diagnostic
hypotheses, not oracle labels. Sources, confidence and static-consensus masks
are stored separately from the two-channel Version A tensor.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

import dense_abstain_support as base
import dense_abstain_support as trial
from dense_abstain_support import corrected_bins

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'diagnostics/dense_abstain_unfolding_v1'
ANCHORS = HERE / 'static_anchor_candidates.csv'
SPEED_LIMIT_MPS = 35.0
TIE_EPS = 1e-6
SOURCE = {'direct_two_sided': 1, 'direct_one_sided': 2,
          'spatial_neighbor': 3, 'temporal_previous': 4, 'zero_order': 5}
CUTS_DB = (-48, -40, -35)


def clip_match_log_power(power_rae: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    logp = np.log1p(np.asarray(power_rae, dtype=np.float64))
    lo, hi = np.quantile(logp, [.001, .999])
    if not np.isfinite(lo + hi) or hi < lo:
        raise ValueError('invalid match-power quantiles')
    return np.clip(logp, lo, hi).astype(np.float32), (float(lo), float(hi))


def patch_cost(center: np.ndarray, adjacent: np.ndarray) -> np.ndarray:
    """Existing texture-plus-amplitude cost, with explicit finite support."""
    valid = np.isfinite(center) & np.isfinite(adjacent)
    count = valid.sum(axis=1)
    denom = np.maximum(count, 1)
    cm = np.where(valid, center, 0).sum(axis=1) / denom
    am = np.where(valid, adjacent, 0).sum(axis=1) / denom
    cd = np.where(valid, center - cm[:, None], 0)
    ad = np.where(valid, adjacent - am[:, None], 0)
    texture = np.abs(cd - ad).sum(axis=1) / denom
    cost = texture + .15 * np.abs(center[:, 4] - adjacent[:, 4])
    return np.where((count >= 5) & np.isfinite(cost), cost, np.inf).astype(np.float32)


def load_features(frames: list[dict], axes: dict) -> tuple[list[dict], float]:
    features = []
    for i, frame in enumerate(frames):
        cube = loadmat(frame['radar_path'], variable_names=['arrDREA'])['arrDREA']
        if cube.shape != base.EXPECTED_DREA or not np.isfinite(cube).all():
            raise ValueError(f'invalid DREA: {frame["radar_stem"]}')
        dra = cube.mean(axis=2)
        pb = dra.argmax(axis=0).astype(np.uint8)
        rr, aa = np.indices(pb.shape)
        peak_e = cube[pb, rr, :, aa].argmax(axis=2).astype(np.uint8)
        log_rae, clip_bounds = clip_match_log_power(cube.max(axis=0))
        with np.load(base.CACHE / 'radar_maps' / f'{frame["radar_stem"]}.npz') as z:
            if not np.array_equal(pb, z['peak_doppler_bin']):
                raise ValueError(f'DREA/cache peak mismatch: {frame["radar_stem"]}')
            power = z['power_peak_ra'].copy()
        vf = axes['velocity_mps'][corrected_bins(pb, len(axes['velocity_mps']))].astype(np.float32)
        features.append(dict(power=power, vf=vf, peak_e=peak_e, peak_bin=pb,
                             log_rae=log_rae, clip_bounds=clip_bounds))
        del cube, dra
        print(f'DREA {i + 1}/50 {frame["radar_stem"]}', flush=True)
    return features, max(float(f['power'].max()) for f in features)


def side_scores(center: dict, neighbor: dict, ri: np.ndarray, ai: np.ndarray,
                orders: np.ndarray, period: float, dt: float, axes: dict,
                pose_c: np.ndarray, pose_n: np.ndarray) -> np.ndarray:
    r = axes['range_m'][ri]
    az = axes['azimuth_deg'][ai]
    el = axes['elevation_deg'][center['peak_e'][ri, ai]]
    vf = center['vf'][ri, ai]
    cp = trial.sample_patches(center['log_rae'], ri.astype(float),
                              center['peak_e'][ri, ai].astype(float), ai.astype(float))
    neighbor_az, neighbor_el = trial.neighbor_coordinates(r, az, el, pose_c, pose_n, True)
    ab = (neighbor_az - axes['azimuth_deg'][0]) / np.median(np.diff(axes['azimuth_deg']))
    eb = (neighbor_el - axes['elevation_deg'][0]) / np.median(np.diff(axes['elevation_deg']))
    step = float(np.median(np.diff(axes['range_m'])))
    out = np.full((len(ri), len(orders)), np.inf, dtype=np.float32)
    for j, k in enumerate(orders):
        v = vf + k * period
        # Pose supplies neighbor direction; this range formula already includes
        # ego translation and must not be added to the static warped range.
        rb = (r + v * dt - axes['range_m'][0]) / step
        ok = ((np.abs(v) <= SPEED_LIMIT_MPS) & (rb >= 0) & (rb <= 255) &
              (ab >= 0) & (ab <= 106) & (eb >= 0) & (eb <= 36))
        if np.any(ok):
            patch = trial.sample_patches(neighbor['log_rae'], rb[ok], eb[ok], ab[ok])
            out[ok, j] = patch_cost(cp[ok], patch)
    return out


def _winner(cost: np.ndarray, orders: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid = np.isfinite(cost)
    safe = np.where(valid, cost, np.inf)
    ix = np.argmin(safe, axis=1)
    best = safe[np.arange(len(ix)), ix]
    second = np.partition(safe, 1, axis=1)[:, 1]
    with np.errstate(invalid='ignore'):
        gap = second - best
    unique = np.isfinite(best) & ((gap > TIE_EPS) | np.isinf(second))
    gap = np.where(np.isfinite(gap), gap, 0).astype(np.float32)
    return orders[ix], best, gap, unique


def choose_direct(before: np.ndarray | None, after: np.ndarray | None,
                  orders: np.ndarray) -> dict[str, np.ndarray]:
    if before is None and after is None:
        raise ValueError('at least one neighbor is required')
    shape = before.shape if before is not None else after.shape
    b = before if before is not None else np.full(shape, np.inf, np.float32)
    a = after if after is not None else np.full(shape, np.inf, np.float32)
    joint = np.where(np.isfinite(b) & np.isfinite(a), b + a, np.inf)
    jk, jbest, jgap, junique = _winner(joint, orders)
    bk, bbest, bgap, bunique = _winner(b, orders)
    ak, abest, agap, aunique = _winner(a, orders)
    n = shape[0]
    k = np.zeros(n, dtype=np.int16)
    source = np.zeros(n, dtype=np.uint8)
    margin = np.zeros(n, dtype=np.float32)
    source[junique] = SOURCE['direct_two_sided']
    k[junique] = jk[junique]
    margin[junique] = jgap[junique]
    # A joint tie is unresolved even if an individual direction has a winner.
    joint_has_finite = np.isfinite(jbest)
    need = ~joint_has_finite
    use_b = need & bunique & (~aunique | (bbest <= abest))
    use_a = need & aunique & ~use_b
    source[use_b | use_a] = SOURCE['direct_one_sided']
    k[use_b] = bk[use_b]
    k[use_a] = ak[use_a]
    margin[use_b] = bgap[use_b]
    margin[use_a] = agap[use_a]
    agreed = junique & bunique & aunique & (jk == bk) & (jk == ak)
    return dict(k=k, source=source, margin=margin, agreed=agreed,
                all_invalid=~np.isfinite(jbest) & ~np.isfinite(bbest) & ~np.isfinite(abest),
                joint_tie=joint_has_finite & ~junique)


def spatial_fill(k: np.ndarray, source: np.ndarray, vf: np.ndarray,
                 period: float, agreed: np.ndarray | None = None) -> np.ndarray:
    """One synchronous pass from bilaterally direct neighboring cells."""
    h, w = source.shape
    direct = source == SOURCE['direct_two_sided']
    if agreed is not None:
        direct &= agreed
    votes = np.zeros((21, h, w), dtype=np.uint8)
    for dr in (-1, 0, 1):
        for da in (-1, 0, 1):
            if dr == 0 and da == 0:
                continue
            rs = slice(max(0, dr), min(h, h + dr))
            ass = slice(max(0, da), min(w, w + da))
            rt = slice(max(0, -dr), min(h, h - dr))
            at = slice(max(0, -da), min(w, w - da))
            # source at (rs,ass) contributes to target at (rt,at)
            valid = direct[rs, ass].copy()
            delta = (vf[rs, ass] - vf[rt, at] + period / 2) % period - period / 2
            valid &= np.abs(delta) <= .5
            for ix, order in enumerate(range(-10, 11)):
                votes[ix, rt, at] += valid & (k[rs, ass] == order)
    maxvote = votes.max(axis=0)
    mode_ix = votes.argmax(axis=0)
    tie = (votes == maxvote[None]).sum(axis=0) > 1
    proposed = mode_ix.astype(np.int16) - 10
    use = ((source == 0) & (maxvote >= 3) & ~tie &
           (np.abs(vf + proposed * period) <= SPEED_LIMIT_MPS))
    k[use] = proposed[use]
    source[use] = SOURCE['spatial_neighbor']
    return use


def temporal_fill(k: np.ndarray, source: np.ndarray, vf: np.ndarray,
                  peak_e: np.ndarray, previous: dict | None, axes: dict,
                  pose_c: np.ndarray, pose_p: np.ndarray, period: float) -> np.ndarray:
    use = np.zeros(source.shape, bool)
    if previous is None:
        return use
    ri, ai = np.where(source == 0)
    if not len(ri):
        return use
    r = axes['range_m'][ri]; a = axes['azimuth_deg'][ai]
    e = axes['elevation_deg'][peak_e[ri, ai]]
    xyz = trial.polar_xyz(r, a, e)
    near = base.world_to_radar(base.radar_to_world(xyz, pose_c), pose_p)
    rp = np.linalg.norm(near, axis=1)
    ap = np.rad2deg(np.arctan2(-near[:, 1], near[:, 0]))
    rb = np.rint((rp - axes['range_m'][0]) / np.median(np.diff(axes['range_m']))).astype(int)
    ab = np.rint((ap - axes['azimuth_deg'][0]) / np.median(np.diff(axes['azimuth_deg']))).astype(int)
    inside = (rb >= 0) & (rb < 256) & (ab >= 0) & (ab < 107)
    rb = np.clip(rb, 0, 255); ab = np.clip(ab, 0, 106)
    pk = previous['direct_k'][rb, ab]
    psource = previous['direct_source'][rb, ab]
    pvf = previous['vf'][rb, ab]
    diff = (pvf - vf[ri, ai] + period / 2) % period - period / 2
    accepted = (inside & (psource == SOURCE['direct_two_sided']) &
                (np.abs(diff) <= .5) &
                (np.abs(vf[ri, ai] + pk * period) <= SPEED_LIMIT_MPS))
    use[ri[accepted], ai[accepted]] = True
    k[use] = pk[accepted]
    source[use] = SOURCE['temporal_previous']
    return use


def fill_orders(direct: dict, vf: np.ndarray, peak_e: np.ndarray,
                previous: dict | None, axes: dict, pose_c: np.ndarray,
                pose_p: np.ndarray | None, period: float) -> dict:
    k = direct['k'].copy()
    source = direct['source'].copy()
    direct_k = k.copy(); direct_source = source.copy()
    spatial_fill(k, source, vf, period, direct['agreed'])
    if pose_p is not None:
        temporal_fill(k, source, vf, peak_e, previous, axes, pose_c, pose_p, period)
    zero = source == 0
    k[zero] = 0
    source[zero] = SOURCE['zero_order']
    return dict(k=k, source=source, direct_k=direct_k,
                direct_source=direct_source, vf=vf.copy())


def choose_threshold(records: list[dict]) -> tuple[float | None, list[dict]]:
    train = [x for x in records if x['frame_index'] <= 24 and x['fold_margin_mps'] >= .5]
    margins = np.array([x['margin'] for x in train if x['agreed'] and x['direct']], float)
    grid = np.unique(np.r_[0, np.quantile(margins, np.linspace(0, .95, 20))]) if len(margins) else np.array([0.])
    curve = []
    for threshold in grid:
        take = [x for x in train if x['agreed'] and x['direct'] and x['margin'] >= threshold]
        rate = float(np.mean([x['k'] == x['k_ref'] for x in take])) if take else None
        curve.append(dict(threshold=float(threshold), n=len(take), pseudo_agreement=rate))
    valid = [x for x in curve if x['n'] >= 50 and x['pseudo_agreement'] is not None and
             x['pseudo_agreement'] >= .90]
    return (min(x['threshold'] for x in valid) if valid else None), curve


def anchor_report(records: list[dict], mask: np.ndarray, frames: list[dict]) -> dict:
    by_half = {}
    for name, predicate in (('first_half', lambda i: i <= 24),
                            ('second_half', lambda i: i >= 25)):
        take = [x for x in records if predicate(x['frame_index']) and
                mask[x['frame_index'], x['range_bin'], x['azimuth_bin']]]
        stable = [x for x in take if x['fold_margin_mps'] >= .5]
        by_half[name] = dict(n=len(take), pseudo_agreement=(float(np.mean([
            x['k'] == x['k_ref'] for x in take])) if take else None),
            stable_n=len(stable), stable_pseudo_agreement=(float(np.mean([
            x['k'] == x['k_ref'] for x in stable])) if stable else None))
    return by_half


def effective_match_lag(index: int, frame_count: int, requested: int) -> int:
    """Use nearby frames at sequence edges, as in the accepted A version."""
    return min(requested, max(1, min(index, frame_count - 1 - index)))


def run(output: Path = OUTPUT, match_lag: int = 2, margin_threshold: float | None = None) -> dict:
    if match_lag not in (1, 2, 3):
        raise ValueError("match_lag must be 1, 2, or 3")
    frames = json.loads((base.CACHE / 'frames.json').read_text())
    if len(frames) != 50 or frames[0]['radar_stem'] != '00281' or frames[-1]['radar_stem'] != '00330':
        raise ValueError('expected sequence-1 first 50 frames')
    with np.load(base.CACHE / 'radar_axes.npz') as z:
        axes = {k: z[k].astype(np.float64) for k in z.files}
    period = float(np.median(np.diff(axes['velocity_mps'])) * len(axes['velocity_mps']))
    orders = trial.candidate_orders(axes['velocity_mps'], period, SPEED_LIMIT_MPS)
    features, global_peak = load_features(frames, axes)
    poses_raw = np.loadtxt(base.POSES)
    poses = [base.pose_matrix(poses_raw[int(f['lidar_stem']) - 1]) for f in frames]
    times = np.array([base.read_labels(Path(f['label_path']))[0] for f in frames])
    rr, aa = np.indices((256, 107))
    ri, ai = rr.ravel(), aa.ravel()
    decisions = []
    for i, center in enumerate(features):
        lag = effective_match_lag(i, len(frames), match_lag)
        p = i - lag if i - lag >= 0 else None
        n = i + lag if i + lag < 50 else None
        before = (side_scores(center, features[p], ri, ai, orders, period,
                              times[p] - times[i], axes, poses[i], poses[p]) if p is not None else None)
        after = (side_scores(center, features[n], ri, ai, orders, period,
                             times[n] - times[i], axes, poses[i], poses[n]) if n is not None else None)
        direct = choose_direct(before, after, orders)
        direct = {key: value.reshape(256, 107) for key, value in direct.items()}
        previous = decisions[i - 1] if i else None
        filled = fill_orders(direct, center['vf'], center['peak_e'], previous,
                             axes, poses[i], poses[i - 1] if i else None, period)
        filled.update(agreed=direct['agreed'], margin=direct['margin'],
                      all_invalid=direct['all_invalid'], joint_tie=direct['joint_tie'],
                      lag=lag, prev_index=p, next_index=n)
        decisions.append(filled)
        counts = Counter(filled['source'].ravel().tolist())
        print(f'Match {i + 1}/50 {frames[i]["radar_stem"]}: {dict(counts)}', flush=True)
    # Anchors are evaluation-only; they never enter the dense k selection.
    anchors = trial.read_anchors(ANCHORS)
    records = []
    for (i, r, a), row in anchors.items():
        ref = float(row['static_reference_mps'])
        vf = float(features[i]['vf'][r, a])
        kref = int(np.rint((ref - vf) / period))
        residual = ref - (vf + kref * period)
        records.append(dict(frame_index=i, range_bin=r, azimuth_bin=a, k_ref=kref,
                            k=int(decisions[i]['k'][r, a]),
                            direct=decisions[i]['source'][r, a] == SOURCE['direct_two_sided'],
                            agreed=bool(decisions[i]['agreed'][r, a]),
                            margin=float(decisions[i]['margin'][r, a]),
                            fold_margin_mps=float(period / 2 - abs(residual))))
    calibrated_threshold, curve = choose_threshold(records)
    threshold = calibrated_threshold if margin_threshold is None else float(margin_threshold)
    if threshold is not None and (not np.isfinite(threshold) or threshold < 0):
        raise ValueError("margin_threshold must be nonnegative and finite")
    reliable = np.zeros((50, 256, 107), bool)
    consensus = np.zeros_like(reliable)
    for i, decision in enumerate(decisions):
        if threshold is not None and 0 < i < 49:
            reliable[i] = ((decision['source'] == SOURCE['direct_two_sided']) &
                           decision['agreed'] & (decision['margin'] >= threshold))
        if 0 < i < 49:
            r = axes['range_m'][ri]; a = axes['azimuth_deg'][ai]
            e = axes['elevation_deg'][features[i]['peak_e'][ri, ai]]
            prior = trial.static_radial_prior(r, a, e, poses[i], poses[i - 1],
                                              poses[i + 1], times[i + 1] - times[i - 1])
            diff = (prior - features[i]['vf'].ravel()) / period
            prior_k = np.rint(diff).astype(np.int16).reshape(256, 107)
            margin = (period * (.5 - np.abs(diff - np.rint(diff)))).reshape(256, 107)
            consensus[i] = reliable[i] & (decision['k'] == prior_k) & (margin >= .5)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'version_a').mkdir(exist_ok=True)
    (output / 'version_b_qc').mkdir(exist_ok=True)
    stack = np.lib.format.open_memmap(output / 'version_a_50x2x256x107.npy', mode='w+',
                                      dtype=np.float32, shape=(50, 2, 256, 107))
    frame_results = []
    for i, frame in enumerate(frames):
        f = features[i]; d = decisions[i]
        speed = (f['vf'] + d['k'] * period).astype(np.float32)
        if not np.isfinite(speed).all() or not np.isfinite(f['power']).all():
            raise ValueError(f'nonfinite dense channel: {frame["radar_stem"]}')
        pair = np.stack((f['power'], speed)).astype(np.float32)
        if not np.array_equal(pair[0], f['power']):
            raise AssertionError('Power was not copied exactly')
        stack[i] = pair
        np.save(output / 'version_a' / f'{frame["radar_stem"]}.npy', pair)
        reliable_speed = np.where(reliable[i], speed, np.nan).astype(np.float32)
        np.savez_compressed(output / 'version_b_qc' / f'{frame["radar_stem"]}.npz',
                            doppler_reliable_ra_mps=reliable_speed,
                            reliable_mask=reliable[i], static_consensus_mask=consensus[i],
                            k_dense=d['k'].astype(np.int8), source_code=d['source'],
                            score_margin=d['margin'], corrected_folded_ra_mps=f['vf'],
                            direct_agreed=d['agreed'])
        db = 10 * np.log10(np.maximum(f['power'], 1) / global_peak)
        frame_results.append(dict(radar_stem=frame['radar_stem'], lag=d['lag'],
            previous_frame=frames[d['prev_index']]['radar_stem'] if d['prev_index'] is not None else None,
            next_frame=frames[d['next_index']]['radar_stem'] if d['next_index'] is not None else None,
            clip_log1p_min=f['clip_bounds'][0], clip_log1p_max=f['clip_bounds'][1],
            reliable_count=int(reliable[i].sum()), static_consensus_count=int(consensus[i].sum()),
            source_counts={name:int(np.sum(d['source'] == code)) for name,code in SOURCE.items()},
            all_invalid_direct_count=int(d['all_invalid'].sum()), joint_tie_count=int(d['joint_tie'].sum()),
            k_counts={str(k):int(n) for k,n in Counter(d['k'].ravel().tolist()).items()},
            velocity_min_mps=float(speed.min()), velocity_max_mps=float(speed.max()),
            velocity_p01_mps=float(np.quantile(speed,.01)),velocity_p99_mps=float(np.quantile(speed,.99)),
            power_strata={str(cut):dict(n=int(np.sum(db >= cut)),
                reliable=int(np.sum(reliable[i] & (db >= cut))),
                static_consensus=int(np.sum(consensus[i] & (db >= cut)))) for cut in CUTS_DB}))
    stack.flush(); del stack
    total = 50 * 256 * 107
    source_counts = Counter(int(v) for d in decisions for v in d['source'].ravel())
    result = dict(frame_count=50, shape=[50,2,256,107], all_cell_count=total,
                  match_lag=match_lag,
                  period_mps=period, speed_limit_mps=SPEED_LIMIT_MPS,
                  strong_threshold_used=False, match_preprocessing='log1p(max_D(arrDREA)) clipped per frame to 0.1-99.9 percentiles',
                  source_codes=SOURCE, source_counts={name:int(source_counts[code]) for name,code in SOURCE.items()},
                  source_fractions={name:float(source_counts[code]/total) for name,code in SOURCE.items()},
                  reliable_count=int(reliable.sum()), reliable_coverage=float(reliable.mean()),
                  static_consensus_count=int(consensus.sum()), static_consensus_coverage=float(consensus.mean()),
                  calibration=dict(threshold=threshold, automatic_threshold=calibrated_threshold, curve=curve,
                                   criterion=('Selected by the first-half-only lag/margin sweep; see parameter_selection.json'
                                              if margin_threshold is not None else
                                              '>=90% stable-anchor pseudo-reference agreement and >=50 anchors in first half')),
                  anchor_count=len(records), pseudo_reference=anchor_report(records, reliable, frames),
                  power_strata={str(cut):dict(n=sum(x['power_strata'][str(cut)]['n'] for x in frame_results),
                       reliable=sum(x['power_strata'][str(cut)]['reliable'] for x in frame_results),
                       static_consensus=sum(x['power_strata'][str(cut)]['static_consensus'] for x in frame_results))
                                for cut in CUTS_DB},
                  frames=frame_results,
                  caveats=['Version A is a dense k hypothesis field, not verified unfolded ground truth.',
                           'Static pseudo-reference and static-consensus QC share odometry.',
                           'Weak cells and dynamic targets have no independent k truth.',
                           'LiDAR label timestamps proxy radar frame times; no separate radar timestamp is available.'])
    (output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (output / 'frames.json').write_text(json.dumps([dict(frame_index=i,radar_stem=f['radar_stem'])
                                                    for i,f in enumerate(frames)], indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--match-lag', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--margin-threshold', type=float, default=None)
    args=parser.parse_args()
    run(args.output, args.match_lag, args.margin_threshold)
