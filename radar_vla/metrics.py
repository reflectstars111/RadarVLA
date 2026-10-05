"""Masked, held-out physical and risk metrics. Undefined metrics are JSON null."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def planning_metrics(rows, truth_samples):
    """Free-generation errors with explicit coverage and full-horizon FDE.

    SHORT plans are not silently counted as successful full-horizon predictions.
    No closed-loop collision/safety claim follows from this offline comparison.
    """
    truth = {sample['sample_id']: sample for sample in truth_samples}
    errors, final_errors, lengths = [], [], []
    valid_count, available, covered = 0, 0, 0
    latency = []
    modes = {'short': 0, 'long': 0}
    for row in rows:
        sample = truth[row['sample_id']]
        xy = np.asarray(sample['ego_future'], dtype=float)
        times = np.asarray(sample['future_times_s'], dtype=float)
        valid = np.asarray(sample['ego_future_mask'], dtype=bool) & np.isfinite(xy).all(-1)
        available += int(valid.sum())
        lengths.append(row.get('generated_tokens', 0))
        elapsed = row.get('amortized_inference_seconds')
        if elapsed is not None and np.isfinite(elapsed) and elapsed >= 0:
            latency.append(float(elapsed))
        plan = row.get('plan') or {}
        if not plan.get('valid'):
            continue
        valid_count += 1
        mode = plan.get('mode')
        if mode in modes:
            modes[mode] += 1
        for point, moment in zip(plan['ego_trajectory'], plan.get('ego_times_s') or []):
            if point is None:
                continue
            point = np.asarray(point, dtype=float)
            index = np.flatnonzero(np.isclose(times, moment, rtol=0, atol=1e-5))
            if point.shape != (2,) or not np.isfinite(point).all() or len(index) != 1:
                continue
            index = int(index[0])
            if not valid[index]:
                continue
            distance = float(np.linalg.norm(point - xy[index]))
            errors.append(distance)
            covered += 1
            if index == len(times) - 1:
                final_errors.append(distance)
    return dict(samples=len(rows), schema_valid_rate=valid_count / len(rows) if rows else None,
                mode_counts=modes, average_output_tokens=float(np.mean(lengths)) if lengths else None,
                average_amortized_inference_seconds=float(np.mean(latency)) if latency else None,
                inference_latency_count=len(latency),
                ego_ade_m=float(np.mean(errors)) if errors else None, ego_ade_points=len(errors),
                ego_final_horizon_fde_m=float(np.mean(final_errors)) if final_errors else None,
                ego_final_horizon_fde_samples=len(final_errors),
                ego_point_coverage=covered / available if available else None,
                note='Only valid generated points scored; report coverage with errors; no closed-loop evaluation')


def binary_metrics(probabilities, targets, bins=10):
    p, y = np.asarray(probabilities, dtype=float), np.asarray(targets, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError('Expected matching one-dimensional probabilities and targets')
    if not np.isfinite(p).all() or not np.isfinite(y).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probabilities or targets')
    if np.any((y != 0) & (y != 1)) or bins < 1:
        raise ValueError('Binary labels and positive calibration bins required')
    result = dict(samples=len(p), positives=int(y.sum()), auroc=None,
                  average_precision=None, brier=None, ece=None)
    if not len(p):
        return result
    result['brier'] = float(np.mean((p - y) ** 2))
    ece = 0.
    bucket = np.minimum((p * bins).astype(int), bins - 1)
    for i in range(bins):
        take = bucket == i
        if take.any():
            ece += take.mean() * abs(p[take].mean() - y[take].mean())
    result['ece'] = float(ece)
    positive, negative = int(y.sum()), len(y) - int(y.sum())
    order = np.argsort(-p, kind='stable')
    p, y = p[order], y[order]
    group_ends = np.r_[np.flatnonzero(np.diff(p) != 0), len(p) - 1]
    tp = np.r_[0., y.cumsum()[group_ends]]
    fp = np.r_[0., (1 - y).cumsum()[group_ends]]
    if positive:
        precision = tp[1:] / (tp[1:] + fp[1:])
        result['average_precision'] = float(np.sum(np.diff(tp / positive) * precision))
    if positive and negative:
        result['auroc'] = float(np.trapz(tp / positive, fp / negative))
    return result


class MetricAccumulator:
    def __init__(self):
        self.samples = 0
        self.risk_p = [[] for _ in range(3)]
        self.risk_y = [[] for _ in range(3)]
        self.errors = {name: [] for name in ('dmin', 'areq', 'position', 'radial',
                                            'velocity_squared', 'agent_ade', 'agent_fde',
                                            'state_derived_radial', 'doppler_projection')}
        self.radial_target_sources = dict(measured_radar=0, legacy_state=0)

    def update(self, prediction, batch):
        def array(value):
            return value.detach().float().cpu().numpy()
        risk = array(prediction['risk'])
        target, mask = array(batch['risk_target']), array(batch['risk_mask']).astype(bool)
        mask &= np.isfinite(target)
        self.samples += len(risk)
        for i in range(3):
            self.risk_p[i].extend(risk[mask[:, i], i].tolist())
            self.risk_y[i].extend(target[mask[:, i], i].tolist())
        for i, name in ((3, 'dmin'), (4, 'areq')):
            self.errors[name].extend(abs(risk[mask[:, i], i] - target[mask[:, i], i]).tolist())
        states, truth = array(prediction['agent_state']), array(batch['agent_state'])
        present = array(batch['agent_mask']).astype(bool)
        state_mask = array(batch['agent_state_mask']).astype(bool) if 'agent_state_mask' in batch else np.ones_like(truth, dtype=bool)
        state_mask &= np.isfinite(truth)
        present &= state_mask[:, :, :2].all(-1)
        if 'agent_supervision_mask' in batch:
            present &= array(batch['agent_supervision_mask']).astype(bool)[:, None]
        future, gt_future = array(prediction['agent_future']), array(batch['agent_future'])
        future_mask = array(batch['agent_future_mask']).astype(bool) & np.isfinite(gt_future).all(-1)
        for b in range(len(states)):
            ids = np.flatnonzero(present[b])
            if not len(ids):
                continue
            cost = np.linalg.norm(states[b, :, None, :2] - truth[b, ids][None, :, :2], axis=-1)
            q, k = linear_sum_assignment(cost)
            gt = ids[k]
            delta = states[b, q] - truth[b, gt]
            self.errors['position'].extend(np.linalg.norm(delta[:, :2], axis=-1).tolist())
            derived = abs(delta[state_mask[b, gt, 2], 2]).tolist()
            self.errors['state_derived_radial'].extend(derived)
            if 'agent_radial_observed' in batch:
                measured = array(batch['agent_radial_observed'])[b, gt]
                valid_radial = array(batch['agent_radial_mask']).astype(bool)[b, gt] & np.isfinite(measured)
                self.errors['radial'].extend(abs(states[b, q, 2][valid_radial] - measured[valid_radial]).tolist())
                self.radial_target_sources['measured_radar'] += int(valid_radial.sum())
                has_projection = 'agent_doppler_projection' in batch
                los = array(batch.get('agent_doppler_projection', batch['agent_los']))[b, gt]
                sensor_velocity = (array(batch['agent_sensor_velocity'])[b, gt] if 'agent_sensor_velocity' in batch
                                   else np.broadcast_to(array(batch['ego_velocity'])[b], los.shape))
                norm = np.linalg.norm(los, axis=-1)
                valid_projection = (valid_radial & np.isfinite(los).all(-1)
                                    & np.isfinite(sensor_velocity).all(-1))
                if not has_projection:
                    valid_projection &= norm > 1e-6
                    los = los / np.maximum(norm[:, None], 1e-6)
                relative = states[b, q, 3:5] - sensor_velocity
                projection = np.sum(relative * los, axis=-1)
                self.errors['doppler_projection'].extend(abs(projection[valid_projection] - measured[valid_projection]).tolist())
            else:
                self.errors['radial'].extend(derived)
                self.radial_target_sources['legacy_state'] += len(derived)
            self.errors['velocity_squared'].extend(np.sum(delta[state_mask[b, gt, 3:5].all(-1), 3:5] ** 2, axis=-1).tolist())
            distance = np.linalg.norm(future[b, q] - gt_future[b, gt], axis=-1)
            valid = future_mask[b, gt]
            self.errors['agent_ade'].extend(distance[valid].tolist())
            self.errors['agent_fde'].extend(distance[valid[:, -1], -1].tolist())

    def compute(self):
        result = dict(samples=self.samples, collision={
            f'{i + 1}s': binary_metrics(self.risk_p[i], self.risk_y[i]) for i in range(3)})
        for name, values in self.errors.items():
            key = 'velocity_rmse_mps' if name == 'velocity_squared' else name + '_mae'
            value = float(np.mean(values)) if values else None
            result[key] = float(np.sqrt(value)) if name == 'velocity_squared' and value is not None else value
            result[key + '_count'] = len(values)
        result['radial_target_sources'] = dict(self.radial_target_sources)
        result['physical_assignment'] = 'GT-matched position Hungarian; not detection AP'
        result['distance_units'] = 'm; areq m/s^2; velocities m/s'
        return result
