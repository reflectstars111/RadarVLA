"""Validation diagnostics; checkpoint selection never consults the test split."""
from __future__ import annotations

import math
import random

import numpy as np

from .metrics import planning_metrics


def select_generation_indices(modes, limit=32, seed=42):
    """Fixed, mode-stratified audit subset; report its IDs rather than hide bias."""
    if limit < 0:
        raise ValueError('generation sample budget must be nonnegative')
    if any(mode not in ('short', 'long', None) for mode in modes):
        raise ValueError('unknown teacher mode')
    rng = random.Random(seed)
    groups = [[i for i, mode in enumerate(modes) if mode == value]
              for value in ('short', 'long', None)]
    for group in groups:
        rng.shuffle(group)
    result = []
    while len(result) < min(limit, len(modes)):
        for group in groups:
            if group and len(result) < limit:
                result.append(group.pop())
    return sorted(result)


def generation_diagnostics(rows, samples, teacher_modes, short_horizon_s=1.):
    """Compare generation with GT and CV on identical observed time points.

    Required coverage is determined by the *teacher*, not the generated mode:
    generating a short plan cannot improve selection by omitting late errors.
    The regular full-horizon metrics are retained alongside these diagnostics.
    """
    samples, teacher_modes = list(samples), list(teacher_modes)
    if len(samples) != len(teacher_modes):
        raise ValueError('one teacher mode per sample is required')
    truth = {s['sample_id']: (s, mode) for s, mode in zip(samples, teacher_modes)}
    identifiers = [row['sample_id'] for row in rows]
    if len(truth) != len(samples) or len(identifiers) != len(truth) or set(identifiers) != set(truth):
        raise ValueError('exactly one prediction per truth sample is required')
    confusion = {mode: dict(short=0, long=0, invalid=0) for mode in ('short', 'long')}
    baseline, required_errors, required_cv_errors, paired_cv_errors = [], [], [], []
    required_points, covered, known_modes = 0, 0, 0
    for row in rows:
        sample, teacher = truth[row['sample_id']]
        times = np.asarray(sample['future_times_s'], dtype=float)
        xy = np.asarray(sample['ego_future'], dtype=float)
        valid = np.asarray(sample['ego_future_mask'], dtype=bool) & np.isfinite(xy).all(-1)
        velocity = np.asarray(sample.get('ego_velocity', [np.nan, np.nan]), dtype=float)
        velocity_known = np.asarray(sample.get('ego_velocity_mask', True), dtype=bool).all()
        if velocity_known and velocity.shape == (2,) and np.isfinite(velocity).all():
            baseline.append(dict(sample_id=row['sample_id'], plan=dict(valid=True, mode='long',
                ego_trajectory=(times[:, None] * velocity).tolist(), ego_times_s=times.tolist())))
        plan = row.get('plan') or {}
        predicted_mode = plan.get('mode') if plan.get('valid') else None
        if teacher in confusion:
            known_modes += 1
            confusion[teacher][predicted_mode if predicted_mode in ('short', 'long') else 'invalid'] += 1
        else:
            continue
        required = valid & ((times <= short_horizon_s + 1e-6) if teacher == 'short' else True)
        required_points += int(required.sum())
        cv_error = None
        if velocity_known and velocity.shape == (2,) and np.isfinite(velocity).all():
            cv_error = np.linalg.norm(times[:, None] * velocity - xy, axis=-1)
            required_cv_errors.extend(cv_error[required].tolist())
        if not plan.get('valid'):
            continue
        found = set()
        for point, moment in zip(plan.get('ego_trajectory', []), plan.get('ego_times_s') or []):
            index = np.flatnonzero(np.isclose(times, moment, rtol=0, atol=1e-5))
            if point is None or len(index) != 1:
                continue
            i = int(index[0])
            point = np.asarray(point, dtype=float)
            if i in found or not required[i] or point.shape != (2,) or not np.isfinite(point).all():
                continue
            found.add(i)
            required_errors.append(float(np.linalg.norm(point - xy[i])))
            if cv_error is not None:
                paired_cv_errors.append(float(cv_error[i]))
            covered += 1
    short_true = sum(confusion['short'].values())
    short_pred = confusion['short']['short'] + confusion['long']['short']
    correct = confusion['short']['short'] + confusion['long']['long']
    cv = planning_metrics(baseline, samples)
    cv.update(required_points=len(required_cv_errors),
              required_point_coverage=len(required_cv_errors)/required_points if required_points else None,
              required_ego_ade_m=float(np.mean(required_cv_errors)) if required_cv_errors else None,
              generated_point_overlap=len(paired_cv_errors),
              generated_point_overlap_ade_m=float(np.mean(paired_cv_errors)) if paired_cv_errors else None)
    return dict(**planning_metrics(rows, samples), mode_confusion=confusion,
                teacher_mode_counts={mode: sum(values.values()) for mode, values in confusion.items()},
                mode_accuracy=correct / known_modes if known_modes else None,
                short_recall=confusion['short']['short'] / short_true if short_true else None,
                short_precision=confusion['short']['short'] / short_pred if short_pred else None,
                required_points=required_points,
                required_point_coverage=covered / required_points if required_points else None,
                required_ego_ade_m=float(np.mean(required_errors)) if required_errors else None,
                constant_velocity=cv,
                selection_semantics='maximize schema validity, then teacher-required coverage, then minimize covered ADE')


def quality_scores(validation):
    """Lexicographic, lower-is-better scores for separate named checkpoints.

    They complement best.pt (loss). They are not a closed-loop safety ranking.
    Undefined metrics never win by being interpreted as zero error.
    """
    result = {}
    detection = validation.get('detection') or {}
    values = [detection.get('f1'), detection.get('count_mae')]
    if all(value is not None and math.isfinite(value) for value in values):
        result['detection'] = (-float(values[0]), float(values[1]))
    generation = validation.get('free_generation') or {}
    values = [generation.get(name) for name in
              ('schema_valid_rate', 'required_point_coverage', 'required_ego_ade_m')]
    if all(value is not None and math.isfinite(value) for value in values):
        result['planning'] = (1. - float(values[0]), 1. - float(values[1]), float(values[2]))
    return result
