"""Score annotated instruction constraints in SI units without guessing semantics.

Free-form text alone is not a reliable success oracle. Dataset authors attach
machine-checkable constraints to language.constraints; these labels never enter
the online planner. Missing constraints/horizons remain explicitly unscored.
"""
from __future__ import annotations

import math
import numpy as np
import torch

from .curves import spline_basis


def evaluate_instruction(record, plan, step_s=.05):
    constraints = record.get('language', {}).get('constraints', [])
    if not isinstance(constraints, list):
        raise ValueError('language.constraints must be a list')
    if not math.isfinite(step_s) or step_s <= 0:
        raise ValueError('step_s must be finite and positive')
    result = dict(success=None, coverage=0., checks=[],
                  semantics='annotated instruction constraints, evaluated offline; unknown text is unscored')
    if not constraints:
        return result
    supported = {'speed_limit', 'stop_by', 'goal_region', 'heading_at'}
    if any(c.get('type') not in supported for c in constraints):
        raise ValueError('Unsupported instruction constraint type')
    if not plan or not plan.get('valid'):
        result['checks'] = [dict(type=c['type'], passed=None) for c in constraints]
        return result
    times = np.asarray(plan['ego_times_s'], float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(np.r_[0., times]) <= 0):
        raise ValueError('plan needs positive increasing times')
    horizon = float(times[-1])
    raw = plan['ego_trajectory']
    if len(raw) != len(times) or any(point is None for point in raw):
        result['checks'] = [dict(type=c['type'], passed=None) for c in constraints]
        return result
    positions = np.vstack(([0., 0.], np.asarray(raw, float)))
    knots = np.r_[0., times]
    controls = plan.get('ego_control_points')
    control_times = plan.get('control_times_s')
    if controls is not None:
        controls = torch.as_tensor(controls, dtype=torch.float64)
        control_times = torch.as_tensor(control_times, dtype=torch.float64)
        if controls.ndim != 2 or controls.shape != (len(control_times), 2):
            raise ValueError('ego spline control dimensions disagree')
    def sample(query):
        query = np.asarray(query, float)
        if controls is not None:
            values = (spline_basis(control_times, query) @ controls).numpy()
            velocity = (spline_basis(control_times, query, derivative=1) @ controls).numpy()
            return values, velocity
        values = np.stack([np.interp(query, knots, positions[:, d]) for d in (0, 1)], -1)
        index = np.searchsorted(knots, query, side='right').clip(1, len(knots)-1)-1
        velocity = np.diff(positions, axis=0)[index] / np.diff(knots)[index, None]
        return values, velocity
    def number(c, key, nonnegative=True):
        value = float(c[key])
        if not math.isfinite(value) or (nonnegative and value < 0):
            raise ValueError(f'constraint {key} has invalid value')
        return value
    for constraint in constraints:
        kind = constraint['type']; passed = None
        if kind in ('speed_limit', 'stop_by'):
            threshold = number(constraint, 'max_mps')
            start = 0. if kind == 'speed_limit' else number(constraint, 'time_s')
            end = number(constraint, 'until_s') if 'until_s' in constraint else float(record.get('future_times_s', [horizon])[-1])
            if end < start and start <= horizon:
                raise ValueError('constraint until_s precedes its start')
            if start <= horizon and end <= horizon:
                query = np.unique(np.r_[start, np.arange(start, end, step_s), end])
                _, velocity = sample(query)
                passed = bool((np.linalg.norm(velocity, axis=-1) <= threshold+1e-6).all())
        elif kind == 'goal_region':
            at = number(constraint, 'at_s')
            lower, upper = np.asarray(constraint['min_xy'], float), np.asarray(constraint['max_xy'], float)
            if lower.shape != (2,) or upper.shape != (2,) or not np.isfinite([lower, upper]).all() or np.any(lower > upper):
                raise ValueError('goal_region requires finite ordered min_xy/max_xy')
            if at <= horizon:
                point, _ = sample([at]); passed = bool(((point[0] >= lower) & (point[0] <= upper)).all())
        elif kind == 'heading_at':
            at = number(constraint, 'at_s')
            target = number(constraint, 'yaw_rad', nonnegative=False)
            tolerance = number(constraint, 'tolerance_rad')
            if at <= horizon and plan.get('ego_yaw') is not None:
                yaw = np.asarray(plan['ego_yaw'], float)
                if yaw.shape != times.shape or not np.isfinite(yaw).all():
                    raise ValueError('ego_yaw must be finite and match plan times')
                observed_yaw = np.interp(at, np.r_[0., times], np.unwrap(np.r_[0., yaw]))
                delta = observed_yaw-target
                passed = abs(math.atan2(math.sin(delta), math.cos(delta))) <= tolerance
        result['checks'].append(dict(type=kind, passed=passed))
    scored = [c['passed'] for c in result['checks'] if c['passed'] is not None]
    result['coverage'] = len(scored)/len(constraints)
    if False in scored:
        result['success'] = False
    elif len(scored) == len(constraints):
        result['success'] = True
    return result


def aggregate_instruction_metrics(rows):
    scored = [r['success'] for r in rows if r['success'] is not None]
    return dict(samples=len(rows), scored_samples=len(scored),
                success_rate=float(np.mean(scored)) if scored else None,
                mean_constraint_coverage=float(np.mean([r['coverage'] for r in rows])) if rows else None)
