"""Full-set motion labels; never infer current membership from Planner selection."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / 'collection/collector/agent_policy.py'
_spec = importlib.util.spec_from_file_location('_radarvla_motion_policy', _path)
_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_policy)
POLICY = _policy.POLICY
exclusion_reason = _policy.exclusion_reason
motion_metadata = _policy.motion_metadata


def rebuild_agents(record, current, metadata, anchor):
    """Rebuild every in-domain actor from immutable world snapshots and real futures."""
    from . import collection_adapter as adapter
    if current['episode_id'] != record['episode_id'] or current['carla_frame'] != record['carla_frame']:
        raise ValueError('Motion migration frame identity mismatch')
    if anchor['carla_frame'] != current['carla_frame']:
        raise ValueError('Motion migration anchor identity mismatch')
    if metadata['agent_label_range_m'] != 50 or metadata['episode_id'] != record['episode_id']:
        raise ValueError('Motion migration domain mismatch')
    geometry = adapter.collector_geometry()
    pos, rot = geometry.radar_pose(current['ego']['pose'], metadata['sensors']['semantic_lidar'])
    actors = current['actors']
    if len({a['actor_id'] for a in actors}) != len(actors):
        raise ValueError('Conflicting raw actor IDs')
    origin = current['ego']['pose']['position_m'][:2]
    yaw = current['ego']['pose']['rotation_deg'][2]
    kept, background = [], []
    for actor in actors:
        if not geometry.actor_intersects_fov(actor, pos, rot, 50.):
            continue
        reason = exclusion_reason(actor)
        label = adapter.agent50_label(adapter._agent(actor, current, origin, yaw,
            anchor['futures_by_actor_id'].get(str(actor['actor_id'])), [4, 9, 14, 19, 24, 29]), actor)
        if reason is None:
            kept.append(label)
        else:
            background.append(dict(id=label['id'], reason=reason, position=label['position'],
                                   size=label['size'], heading=label['heading'], type_id=actor['type_id']))
    new = copy.deepcopy(record)
    new['agents'] = kept
    new['risk_agents'] = copy.deepcopy(kept)
    new['planner_agent_ids'] = adapter._nearest_planner_ids(kept)
    new['ignored_agents'] = []
    new['agent_ignore_policy'] = 'none'
    new['agent_selection_policy'] = POLICY
    new['agent_selection'] = motion_metadata()
    new['background_agents'] = background  # Audit only; never feeds detection ignore masks.
    new['preparation_contract'] = POLICY
    return new


def attach_supervision(record):
    """Recompute all active Risk/Mode teachers after changing current membership."""
    from .local_risk import compute_local_risk, check_label, LOCAL_RISK_SCHEMA
    from .task_supervision import annotate_tasks
    new = copy.deepcopy(record)
    new.pop('supervision_protocol', None)
    new['risk_supervision_schema'] = LOCAL_RISK_SCHEMA
    risk, individual = compute_local_risk(new)
    for label in [risk, *individual.values()]:
        check_label(label)
    # Active compatibility aliases also refer to the motion-only set.
    new.update(local_risk=risk, local_agent_risk_labels=individual,
               risk_label=copy.deepcopy(risk), agent_risk_labels=copy.deepcopy(individual))
    for key in ('global_risk', 'global_risk_schema', 'legacy_teacher_mode',
                'legacy_planner_supervision_available'):
        new.pop(key, None)
    new['risk_provenance'] = dict(target_set='all current in-domain moving vehicles; no count truncation',
                                  agent_selection_policy=POLICY,
                                  local_agent_ids=[str(a['id']) for a in new['agents']])
    return annotate_tasks(new)


def validate_motion(record):
    if record.get('agent_selection_policy') != POLICY:
        return
    import math
    if record.get('agent_selection') != motion_metadata() or record.get('agent_ignore_policy') != 'none' or record.get('ignored_agents'):
        raise ValueError('Motion policy cannot ignore background objects')
    for agent in record['agents']:
        raw = dict(type_id=agent['type_id'], velocity_world_mps=[*agent['velocity'], 0.])
        # Rotation preserves horizontal speed; allow floating point roundoff at the boundary.
        if exclusion_reason(raw) not in (None, 'stationary') or math.hypot(*agent['velocity']) < .1 - 1e-12:
            raise ValueError('Illegal motion Agent positive')
    ids = {str(a['id']) for a in record['agents']}
    if {str(a['id']) for a in record['risk_agents']} != ids:
        raise ValueError('Motion Risk must use the complete motion set')
    if set(record['local_agent_risk_labels']) != ids or set(record['agent_risk_labels']) != ids:
        raise ValueError('Stale motion risk target IDs')
