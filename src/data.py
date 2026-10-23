"""Validated canonical RadarVLA datasets with physical sensor observations."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from .task_supervision import TASKWISE_PROTOCOL, LEGACY_PROTOCOL, supervision_protocol, validate_tasks, MODE_INDEX
from .geometry import build_risk_labels
from .local_risk import LOCAL_RISK_SCHEMA, LOCAL_RISK_SCHEMAS, LEGACY_RISK_SCHEMA, risk_schema, check_label
from .risk_contract import horizon_schema, SHORT_HORIZONS
from .coordinates import validate_transform
from .radar_processing import radar_geometry, associate_doppler
from .records import road_targets
from .categories import category_name, detection_classes, record_class_schema
from .annotation_domain import detection_domain_values

SCHEMA_VERSION = 'radar_vla_v2'
SUPPORTED_SCHEMAS = ('radar_vla_v1', SCHEMA_VERSION, 'radar_vla_v3')


def _finite(value, location):
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f'{location}.{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite(item, f'{location}[{i}]')
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f'{location} must be finite')


def _array(value, shape, name, positive=False):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric array') from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite with shape {shape}')
    if positive and np.any(result <= 0):
        raise ValueError(f'{name} must be positive')
    return result


def _validity(values, length, name):
    if not isinstance(values, list) or len(values) != length or any(type(v) is not bool for v in values):
        raise ValueError(f'{name} must be a boolean list of length {length}')


def _validate_record(record, index, supervision=True):
    name = f'record {index}'
    if not isinstance(record, dict):
        raise ValueError(f'{name} must be an object')
    _finite(record, name)
    detection_domain_values(record.get('detection_annotation_domain'))
    taskwise = supervision_protocol(record) == TASKWISE_PROTOCOL
    class_schema = record_class_schema(record)
    if class_schema == 'agent_v1' and record.get('detection_annotation_domain') is None:
        raise ValueError('agent50_v1 requires an explicit annotation domain')
    if record.get('schema_version') not in SUPPORTED_SCHEMAS:
        raise ValueError(f'{name} schema_version must be {SCHEMA_VERSION}')
    for key in ('sample_id', 'scene_id'):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError(f'{name} {key} must be a nonempty string')
    if record.get('split') not in ('train', 'val', 'test'):
        raise ValueError(f'{name} invalid split')
    timestamp = record.get('timestamp_s')
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
        raise ValueError(f'{name} timestamp_s must be finite')
    radar = record.get('sensors', {}).get('radar', {})
    for key in ('power', 'unfolded_doppler'):
        paths = radar.get(key)
        if isinstance(paths, str):
            paths = [paths]
        if (not isinstance(paths, list) or not paths or
                any(not isinstance(path, str) or not path for path in paths)):
            raise ValueError(f'{name} radar requires {key} paths; folded_doppler is not unfolded velocity')
        if len(paths) not in (1, len(radar.get('time_offsets_s', []))):
            raise ValueError(f'{name} radar {key} path count must match history')
    for key in ('range_m', 'azimuth_rad', 'time_offsets_s'):
        values = radar.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f'{name} radar {key} must be a nonempty list')
        values = _array(values, (len(values),), f'{name} radar {key}')
        if np.any(np.diff(values) <= 0):
            raise ValueError(f'{name} radar {key} must increase strictly')
        if key == 'range_m' and np.any(values < 0):
            raise ValueError(f'{name} range_m must be nonnegative')
        if key == 'time_offsets_s' and (np.any(values > 1e-8) or (record['schema_version'] == 'radar_vla_v1' and abs(values[-1]) > 1e-8)):
            raise ValueError(f'{name} time_offsets_s must contain causal offsets (v1 must end at zero)')
    times = record.get('future_times_s')
    if not isinstance(times, list) or not times:
        raise ValueError(f'{name} future_times_s must be nonempty')
    times = _array(times, (len(times),), f'{name} future_times_s', positive=True)
    if np.any(np.diff(times) <= 0):
        raise ValueError(f'{name} future_times_s must increase strictly')
    horizon = len(times)
    ego = record.get('ego', {})
    for key in ('velocity', 'acceleration', 'box_size'):
        _array(ego.get(key), (2,), f'{name} ego.{key}', positive=key == 'box_size')
    _array(ego.get('yaw_rate'), (), f'{name} ego.yaw_rate')
    if 'future_xy' in ego:
        _array(ego['future_xy'], (horizon, 2), f'{name} ego.future_xy')
    if 'future_valid' in ego:
        _validity(ego['future_valid'], horizon, f'{name} ego.future_valid')
        if 'future_xy' not in ego and any(ego['future_valid']):
            raise ValueError(f'{name} ego future_valid cannot label absent future_xy')
    agents = record.get('agents', [] if not supervision else None)
    if not isinstance(agents, list):
        raise ValueError(f'{name} requires agents list with known tracking coverage; [] means observed empty')
    ignored=record.get('ignored_agents',[])
    if not isinstance(ignored,list):raise ValueError('ignored_agents must be a list')
    ignored_ids=set()
    for a in ignored:
        if str(a['id']) in ignored_ids:raise ValueError('Duplicate ignored Agent ID')
        ignored_ids.add(str(a['id']))
        _array(a.get('position'),(2,), 'ignored position')
        _array(a.get('size'),(2,), 'ignored size',positive=True)
        _array(a.get('heading'),(), 'ignored heading')
    if ignored and class_schema!='agent_v1':raise ValueError('Ignored-Agent policy requires agent_v1')
    if {str(a['id']) for a in agents}&ignored_ids:raise ValueError('Positive and ignored Agent IDs overlap')
    identifiers = set()
    for agent in agents:
        if not isinstance(agent, dict) or not isinstance(agent.get('id'), (str, int)):
            raise ValueError(f'{name} agents require id')
        if str(agent['id']) in identifiers:
            raise ValueError(f'{name} duplicate agent id')
        identifiers.add(str(agent['id']))
        category = category_name(agent)
        if class_schema == 'agent_v1' and category != 'agent':
            raise ValueError('agent50_v1 requires category=agent for every retained instance')
        if class_schema != 'agent_v1' and category == 'agent':
            raise ValueError('category=agent requires explicit grounding_label_schema')
        for key in ('position', 'velocity', 'size'):
            _array(agent.get(key), (2,), f'{name} agent.{key}', positive=key == 'size')
        _array(agent.get('heading'), (), f'{name} agent.heading')
        if 'future_xy' in agent:
            _array(agent['future_xy'], (horizon, 2), f'{name} agent.future_xy')
            _array(agent.get('future_yaw'), (horizon,), f'{name} agent.future_yaw')
        if 'future_valid' in agent:
            _validity(agent['future_valid'], horizon, f'{name} agent.future_valid')
            if 'future_xy' not in agent and any(agent['future_valid']):
                raise ValueError(f'{name} agent future_valid cannot label absent future_xy')
    if 'planner_agent_ids' in record:
        chosen = record['planner_agent_ids']
        if (not isinstance(chosen, list) or len(chosen) > 8 or
                len({str(value) for value in chosen}) != len(chosen) or
                any(str(value) not in identifiers for value in chosen)):
            raise ValueError(f'{name} planner_agent_ids must select at most eight current agents')
    corridor = record.get('road_supervision')
    if record['schema_version'] == 'radar_vla_v3':
        segments = record.get('road_segments')
        if not isinstance(segments, list) or (not segments and not taskwise):
            raise ValueError(f'{name} v3 requires nonempty road_segments')
        for ordinal, segment in enumerate(segments):
            if segment.get('ordinal') != ordinal:
                raise ValueError(f'{name} road segments must have local consecutive ordinals')
            points = segment.get('centerline_ego_m')
            if not isinstance(points, list) or (len(points) < 2 and not taskwise):
                raise ValueError(f'{name} road segment needs two sourced points')
            _array(points, (len(points), 2), f'{name} segment {ordinal} centerline')
            if type(segment.get('width_valid')) is not bool:
                raise ValueError(f'{name} road segment width_valid must be boolean')
            if segment['width_valid']:
                _array(segment.get('width_m'), (), f'{name} segment width', positive=True)
    if corridor is not None:
        if not isinstance(corridor, dict):
            raise ValueError(f'{name} road_supervision must be an object')
        points = _array(corridor.get('road_centerline'), (16, 2), f'{name} road_centerline')
        _validity(corridor.get('road_centerline_mask'), 16, f'{name} road_centerline_mask')
        width = _array(corridor.get('road_width'), (1,), f'{name} road_width')
        _validity(corridor.get('road_mask'), 1, f'{name} road_mask')
        if not all(corridor['road_centerline_mask']) or not corridor['road_mask'][0] or width[0] <= 0:
            raise ValueError(f'{name} explicit road supervision must be complete and positive')
    if record['schema_version'] in (SCHEMA_VERSION, 'radar_vla_v3'):
        poses = radar.get('poses_current_ego')
        if not isinstance(poses, list) or len(poses) != len(radar['time_offsets_s']):
            raise ValueError(f'{name} v2 requires radar poses_current_ego for every frame')
        for pose in poses:
            validate_transform(pose, 'radar pose')
        _array(radar.get('sensor_velocity_current_ego'), (len(poses), 2), 'sensor velocities')
        if record.get('coordinate_frame') != 'current_ego':
            raise ValueError('prepared v2 coordinate_frame must be current_ego')
    if not isinstance(record.get('language', {}).get('instruction'), str):
        raise ValueError(f'{name} language.instruction must be a string')
    if 'tracking_coverage_valid' in record:
        _validity(record['tracking_coverage_valid'], horizon, 'tracking_coverage_valid')
    validate_tasks(record)
    cache = record.get('agent_risk_labels', {})
    if not isinstance(cache, dict) or any(key not in identifiers | ignored_ids for key in cache):
        raise ValueError('agent_risk_labels must map current or explicitly ignored agent ids to risk targets')
    schema = risk_schema(record)
    if schema == LEGACY_RISK_SCHEMA and any(k in record for k in ('local_risk','local_agent_risk_labels')):
        raise ValueError('Local Risk fields require an explicit risk_supervision_schema')
    all_risks = list(cache.values())
    if schema in LOCAL_RISK_SCHEMAS:
        from .moving_agents import POLICY, validate_motion
        validate_motion(record)
        if record.get('agent_selection_policy') != POLICY and (record.get('global_risk_schema') != LEGACY_RISK_SCHEMA or record.get('global_risk') != record.get('risk_label')):
            raise ValueError('Versioned Global Risk must preserve legacy risk_label exactly')
        local_cache = record.get('local_agent_risk_labels')
        if not isinstance(local_cache, dict) or set(local_cache) != identifiers:
            raise ValueError('Local Risk requires labels for every retained Agent')
        if not isinstance(record.get('local_risk'), dict):
            raise ValueError('Versioned Local Risk requires local_risk; legacy fallback forbidden')
        if record['local_risk'].get('label_semantics') != schema:
            raise ValueError('Local Risk label semantics mismatch')
        for label in [record['local_risk'], *local_cache.values()]:
            check_label(label)
        all_risks += [record['local_risk'], *local_cache.values()]

    if 'risk_label' in record:
        all_risks.append(record['risk_label'])
    for risk in all_risks:
        if not isinstance(risk, dict):
            raise ValueError('cached risk label must be an object')
        pcol = _array(risk.get('pcol'), (3,), f'{name} risk_label.pcol')
        if np.any((pcol != 0) & (pcol != 1)):
            raise ValueError(f'{name} pcol targets must be binary 0 or 1')
        _validity(risk.get('pcol_mask'), 3, f'{name} risk_label.pcol_mask')
        if np.any(np.diff(pcol[np.asarray(risk['pcol_mask'], bool)]) < 0):
            raise ValueError(f'{name} known cumulative collision targets must be nondecreasing')
        for key in ('dmin', 'areq'):
            val = _array(risk.get(key), (), f'{name} risk_label.{key}')
            if val < 0 or type(risk.get(f'{key}_mask')) is not bool:
                raise ValueError(f'{name} invalid risk_label.{key}')


def load_manifest(manifest: Path, *, supervision=True):
    """Validate metadata, duplicate IDs and scene-disjoint splits across all rows.

    Radar arrays are checked when loaded by RadarDataset; no pickle is allowed.
    """
    manifest = Path(manifest)
    records, identifiers, scenes = [], set(), {}
    with manifest.open(encoding='utf-8-sig') as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{manifest}:{lineno}: invalid JSON') from exc
            _validate_record(record, lineno, supervision=supervision)
            if record['sample_id'] in identifiers:
                raise ValueError(f'duplicate sample_id: {record["sample_id"]}')
            identifiers.add(record['sample_id'])
            scene, split = record['scene_id'], record['split']
            if scene in scenes and scenes[scene] != split:
                raise ValueError(f'scene {scene} occurs across splits; data leakage')
            scenes[scene] = split
            records.append(record)
    if not records:
        raise ValueError('manifest is empty')
    return records


class RadarDataset(Dataset):
    def __init__(self, manifest: Path, split: str, max_agents=None, *, supervision=True):
        self.manifest = Path(manifest).resolve()
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if max_agents is not None and (type(max_agents) is not int or max_agents < 1):
            raise ValueError('max_agents must be positive')
        self.max_agents = max_agents
        self.supervision = supervision
        self.records = [r for r in load_manifest(self.manifest, supervision=supervision) if r['split'] == split]
        if not self.records:
            raise ValueError(f'no records for split {split}')
        if supervision and max_agents is not None and any(len(r.get('agents', [])) > max_agents for r in self.records):
            raise ValueError('record exceeds max_agents; increase capacity (silent truncation is forbidden)')
        schemas = {record_class_schema(r) for r in self.records}
        if len(schemas) != 1:
            raise ValueError('Mixed detection class schemas in one dataset split')
        self.class_schema = schemas.pop()
        self.class_names = detection_classes(self.class_schema)
        risk_schemas = {risk_schema(r) for r in self.records}
        if len(risk_schemas) != 1:
            raise ValueError('Mixed Risk supervision schemas in dataset')
        self.risk_schema = risk_schemas.pop()
        protocols = {supervision_protocol(r) for r in self.records}
        if len(protocols) != 1:raise ValueError('Mixed task supervision protocols')
        self.supervision_protocol = protocols.pop()
        if len({r.get('agent_selection_policy', 'legacy_all_agents') for r in self.records}) != 1:
            raise ValueError('Mixed Agent selection policies')
        self._risk_cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        radar = r['sensors']['radar']
        shape = tuple(len(radar[k]) for k in ('time_offsets_s', 'range_m', 'azimuth_rad'))
        arrays = []
        for key in ('power', 'unfolded_doppler'):
            paths = radar[key]
            if isinstance(paths, list):
                array = np.stack([np.load(self.manifest.parent / path, allow_pickle=False)
                                  for path in paths])
            else:
                array = np.load(self.manifest.parent / paths, allow_pickle=False)
            if (array.shape != shape or not np.issubdtype(array.dtype, np.number)
                    or np.iscomplexobj(array)):
                raise ValueError(f'{r["sample_id"]} {key} must be real numeric with shape {shape}')
            if not np.isfinite(array).all():
                raise ValueError(f'{r["sample_id"]} {key} must be finite')
            if key == 'power' and (array < 0).any():
                raise ValueError('power must be nonnegative linear power (not dB)')
            arrays.append(array.astype(np.float32))
        ego = r['ego']
        ego_velocity = np.asarray(ego['velocity'], np.float32)
        agents = (sorted(r.get('agents', []), key=lambda a: (np.linalg.norm(a['position']), str(a['id'])))
                  if self.supervision else [])
        planner_ids = {str(value) for value in r.get('planner_agent_ids', [agent['id'] for agent in agents])}
        slots = self.max_agents if self.max_agents is not None else len(agents)
        known_agents = bool(r.get('agent_supervision_available', 'agents' in r)) and self.supervision
        poses = np.asarray(radar.get('poses_current_ego', [np.eye(4).tolist()] * shape[0]), np.float32)
        sensor_velocities = np.asarray(radar.get('sensor_velocity_current_ego', [ego_velocity.tolist()] * shape[0]), np.float32)
        positions, line_of_sight = radar_geometry(radar['range_m'], radar['azimuth_rad'], poses)
        doppler_valid = np.ones(shape, bool)
        if 'doppler_valid' in radar:
            doppler_valid = np.load(self.manifest.parent / radar['doppler_valid'], allow_pickle=False)
            if doppler_valid.shape != shape or not np.isin(doppler_valid, [0, 1]).all():
                raise ValueError('doppler_valid must have radar shape and binary values')
            doppler_valid = doppler_valid.astype(bool)
        association = associate_doppler(arrays[0][-1], arrays[1][-1], radar['range_m'], radar['azimuth_rad'],
                                        poses[-1], agents, doppler_valid=doppler_valid[-1],
                                        min_power=radar.get('association_min_power', 0.),
                                        min_cells=radar.get('association_min_cells', 1))
        synchronous_radar = abs(float(radar['time_offsets_s'][-1])) <= 1e-6
        if not synchronous_radar:
            # Current annotation boxes cannot label an earlier measurement without synchronized actor tracks.
            association['mask'][:] = False
        horizon = len(r['future_times_s'])
        states = np.zeros((slots, 5), np.float32)
        active = np.zeros(slots, bool)
        classes = np.full(slots, -1, np.int64)
        sizes = np.zeros((slots, 2), np.float32)
        headings = np.zeros(slots, np.float32)
        futures = np.zeros((slots, horizon, 2), np.float32)
        future_mask = np.zeros((slots, horizon), bool)
        for j, agent in enumerate(agents):
            xy, velocity = np.asarray(agent['position']), np.asarray(agent['velocity'])
            direction = association['projection'][j] if association['mask'][j] else association['los'][j]
            radial = np.dot(velocity - sensor_velocities[-1], direction)
            states[j] = [*xy, radial, *velocity]
            active[j] = True
            category = category_name(agent)
            if category in self.class_names:
                classes[j] = self.class_names.index(category)
            sizes[j], headings[j] = agent['size'], agent['heading']
            if 'future_xy' in agent:
                futures[j] = agent['future_xy']
                future_mask[j] = agent.get('future_valid', [True] * horizon)
        ego_future = np.asarray(ego.get('future_xy', np.zeros((horizon, 2))), np.float32)
        ego_future_mask = np.asarray(ego.get('future_valid', ['future_xy' in ego] * horizon), bool)
        missing_risk = dict(pcol=[0.] * 3, pcol_mask=[False] * 3, dmin=0., dmin_mask=False,
                            areq=0., areq_mask=False)
        if index not in self._risk_cache:
            if risk_schema(r) in LOCAL_RISK_SCHEMAS:
                # A known positive remains supervised even if set coverage is unknown.
                risk = r['local_risk'] if self.supervision else missing_risk
                individual = [r['local_agent_risk_labels'][str(a['id'])] for a in agents] if known_agents else []
            else:
                risk = (r.get('risk_label') or build_risk_labels(r)) if known_agents else missing_risk
                individual = [(r.get('agent_risk_labels', {}).get(str(agent['id'])) or
                               build_risk_labels(dict(r, agents=[agent], risk_agents=[agent])))
                              for agent in agents] if known_agents else []
            self._risk_cache[index] = risk, individual
        risk, individual = self._risk_cache[index]
        agent_risk, agent_risk_mask = np.zeros((slots, 5), np.float32), np.zeros((slots, 5), bool)
        for j, labels in enumerate(individual):
            agent_risk[j] = [*labels['pcol'], labels['dmin'], labels['areq']]
            agent_risk_mask[j] = [*labels['pcol_mask'], labels['dmin_mask'], labels['areq_mask']]
        def risk_order(j):
            positive = np.flatnonzero((agent_risk[j, :3] > .5) & agent_risk_mask[j, :3])
            first = positive[0] if len(positive) else 4
            clearance = agent_risk[j, 3] if agent_risk_mask[j, 3] else float('inf')
            return first, clearance, str(agents[j]['id'])
        ordered = sorted(range(len(agents)), key=risk_order) + list(range(len(agents), slots))
        radial_observed, radial_mask, radial_confidence = np.zeros(slots, np.float32), np.zeros(slots, bool), np.zeros(slots, np.float32)
        agent_los = np.zeros((slots, 2), np.float32)
        doppler_projection = np.zeros_like(agent_los)
        count = len(agents)
        radial_observed[:count], radial_mask[:count] = association['observed'], association['mask']
        radial_confidence[:count], agent_los[:count] = association['confidence'], association['los']
        doppler_projection[:count] = association['projection']
        if not known_agents:
            active[:] = False; future_mask[:] = False; radial_mask[:] = False
        if not self.supervision:
            # Observation-only inference batches contain no GT physical/future tensors,
            # even if the source manifest happens to retain evaluation annotations.
            states[:] = 0.; futures[:] = 0.; ego_future[:] = 0.; ego_future_mask[:] = False
            radial_observed[:] = 0.; radial_confidence[:] = 0.; agent_los[:] = 0.; doppler_projection[:] = 0.
            agent_risk[:] = 0.; agent_risk_mask[:] = False
            classes[:] = -1; sizes[:] = 0.; headings[:] = 0.
        state_mask = np.broadcast_to(active[:, None], states.shape).copy()
        state_mask[:, 2] &= synchronous_radar
        ignored = r.get('ignored_agents', [])
        ignore_boxes = np.asarray([[*a['position'], *a['size'], a['heading']] for a in ignored], np.float32).reshape(-1,5)
        annotation_domain, annotation_domain_known = detection_domain_values(r.get('detection_annotation_domain'))
        values = {
            'radar': np.stack(arrays, axis=1),
            'range_m': np.asarray(radar['range_m'], np.float32),
            'azimuth_rad': np.asarray(radar['azimuth_rad'], np.float32),
            'time_offsets_s': np.asarray(radar['time_offsets_s'], np.float32),
            'ego': np.asarray([np.linalg.norm(ego_velocity), ego['acceleration'][0], ego['yaw_rate']], np.float32),
            'ego_velocity': ego_velocity, 'ego_acceleration': np.asarray(ego['acceleration'], np.float32),
            'radar_pose': poses, 'radar_cartesian': positions, 'radar_los': line_of_sight,
            'radar_sensor_velocity': sensor_velocities, 'radar_doppler_valid': doppler_valid,
            'agent_state': states, 'agent_mask': active,
            'agent_planner_mask': np.asarray([str(agent['id']) in planner_ids for agent in agents] +
                                             [False] * (slots - len(agents)), bool),
            'detection_ignore_boxes': ignore_boxes,
            'detection_ignore_mask': np.ones(len(ignored), bool),
            'detection_ignore_margin_m': np.asarray(r.get('detection_ignore_margin_m',2.),np.float32),
            'taskwise_protocol_id': np.asarray(int(self.supervision_protocol == TASKWISE_PROTOCOL),np.int64),
            'agent_selection_policy_id': np.asarray(int(r.get('agent_selection_policy') == 'moving_vehicle_01_background_v1'),np.int64),
            'mode_target': np.asarray(MODE_INDEX.get(r.get('mode_label'),-1),np.int64),
            'mode_supervision_mask': np.asarray(r.get('mode_mask',False) and self.supervision,bool),
            'local_set_known': np.asarray(known_agents,bool),
            'grounding_available': np.asarray(r.get('grounding_available',True),bool),
            'sft_available': np.asarray(r.get('sft_available',r.get('planner_supervision_available',True)),bool),
            'planner_supervision_available': np.asarray(r.get('planner_supervision_available',True),bool),
            'detection_class_schema_id': np.asarray(int(self.class_schema == 'agent_v1'), np.int64),
            'agent_class': classes, 'agent_class_mask': active & (classes >= 0),
            'agent_size': sizes, 'agent_heading': headings, 'agent_box_mask': active.copy(),
            'agent_radial_observed': radial_observed, 'agent_radial_mask': radial_mask,
            'agent_doppler_confidence': radial_confidence, 'agent_los': agent_los,
            'agent_doppler_projection': doppler_projection,
            'agent_sensor_velocity': np.broadcast_to(sensor_velocities[-1], (slots, 2)).copy(),
            'agent_risk': agent_risk, 'agent_risk_mask': agent_risk_mask,
            'critical_agent_order': np.asarray(ordered, np.int64),
            'agent_state_mask': state_mask,
            'agent_supervision_mask': np.asarray(known_agents),
            'detection_annotation_domain': annotation_domain,
            'detection_annotation_domain_mask': annotation_domain_known,
            'agent_future': futures, 'agent_future_mask': future_mask,
            'ego_future': ego_future, 'ego_future_mask': ego_future_mask,
            'future_times_s': np.asarray(r['future_times_s'], np.float32),
            'risk_supervision_schema_id': np.asarray(int(risk_schema(r) in LOCAL_RISK_SCHEMAS),np.int64),
            'risk_horizon_schema_id': np.asarray(int(horizon_schema(risk_schema(r)) == SHORT_HORIZONS),np.int64),
            'risk_target': np.asarray([*risk['pcol'], risk['dmin'], risk['areq']], np.float32),
            'risk_mask': np.asarray([*risk['pcol_mask'], risk['dmin_mask'], risk['areq_mask']], bool),
        }
        if r.get('road_supervision') is None:
            values.update(road_targets(r.get('map')))
        else:
            explicit = r['road_supervision']
            values.update(road_centerline=np.asarray(explicit['road_centerline'], np.float32),
                          road_centerline_mask=np.asarray(explicit['road_centerline_mask'], bool),
                          road_width=np.asarray(explicit['road_width'], np.float32),
                          road_mask=np.asarray(explicit['road_mask'], bool))
        result = {key: torch.from_numpy(value) for key, value in values.items()}
        result.update(instruction=r['language']['instruction'], sample_id=r['sample_id'], scene_id=r['scene_id'])
        result['agent_ids'] = [str(a['id']) for a in agents] + [''] * (slots-len(agents))
        result['metadata'] = dict(map=r.get('map'), sensors=r['sensors'], ego_pose=ego.get('pose'))
        if r['schema_version'] == 'radar_vla_v3' or self.supervision_protocol == TASKWISE_PROTOCOL:
            result['road_segments'] = r.get('road_segments',[])
        if self.supervision_protocol == TASKWISE_PROTOCOL:
            result['road_task_masks'] = r['task_masks']['road']
        return result


def collate_batch(records):
    """Pad variable agent annotations to this batch's maximum; never truncate."""
    metadata_keys = ('metadata', 'agent_ids', 'road_segments', 'road_task_masks')
    schemas = {int(row.get('detection_class_schema_id', 0)) for row in records}
    if len(schemas) != 1:
        raise ValueError('Cannot collate different detection class schemas')
    if len({int(row.get('risk_supervision_schema_id',0)) for row in records}) != 1:
        raise ValueError('Cannot collate different Risk supervision schemas')
    if len({int(row.get('risk_horizon_schema_id',0)) for row in records}) != 1:
        raise ValueError('Cannot collate different Risk horizons')
    if len({int(row.get('taskwise_protocol_id',0)) for row in records})!=1:
        raise ValueError('Cannot collate different task supervision protocols')
    if len({int(row.get('agent_selection_policy_id',0)) for row in records}) != 1:
        raise ValueError('Cannot collate different Agent selection policies')
    slots = max(len(row['agent_mask']) for row in records)
    ignore_slots = max(len(row.get('detection_ignore_mask', [])) for row in records)
    padded = []
    for row in records:
        sample = {k: v for k, v in row.items() if k not in metadata_keys}
        ignore_count=len(row.get('detection_ignore_mask', []))
        sample['detection_ignore_boxes']=torch.cat((row.get('detection_ignore_boxes',torch.zeros(0,5)),torch.zeros(ignore_slots-ignore_count,5)),dim=0)
        sample['detection_ignore_mask']=torch.cat((row.get('detection_ignore_mask',torch.zeros(0,dtype=torch.bool)),torch.zeros(ignore_slots-ignore_count,dtype=torch.bool)))
        count = len(row['agent_mask'])
        for key, value in sample.items():
            if ((key.startswith('agent_') and key != 'agent_supervision_mask')
                    or key == 'critical_agent_order') and torch.is_tensor(value) and value.ndim:
                if count < slots:
                    tail = value.new_full((slots-count, *value.shape[1:]), -1 if key == 'agent_class' else 0)
                    if key == 'critical_agent_order':
                        tail = torch.arange(count, slots, dtype=value.dtype, device=value.device)
                    sample[key] = torch.cat((value, tail))
        padded.append(sample)
    result = default_collate(padded)
    for key in metadata_keys:
        if key in ('road_segments','road_task_masks') and key in records[0]:
            result[key] = [row[key] for row in records]
        elif key in records[0]:
            result[key] = [row[key] + [''] * (slots-len(row[key])) if key == 'agent_ids' else row[key]
                           for row in records]
    return result


def prepare_labels(manifest: Path, output: Path, **risk_kwargs):
    """Write risk-labeled JSONL, rebasing relative sensor paths for the new file."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    records = load_manifest(manifest)
    if any(risk_schema(r) in LOCAL_RISK_SCHEMAS for r in records):
        raise ValueError('Legacy prepare_labels cannot rewrite Local Risk data; use the versioned migration entry')
    output.parent.mkdir(parents=True, exist_ok=True)
    labelled = []
    for source in records:
        record = copy.deepcopy(source)
        record['risk_label'] = build_risk_labels(record, **risk_kwargs)
        record['agent_risk_labels'] = {str(agent['id']): build_risk_labels(
            dict(record, agents=[agent], risk_agents=[agent]),
            **risk_kwargs) for agent in record.get('agents', [])}
        radar = record['sensors']['radar']
        for key in ('power', 'unfolded_doppler', 'folded_doppler', 'doppler_valid', 'doppler_prior', 'raw_points', 'raw_cube'):
            if key in radar:
                if isinstance(radar[key], list):
                    radar[key] = [os.path.relpath((manifest.parent / path).resolve(), output.parent)
                                  for path in radar[key]]
                else:
                    radar[key] = os.path.relpath((manifest.parent / radar[key]).resolve(), output.parent)
        for camera in record['sensors'].get('camera', {}).values():
            camera['path'] = os.path.relpath((manifest.parent / camera['path']).resolve(), output.parent)
        if 'lidar' in record['sensors']:
            lidar = record['sensors']['lidar']
            lidar['pointcloud'] = os.path.relpath((manifest.parent / lidar['pointcloud']).resolve(), output.parent)
        labelled.append(record)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in labelled), encoding='utf-8')
    temporary.replace(output)
    return output
