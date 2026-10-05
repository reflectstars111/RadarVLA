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

from .geometry import build_risk_labels
from .coordinates import validate_transform
from .radar_processing import radar_geometry, associate_doppler
from .records import road_targets

SCHEMA_VERSION = 'radar_vla_v2'
SUPPORTED_SCHEMAS = ('radar_vla_v1', SCHEMA_VERSION)


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
        if not isinstance(radar.get(key), str) or not radar[key]:
            raise ValueError(f'{name} radar requires {key}; folded_doppler is not unfolded velocity')
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
    identifiers = set()
    for agent in agents:
        if not isinstance(agent, dict) or not isinstance(agent.get('id'), (str, int)):
            raise ValueError(f'{name} agents require id')
        if str(agent['id']) in identifiers:
            raise ValueError(f'{name} duplicate agent id')
        identifiers.add(str(agent['id']))
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
    if record['schema_version'] == SCHEMA_VERSION:
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
    cache = record.get('agent_risk_labels', {})
    if not isinstance(cache, dict) or any(key not in identifiers for key in cache):
        raise ValueError('agent_risk_labels must map current agent ids to risk targets')
    all_risks = list(cache.values())
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
    def __init__(self, manifest: Path, split: str, max_agents=8, *, supervision=True):
        self.manifest = Path(manifest).resolve()
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if max_agents < 1:
            raise ValueError('max_agents must be positive')
        self.max_agents = max_agents
        self.supervision = supervision
        self.records = [r for r in load_manifest(self.manifest, supervision=supervision) if r['split'] == split]
        if not self.records:
            raise ValueError(f'no records for split {split}')
        if any(len(r.get('agents', [])) > max_agents for r in self.records):
            raise ValueError('record exceeds max_agents; increase capacity (silent truncation is forbidden)')
        self._risk_cache = {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        radar = r['sensors']['radar']
        shape = tuple(len(radar[k]) for k in ('time_offsets_s', 'range_m', 'azimuth_rad'))
        arrays = []
        for key in ('power', 'unfolded_doppler'):
            array = np.load(self.manifest.parent / radar[key], allow_pickle=False)
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
        agents = sorted(r.get('agents', []), key=lambda a: (np.linalg.norm(a['position']), str(a['id'])))
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
        states = np.zeros((self.max_agents, 5), np.float32)
        active = np.zeros(self.max_agents, bool)
        futures = np.zeros((self.max_agents, horizon, 2), np.float32)
        future_mask = np.zeros((self.max_agents, horizon), bool)
        for j, agent in enumerate(agents):
            xy, velocity = np.asarray(agent['position']), np.asarray(agent['velocity'])
            direction = association['projection'][j] if association['mask'][j] else association['los'][j]
            radial = np.dot(velocity - sensor_velocities[-1], direction)
            states[j] = [*xy, radial, *velocity]
            active[j] = True
            if 'future_xy' in agent:
                futures[j] = agent['future_xy']
                future_mask[j] = agent.get('future_valid', [True] * horizon)
        ego_future = np.asarray(ego.get('future_xy', np.zeros((horizon, 2))), np.float32)
        ego_future_mask = np.asarray(ego.get('future_valid', ['future_xy' in ego] * horizon), bool)
        missing_risk = dict(pcol=[0.] * 3, pcol_mask=[False] * 3, dmin=0., dmin_mask=False,
                            areq=0., areq_mask=False)
        if index not in self._risk_cache:
            risk = (r.get('risk_label') or build_risk_labels(r)) if known_agents else missing_risk
            individual = [(r.get('agent_risk_labels', {}).get(str(agent['id'])) or
                           build_risk_labels(dict(r, agents=[agent], risk_agents=[agent], tracking_coverage_valid=[True] * horizon)))
                          for agent in agents] if known_agents else []
            self._risk_cache[index] = risk, individual
        risk, individual = self._risk_cache[index]
        agent_risk, agent_risk_mask = np.zeros((self.max_agents, 5), np.float32), np.zeros((self.max_agents, 5), bool)
        for j, labels in enumerate(individual):
            agent_risk[j] = [*labels['pcol'], labels['dmin'], labels['areq']]
            agent_risk_mask[j] = [*labels['pcol_mask'], labels['dmin_mask'], labels['areq_mask']]
        def risk_order(j):
            positive = np.flatnonzero((agent_risk[j, :3] > .5) & agent_risk_mask[j, :3])
            first = positive[0] if len(positive) else 4
            clearance = agent_risk[j, 3] if agent_risk_mask[j, 3] else float('inf')
            return first, clearance, str(agents[j]['id'])
        ordered = sorted(range(len(agents)), key=risk_order) + list(range(len(agents), self.max_agents))
        radial_observed, radial_mask, radial_confidence = np.zeros(self.max_agents, np.float32), np.zeros(self.max_agents, bool), np.zeros(self.max_agents, np.float32)
        agent_los = np.zeros((self.max_agents, 2), np.float32)
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
        state_mask = np.broadcast_to(active[:, None], states.shape).copy()
        state_mask[:, 2] &= synchronous_radar
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
            'agent_radial_observed': radial_observed, 'agent_radial_mask': radial_mask,
            'agent_doppler_confidence': radial_confidence, 'agent_los': agent_los,
            'agent_doppler_projection': doppler_projection,
            'agent_sensor_velocity': np.broadcast_to(sensor_velocities[-1], (self.max_agents, 2)).copy(),
            'agent_risk': agent_risk, 'agent_risk_mask': agent_risk_mask,
            'critical_agent_order': np.asarray(ordered, np.int64),
            'agent_state_mask': state_mask,
            'agent_supervision_mask': np.asarray(known_agents),
            'agent_future': futures, 'agent_future_mask': future_mask,
            'ego_future': ego_future, 'ego_future_mask': ego_future_mask,
            'future_times_s': np.asarray(r['future_times_s'], np.float32),
            'risk_target': np.asarray([*risk['pcol'], risk['dmin'], risk['areq']], np.float32),
            'risk_mask': np.asarray([*risk['pcol_mask'], risk['dmin_mask'], risk['areq_mask']], bool),
        }
        values.update(road_targets(r.get('map')))
        result = {key: torch.from_numpy(value) for key, value in values.items()}
        result.update(instruction=r['language']['instruction'], sample_id=r['sample_id'], scene_id=r['scene_id'])
        result['agent_ids'] = [str(a['id']) for a in agents] + [''] * (self.max_agents-len(agents))
        result['metadata'] = dict(map=r.get('map'), sensors=r['sensors'], ego_pose=ego.get('pose'))
        return result


def collate_batch(records):
    """Stack fixed-shape tensors and retain strings as lists."""
    metadata_keys = ('metadata', 'agent_ids')
    result = default_collate([{k: v for k, v in row.items() if k not in metadata_keys} for row in records])
    for key in metadata_keys:
        if key in records[0]:
            result[key] = [row[key] for row in records]
    return result


def prepare_labels(manifest: Path, output: Path, **risk_kwargs):
    """Write risk-labeled JSONL, rebasing relative sensor paths for the new file."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    records = load_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    labelled = []
    for source in records:
        record = copy.deepcopy(source)
        record['risk_label'] = build_risk_labels(record, **risk_kwargs)
        record['agent_risk_labels'] = {str(agent['id']): build_risk_labels(
            dict(record, agents=[agent], risk_agents=[agent], tracking_coverage_valid=[True] * len(record['future_times_s'])),
            **risk_kwargs) for agent in record.get('agents', [])}
        radar = record['sensors']['radar']
        for key in ('power', 'unfolded_doppler', 'folded_doppler', 'doppler_valid', 'doppler_prior', 'raw_points', 'raw_cube'):
            if key in radar:
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
