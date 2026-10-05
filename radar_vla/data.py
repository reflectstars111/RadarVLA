"""Strict JSONL manifest ingestion for the isolated RadarVLA prototype."""
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

SCHEMA_VERSION = 'radar_vla_v1'


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


def _validate_record(record, index):
    name = f'record {index}'
    if not isinstance(record, dict):
        raise ValueError(f'{name} must be an object')
    _finite(record, name)
    if record.get('schema_version') != SCHEMA_VERSION:
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
        if key == 'time_offsets_s' and (np.any(values > 1e-8) or abs(values[-1]) > 1e-8):
            raise ValueError(f'{name} time_offsets_s must end at zero, without future frames')
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
    agents = record.get('agents')
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
    if not isinstance(record.get('language', {}).get('instruction'), str):
        raise ValueError(f'{name} language.instruction must be a string')
    if 'risk_label' in record:
        risk = record['risk_label']
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


def load_manifest(manifest: Path):
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
            _validate_record(record, lineno)
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
    def __init__(self, manifest: Path, split: str, max_agents=8):
        self.manifest = Path(manifest).resolve()
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val or test')
        if max_agents < 1:
            raise ValueError('max_agents must be positive')
        self.max_agents = max_agents
        self.records = [r for r in load_manifest(self.manifest) if r['split'] == split]
        if not self.records:
            raise ValueError(f'no records for split {split}')
        if any(len(r['agents']) > max_agents for r in self.records):
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
        agents = sorted(r['agents'], key=lambda a: (np.linalg.norm(a['position']), str(a['id'])))
        horizon = len(r['future_times_s'])
        states = np.zeros((self.max_agents, 5), np.float32)
        active = np.zeros(self.max_agents, bool)
        futures = np.zeros((self.max_agents, horizon, 2), np.float32)
        future_mask = np.zeros((self.max_agents, horizon), bool)
        for j, agent in enumerate(agents):
            xy, velocity = np.asarray(agent['position']), np.asarray(agent['velocity'])
            direction = xy / max(np.linalg.norm(xy), 1e-8)
            radial = np.dot(velocity - ego_velocity, direction)
            states[j] = [*xy, radial, *velocity]
            active[j] = True
            if 'future_xy' in agent:
                futures[j] = agent['future_xy']
                future_mask[j] = agent.get('future_valid', [True] * horizon)
        ego_future = np.asarray(ego.get('future_xy', np.zeros((horizon, 2))), np.float32)
        ego_future_mask = np.asarray(ego.get('future_valid', ['future_xy' in ego] * horizon), bool)
        if index not in self._risk_cache:
            self._risk_cache[index] = r.get('risk_label') or build_risk_labels(r)
        risk = self._risk_cache[index]
        values = {
            'radar': np.stack(arrays, axis=1),
            'range_m': np.asarray(radar['range_m'], np.float32),
            'azimuth_rad': np.asarray(radar['azimuth_rad'], np.float32),
            'time_offsets_s': np.asarray(radar['time_offsets_s'], np.float32),
            'ego': np.asarray([np.linalg.norm(ego_velocity), ego['acceleration'][0], ego['yaw_rate']], np.float32),
            'ego_velocity': ego_velocity, 'agent_state': states, 'agent_mask': active,
            'agent_state_mask': np.broadcast_to(active[:, None], states.shape).copy(),
            'agent_supervision_mask': np.asarray(True),
            'agent_future': futures, 'agent_future_mask': future_mask,
            'ego_future': ego_future, 'ego_future_mask': ego_future_mask,
            'future_times_s': np.asarray(r['future_times_s'], np.float32),
            'risk_target': np.asarray([*risk['pcol'], risk['dmin'], risk['areq']], np.float32),
            'risk_mask': np.asarray([*risk['pcol_mask'], risk['dmin_mask'], risk['areq_mask']], bool),
        }
        result = {key: torch.from_numpy(value) for key, value in values.items()}
        result.update(instruction=r['language']['instruction'], sample_id=r['sample_id'], scene_id=r['scene_id'])
        return result


def collate_batch(records):
    """Stack fixed-shape tensors and retain strings as lists."""
    return default_collate(records)


def prepare_labels(manifest: Path, output: Path, **risk_kwargs):
    """Write risk-labeled JSONL, rebasing relative sensor paths for the new file."""
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    records = load_manifest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    labelled = []
    for source in records:
        record = copy.deepcopy(source)
        record['risk_label'] = build_risk_labels(record, **risk_kwargs)
        radar = record['sensors']['radar']
        for key in ('power', 'unfolded_doppler', 'folded_doppler'):
            if key in radar:
                radar[key] = os.path.relpath((manifest.parent / radar[key]).resolve(), output.parent)
        labelled.append(record)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in labelled), encoding='utf-8')
    temporary.replace(output)
    return output
