"""Persistent response episodes annotated from GT; never actor observations.

Appearance and hazard onset are separate episode boundaries. A target can
reappear after an observed absence, or become dangerous more than once. Unknown
labels neither prove safety nor reset an active hazard episode.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from .geometry import box_distance
from .response_annotations import reward_record
from .response_reward import ResponseRewardConfig, event_baseline


EVENT_KINDS = frozenset(('appearance', 'reappearance', 'hazard_transition', 'hazard_start'))
_CORE_FIELDS = {'event_id', 'target_agent_id', 'sample_ids'}
_V2_FIELDS = {'schema_version', 'event_kind', 'trigger_sample_id', 'trigger_timestamp_s',
              'scene_id', 'split', 'episode_index', 'hazard', 'annotation_horizon_s',
              'annotation_coverage_complete', 'requested_safety_horizon_s',
              'response_window_s', 'response_window_complete', 'end_timestamp_s'}


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _known(record):
    return bool(record.get('agent_supervision_available',
                           record.get('agent_supervision', 'agents' in record)))


def _present(record, target):
    return any(a['id'] == target and a.get('present_valid', True)
               for a in record.get('agents', []))


def _index_records(records):
    by_id, scenes = {}, defaultdict(list)
    for row in records:
        sample = row['sample_id']
        if not isinstance(sample, str) or not sample or sample in by_id:
            raise ValueError('Event frames require unique nonempty sample_id strings')
        if not isinstance(row['timestamp_s'], (int, float)) or not math.isfinite(row['timestamp_s']):
            raise ValueError('Event frames require finite timestamp_s')
        by_id[sample] = row
        scenes[row['scene_id']].append(row)
    for rows in scenes.values():
        if len({r['split'] for r in rows}) != 1:
            raise ValueError('Event scene cannot cross a split')
        rows.sort(key=lambda r: r['timestamp_s'])
        if any(b['timestamp_s'] <= a['timestamp_s'] for a, b in zip(rows, rows[1:])):
            raise ValueError('Event scene timestamps must be unique')
    return by_id, scenes


def load_events(path, records):
    """Load v1 appearance windows or explicit v2 persistent episode windows.

    V1 starts at the observed appearance after its absence frame. V2 may begin
    with an already visible hazard. Earlier track occurrences are allowed;
    overlapping active windows for the same scene/track are not. A single
    historical frame before a trigger may overlap an earlier event's window.
    """
    by_id, scenes = _index_records(records)
    positions = {sample['sample_id']: i for rows in scenes.values() for i, sample in enumerate(rows)}
    result, identifiers = [], set()
    for event in _jsonl(path):
        if not isinstance(event, dict) or not _CORE_FIELDS <= event.keys() or event.keys() - (_CORE_FIELDS | _V2_FIELDS):
            raise ValueError('Events require event_id, target_agent_id, sample_ids and supported v2 fields')
        eid, target, ids = (event[k] for k in ('event_id', 'target_agent_id', 'sample_ids'))
        if not isinstance(eid, str) or not eid or eid in identifiers:
            raise ValueError('Event IDs must be unique nonempty strings')
        if not isinstance(target, (str, int)) or isinstance(target, bool) or target == '':
            raise ValueError('target_agent_id must be a stable string or integer ID')
        if (not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or any(i not in by_id for i in ids)):
            raise ValueError('Events require distinct known sample_ids')
        rows = [by_id[i] for i in ids]
        scene, split = rows[0]['scene_id'], rows[0]['split']
        if len({(r['scene_id'], r['split']) for r in rows}) != 1:
            raise ValueError('Event cannot cross a scene or split')
        if event.get('scene_id', scene) != scene or event.get('split', split) != split:
            raise ValueError('Event scene_id or split does not match its frames')
        start = positions[ids[0]]
        if [r['sample_id'] for r in scenes[scene][start:start + len(ids)]] != ids:
            raise ValueError('Event sample_ids must be consecutive and chronological')
        v1 = not ('event_kind' in event or 'trigger_sample_id' in event)
        if v1:
            if len(ids) < 2:
                raise ValueError('V1 appearance events require >=2 sample_ids')
            trigger = ids[1]
            appeared = any(_present(r, target) for r in scenes[scene][:start])
            kind = 'reappearance' if appeared else 'appearance'
        else:
            trigger, kind = event.get('trigger_sample_id'), event.get('event_kind')
            if kind not in EVENT_KINDS:
                raise ValueError('Unsupported event_kind')
            if trigger not in ids:
                raise ValueError('trigger_sample_id must be an event frame')
        index = positions[trigger]
        record = by_id[trigger]
        if not _known(record) or sum(a['id'] == target and a.get('present_valid', True)
                                      for a in record.get('agents', [])) != 1:
            raise ValueError('Event trigger requires one present target_agent_id')
        if kind in ('appearance', 'reappearance'):
            if index == 0 or not _known(scenes[scene][index - 1]) or _present(scenes[scene][index - 1], target):
                raise ValueError('Appearance requires observed absence immediately before trigger')
            if positions[ids[0]] >= index:
                raise ValueError('Appearance sample_ids must include its preceding absence')
        timestamp = record['timestamp_s']
        if ('trigger_timestamp_s' in event and (not isinstance(event['trigger_timestamp_s'], (int, float))
                or not math.isfinite(event['trigger_timestamp_s'])
                or abs(event['trigger_timestamp_s'] - timestamp) > 1e-7)):
            raise ValueError('Event trigger timestamp does not match trigger frame')
        if 'end_timestamp_s' in event and event['end_timestamp_s'] != rows[-1]['timestamp_s']:
            raise ValueError('Event end timestamp does not match final frame')
        if 'hazard' in event and event['hazard'] is not None and type(event['hazard']) is not bool:
            raise ValueError('hazard annotation must be boolean or null')
        if 'episode_index' in event and (type(event['episode_index']) is not int or event['episode_index'] < 0):
            raise ValueError('episode_index must be a nonnegative integer')
        for field in ('annotation_coverage_complete', 'response_window_complete'):
            if field in event and type(event[field]) is not bool:
                raise ValueError(f'{field} must be boolean')
        for field in ('annotation_horizon_s', 'requested_safety_horizon_s', 'response_window_s'):
            if field in event and (not isinstance(event[field], (float, int))
                                   or not math.isfinite(event[field]) or event[field] < 0):
                raise ValueError(f'{field} must be finite and nonnegative')
        identifiers.add(eid)
        result.append({**event, 'scene_id': scene, 'split': split, 'event_kind': kind,
                       'trigger_sample_id': trigger, 'trigger_timestamp_s': float(timestamp),
                       'end_timestamp_s': float(rows[-1]['timestamp_s'])})
    if not result:
        raise ValueError('No response events found; supply consecutive annotated scenes')
    tracks = defaultdict(list)
    for event in result:
        tracks[(event['scene_id'], event['target_agent_id'])].append(event)
    for track_events in tracks.values():
        track_events.sort(key=lambda e: e['trigger_timestamp_s'])
        episode_ids = set()
        for i, event in enumerate(track_events):
            if i and event['trigger_timestamp_s'] <= track_events[i - 1]['end_timestamp_s']:
                raise ValueError('Duplicate overlapping same-track episodes would reset the response clock')
            event.setdefault('episode_index', i)
            if event['episode_index'] in episode_ids:
                raise ValueError('Duplicate same-track episode_index')
            episode_ids.add(event['episode_index'])
    return result


def _annotation(record, target, cfg):
    """Confirm positives with available GT; only full coverage confirms safety."""
    result = dict(hazard=None, annotation_horizon_s=0., annotation_coverage_complete=False,
                  requested_safety_horizon_s=float(cfg.safety_horizon_s))
    agents = record.get('risk_agents', record.get('agents', []))
    matches = [a for a in agents if a['id'] == target and a.get('present_valid', True)]
    if (not _known(record) or len(matches) != 1 or 'future_times_s' not in record
            or not {'velocity', 'box_size'} <= record.get('ego', {}).keys()
            or not {'position', 'velocity', 'heading', 'size'} <= matches[0].keys()):
        return result
    times = np.asarray(record['future_times_s'], dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or times[0] <= 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Event annotation requires increasing finite future_times_s')
    agent = matches[0]
    valid = np.asarray(agent.get('future_valid', ['future_xy' in agent] * len(times)), dtype=bool)
    if valid.shape != times.shape:
        raise ValueError('Target future_valid must match future_times_s')
    # Restrict to the contiguous known prefix. Holes cannot be bridged by future
    # interpolation to manufacture a safe baseline or a new hazard episode.
    prefix = int(np.flatnonzero(~valid)[0]) if (~valid).any() else len(valid)
    horizon = min(float(times[prefix - 1]), cfg.safety_horizon_s) if prefix else 0.
    result['annotation_horizon_s'] = horizon
    current_distance = box_distance([0., 0.], 0., record['ego']['box_size'],
                                    agent['position'], agent['heading'], agent['size'])
    if horizon <= cfg.sustain_s:
        if current_distance < cfg.hazard_distance_m:
            result['hazard'] = True
        return result
    limited = replace(cfg, safety_horizon_s=horizon)
    baseline = event_baseline(record, target, limited)
    complete = bool(baseline['coverage_complete'] and horizon >= cfg.safety_horizon_s - 1e-7)
    result['annotation_coverage_complete'] = complete
    hazard = baseline['hazard'] or current_distance < cfg.hazard_distance_m
    result['hazard'] = bool(hazard) if hazard or complete else None
    return result


def prepare_response_events(manifest, output, window_s=1., max_gap_s=.5, reward_config=None):
    """Build appearance, reappearance and hazard-onset episodes from labels.

    The requested replay window is at least the response deadline. A later
    episode, observed absence, recording gap or scene end can truncate it; the
    response_window_complete field makes that censoring explicit. GT selection
    uses a recorded target horizon capped at the reward horizon and records when
    that evidence cannot establish full safety coverage.
    """
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in (window_s, max_gap_s)):
        raise ValueError('Event window and maximum frame gap must be positive')
    cfg = (ResponseRewardConfig(**reward_config) if isinstance(reward_config, dict)
           else reward_config or ResponseRewardConfig())
    window_s = max(float(window_s), float(cfg.response_deadline_s))
    output = Path(output)
    if output.exists():
        raise FileExistsError('Event output already exists')
    records = _jsonl(manifest)
    _, scenes = _index_records(records)
    events = []
    for scene, rows in sorted(scenes.items()):
        seen, hazard_state, starts, states = set(), {}, defaultdict(list), {}
        for i, current in enumerate(rows):
            previous = rows[i - 1] if i else None
            continuous = bool(previous is not None and current['timestamp_s'] - previous['timestamp_s'] <= max_gap_s)
            if previous is not None and not continuous:
                hazard_state.clear()
            if not _known(current):
                continue
            present = {a['id'] for a in current.get('agents', []) if a.get('present_valid', True)}
            if _known(current):
                for target in hazard_state.keys() - present:
                    hazard_state[target] = None
            annotated = (reward_record(current, rows, max_horizon_s=cfg.safety_horizon_s)
                         if 'future_times_s' in current and 'ego' in current else current)
            for target in sorted(present, key=lambda x: (str(type(x)), str(x))):
                annotation = _annotation(annotated, target, cfg)
                status, old = annotation['hazard'], hazard_state.get(target)
                states[(target, i)] = status
                appeared = continuous and _known(previous) and not _present(previous, target)
                kind = None
                if appeared:
                    kind = 'reappearance' if target in seen else 'appearance'
                elif status is True and old is not True:
                    kind = 'hazard_transition' if old is False else 'hazard_start'
                if kind is not None:
                    starts[target].append(dict(index=i, event_kind=kind, **annotation))
                if status is not None:
                    hazard_state[target] = status
            seen |= present
        for target, episodes in starts.items():
            for number, onset in enumerate(episodes):
                i = onset.pop('index')
                stop = episodes[number + 1]['index'] if number + 1 < len(episodes) else len(rows)
                first = i - 1 if i and rows[i]['timestamp_s'] - rows[i - 1]['timestamp_s'] <= max_gap_s else i
                last = i
                for j in range(i + 1, stop):
                    if (rows[j]['timestamp_s'] - rows[j - 1]['timestamp_s'] > max_gap_s
                            or (_known(rows[j]) and not _present(rows[j], target))
                            or (onset['hazard'] is True and states.get((target, j)) is False)):
                        break
                    last = j
                    if rows[j]['timestamp_s'] - rows[i]['timestamp_s'] >= window_s - 1e-7:
                        break
                events.append(dict(schema_version='radar_vla_response_events_v2',
                    event_id=f'response_{len(events):06d}', target_agent_id=target,
                    scene_id=scene, split=rows[i]['split'], episode_index=number,
                    sample_ids=[r['sample_id'] for r in rows[first:last + 1]],
                    trigger_sample_id=rows[i]['sample_id'], trigger_timestamp_s=rows[i]['timestamp_s'],
                    end_timestamp_s=rows[last]['timestamp_s'], response_window_s=window_s,
                    response_window_complete=rows[last]['timestamp_s'] - rows[i]['timestamp_s'] >= window_s - 1e-7,
                    **onset))
    if not events:
        raise ValueError('No response events found; no observed appearances or confirmed hazard starts')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    try:
        temporary.write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in events))
        load_events(temporary, records)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(events=len(events), output=str(output), response_window_s=window_s,
                incomplete_windows=sum(not e['response_window_complete'] for e in events))
