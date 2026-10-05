"""Explicit common-cohort construction; never filter silently inside training."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from .data import RadarDataset, collate_batch, load_manifest
from .planner import PlannerConfig, create_supervision
from .tokenizer import PhysicalTokenizer

SENSOR_PATH_KEYS = {'path', 'power', 'unfolded_doppler', 'folded_doppler', 'doppler_valid',
                    'doppler_prior', 'raw_points', 'raw_cube', 'pointcloud', 'points', 'cube'}


def absolute_sensor_paths(value, base):
    if isinstance(value, dict):
        return {k: str((base / v).resolve()) if k in SENSOR_PATH_KEYS and isinstance(v, str)
                else absolute_sensor_paths(v, base) for k, v in value.items()}
    if isinstance(value, list):
        return [absolute_sensor_paths(v, base) for v in value]
    return value


def build_cohort(manifest, output, config=None, require_oracle=False):
    """Select a shared fully supervised cohort for all short/long/adaptive arms.

    Save every exclusion reason and sample ID. This selects annotation availability,
    never a model's prediction quality. Original data remain unchanged. Oracle adds
    the strict five-known-KRS constraint, including uncensored required braking.
    """
    from .pipeline import resolve_config, data_fingerprint, write_json, select_risk
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    report_path = output.with_suffix(output.suffix + '.cohort.json')
    if output.exists() or report_path.exists():
        raise ValueError('Cohort output/report already exists')
    config = resolve_config(config)
    rows = load_manifest(manifest)
    retained, excluded = [], []
    for split in sorted({row['split'] for row in rows}):
        dataset = RadarDataset(manifest, split, max_agents=config['model']['max_agents'])
        for index, record in enumerate(dataset.records):
            reason = None
            try:
                sample = collate_batch([dataset[index]])
                if not bool(sample['agent_supervision_mask'].all()):
                    raise ValueError('missing agent annotation coverage')
                if require_oracle:
                    select_risk(sample['risk_target'], sample, 'oracle')
                for policy in ('adaptive', 'always_long', 'always_short'):
                    options = {**config['planner'], 'reasoning_policy': policy}
                    cfg = PlannerConfig(**options)
                    tok = PhysicalTokenizer(cfg.bins, cfg.coordinate_limit_m, cfg.velocity_limit_mps)
                    teacher = create_supervision(sample, tok, cfg)['rows'][0]
                    if teacher['mode'] is None or teacher['ego'] is None or not teacher['ego']['fitted']:
                        raise ValueError(f'{policy}: missing mode or observed ego trajectory endpoints')
                    if teacher['road'] is not None and not (teacher['road']['fitted'] and teacher['road']['width_mask']):
                        raise ValueError(f'{policy}: missing map corridor supervision')
                    if teacher['mode'] == 'long' and any(not a['trajectory']['fitted'] for a in teacher['agents']):
                        raise ValueError(f'{policy}: missing agent future endpoints')
            except ValueError as error:
                reason = str(error)
            if reason is not None:
                excluded.append(dict(sample_id=record['sample_id'], scene_id=record['scene_id'], split=split, reason=reason))
            else:
                result = copy.deepcopy(record)
                result['sensors'] = absolute_sensor_paths(result['sensors'], manifest.parent)
                retained.append(result)
    report = dict(schema='radar_vla_common_cohort_v2', source=str(manifest), source_fingerprint=data_fingerprint(manifest),
                  require_oracle=require_oracle, config=config, original_samples=len(rows), retained_samples=len(retained),
                  retained_ids=[r['sample_id'] for r in retained], excluded=excluded,
                  selection='annotation availability for all three reasoning policies; use this same manifest for every arm')
    if {r['split'] for r in rows} != {r['split'] for r in retained}:
        raise ValueError(f'Cohort selection empties a split; retained={len(retained)}, first exclusions={excluded[:3]}')
    write_json(report_path, report)
    output.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in retained), encoding='utf-8')
    return report
