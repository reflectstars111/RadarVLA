"""Paired Q1/Q2/Q3/Q5 experiment matrix and controlled Q4 observations."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import torch


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_experiment_plan(config, manifest, output, seeds=(42, 43, 44), epochs=5,
                          batch_size=1, accumulation_steps=4, world_size=8, lr=3e-4):
    if (not seeds or len(set(seeds)) != len(seeds)
            or min(epochs, batch_size, accumulation_steps, world_size) < 1
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError('Invalid experiment seeds/budget')
    variants = [
        ('power_single', 'grounding', {'use_doppler': False, 'use_temporal': False}, {}),
        ('power_multi', 'grounding', {'use_doppler': False, 'use_temporal': True}, {}),
        ('doppler_single', 'grounding', {'use_doppler': True, 'use_temporal': False}, {}),
        ('doppler_multi', 'grounding', {'use_doppler': True, 'use_temporal': True}, {}),
        ('without_ego', 'grounding', {'use_doppler': True, 'use_temporal': True, 'use_ego': False}, {}),
        ('risk_predicted', 'sft', {}, {'risk_source': 'predicted', 'use_risk_token': True, 'reasoning_policy': 'adaptive'}),
        ('risk_none', 'sft', {}, {'risk_source': 'none', 'use_risk_token': False, 'reasoning_policy': 'adaptive'}),
        ('risk_oracle', 'sft', {}, {'risk_source': 'oracle', 'use_risk_token': True, 'reasoning_policy': 'adaptive'}),
        ('always_long', 'sft', {}, {'risk_source': 'predicted', 'use_risk_token': True, 'reasoning_policy': 'always_long'}),
        ('always_short', 'sft', {}, {'risk_source': 'predicted', 'use_risk_token': True, 'reasoning_policy': 'always_short'}),
    ]
    output = Path(output).resolve()
    tasks = []
    for name, stage, model_changes, planner_changes in variants:
        for seed in seeds:
            cfg = copy.deepcopy(config)
            cfg.setdefault('model', {}).update(use_doppler=True, use_temporal=True, use_ego=True)
            cfg['model'].update(model_changes)
            cfg.setdefault('planner', {}).update(planner_changes)
            task_id = f'{name}_seed{seed}'
            dependencies = [f'doppler_multi_seed{seed}'] if stage == 'sft' else []
            tasks.append(dict(id=task_id, variant=name, stage=stage, seed=seed, config=cfg,
                              output=str(output / task_id), dependencies=dependencies,
                              epochs=epochs, batch_size=batch_size, accumulation_steps=accumulation_steps,
                              lr=lr, world_size=world_size))
    plan = dict(schema='radar_vla_experiments_v2', manifest=str(Path(manifest).resolve()),
                output=str(output), tasks=tasks,
                comparisons={'Q1_Q2': ['power_single', 'power_multi', 'doppler_single', 'doppler_multi', 'without_ego'],
                             'Q3': ['risk_none', 'risk_predicted', 'risk_oracle'],
                             'Q5': ['always_long', 'always_short', 'risk_predicted']},
                selection='validation only; tests evaluated explicitly after all runs',
                oracle_cohort='Q3 requires the SAME manifest/cohort with all five risk labels valid for every arm; no implicit filtering')
    from .pipeline import data_fingerprint
    plan['data_fingerprint'] = data_fingerprint(manifest)
    plan['sha256'] = _digest(plan)
    return plan


def run_experiment_plan(plan_path, device='cuda', resume=False):
    """Sequential, explicit execution; no daemon or experiment starts on import."""
    from .pipeline import write_json, data_fingerprint
    plan = json.loads(Path(plan_path).read_text())
    if plan.get('sha256') != _digest({k: v for k, v in plan.items() if k != 'sha256'}):
        raise ValueError('Experiment plan hash differs')
    if plan.get('schema') != 'radar_vla_experiments_v2':
        raise ValueError('Unsupported experiment plan')
    if plan.get('data_fingerprint') != data_fingerprint(plan['manifest']):
        raise ValueError('Dataset changed since experiment plan was frozen')
    # Q3 arms must share the exact cohort. Validate BEFORE starting any task,
    # including Stage 1; never silently filter the oracle arm after the fact.
    from .data import RadarDataset, collate_batch, load_manifest
    from .pipeline import select_risk
    oracle_tasks = [t for t in plan['tasks'] if t['variant'] == 'risk_oracle']
    if oracle_tasks:
        max_agents = oracle_tasks[0]['config'].get('model', {}).get('max_agents', 8)
        splits = sorted({r['split'] for r in load_manifest(Path(plan['manifest']))})
        for split in splits:
            dataset = RadarDataset(Path(plan['manifest']), split, max_agents=max_agents)
            for index in range(len(dataset)):
                sample = collate_batch([dataset[index]])
                try:
                    select_risk(sample['risk_target'], sample, 'oracle')
                except ValueError as error:
                    raise ValueError(f'Q3 common cohort: {dataset.records[index]["sample_id"]}: {error}') from error
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    by_id = {task['id']: task for task in plan['tasks']}
    if len(by_id) != len(plan['tasks']):
        raise ValueError('Duplicate experiment IDs')
    for task in plan['tasks']:
        if plan['data_fingerprint'] != data_fingerprint(plan['manifest']):
            raise ValueError('Dataset changed between paired runs')
        directory = Path(task['output'])
        if not directory.is_relative_to(root):
            raise ValueError('Experiment output is outside plan directory')
        args = ['-m', 'radar_vla', 'train', '--manifest', plan['manifest'],
                '--output', str(directory), '--stage', task['stage'], '--seed', str(task['seed']),
                '--epochs', str(task['epochs']), '--batch-size', str(task['batch_size']),
                '--accumulation-steps', str(task['accumulation_steps']), '--lr', str(task['lr']),
                '--precision', 'bfloat16' if device.startswith('cuda') else 'float32', '--device', device]
        for dependency in task['dependencies']:
            parent = Path(by_id[dependency]['output'])
            status = json.loads((parent / 'status.json').read_text())
            if status['state'] != 'complete':
                raise ValueError(f'Incomplete dependency {dependency}')
            if not resume or not (directory / 'last.pt').exists():
                args += ['--init-grounding', str(parent / 'best.pt')]
        if resume and (directory / 'last.pt').exists():
            args += ['--resume']
        cfg_path = root / 'configs' / (task['id'] + '.json')
        write_json(cfg_path, task['config'])
        args += ['--config', str(cfg_path)]
        command = [sys.executable]
        if task['world_size'] > 1:
            command += ['-m', 'torch.distributed.run', '--standalone',
                        f'--nproc-per-node={task["world_size"]}']
        command += args
        status_path = root / 'queue_status.json'
        write_json(status_path, dict(state='running', task=task['id'], command=command))
        with (root / f'{task["id"]}.log').open('a') as stream:
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parent.parent,
                                       stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(status_path, dict(state='failed', task=task['id'], exit_code=completed.returncode))
            raise RuntimeError(f'Experiment {task["id"]} failed; see its log')
    write_json(root / 'queue_status.json', dict(state='complete', tasks=len(plan['tasks']), test_evaluated=False))
    return dict(state='complete', tasks=len(plan['tasks']))


def intervene_doppler(batch, velocity_mps, range_interval, azimuth_interval, history='current'):
    """Counterfactual sensor intervention; never modifies Power, GT or language.

    This tests model sensitivity, not a physically re-simulated driving scene.
    Intervals are in each radar's measured range/azimuth coordinate system.
    """
    if history not in ('current', 'all') or not math.isfinite(velocity_mps):
        raise ValueError('Invalid history/velocity intervention')
    low_r, high_r = range_interval
    low_a, high_a = azimuth_interval
    if not all(math.isfinite(x) for x in (low_r, high_r, low_a, high_a)) or low_r >= high_r or low_a >= high_a:
        raise ValueError('Intervention intervals must be finite and increasing')
    result = dict(batch)
    result['radar'] = batch['radar'].clone()
    region = ((batch['range_m'] >= low_r) & (batch['range_m'] <= high_r))[:, :, None] & (
        (batch['azimuth_rad'] >= low_a) & (batch['azimuth_rad'] <= high_a))[:, None, :]
    if not region.flatten(1).any(1).all():
        raise ValueError('Intervention region is empty for a sample')
    frames = range(result['radar'].shape[1]) if history == 'all' else [-1]
    for frame in frames:
        result['radar'][:, frame, 1] = torch.where(region, velocity_mps, result['radar'][:, frame, 1])
    return result


@torch.no_grad()
def doppler_sweep(checkpoint, manifest, output, sample_id, range_interval, azimuth_interval,
                  velocities=(-2., -5., -8.), device='cpu', split='test', history='current'):
    from .data import RadarDataset, collate_batch
    from .pipeline import load_models, to_device, autocast_context, write_json
    grounder, planner, state = load_models(checkpoint, device)
    dataset = RadarDataset(manifest, split, max_agents=state['protocol']['config']['model']['max_agents'], supervision=False)
    indices = [i for i, r in enumerate(dataset.records) if r['sample_id'] == sample_id]
    if len(indices) != 1:
        raise ValueError('sample_id must identify exactly one observation')
    batch = to_device(collate_batch([dataset[indices[0]]]), device)
    results = []
    for velocity in velocities:
        changed = intervene_doppler(batch, float(velocity), range_interval, azimuth_interval, history)
        with autocast_context(device, state['protocol'].get('precision', 'float32')):
            pred = grounder(changed)
            plan = None
            if planner is not None:
                ids = planner.generate(pred['radar_tokens'], pred['risk'], batch['instruction'], max_new_tokens=1024)
                plan = planner.decode(ids[0], future_times_s=batch['future_times_s'][0].cpu().tolist())
        results.append(dict(velocity_mps=velocity, risk=pred['risk'][0].cpu().tolist(),
                            agent_future=pred['agent_future'][0].cpu().tolist(), plan=plan))
    report = dict(schema='radar_doppler_intervention_v2', sample_id=sample_id,
                  range_interval=list(range_interval), azimuth_interval=list(azimuth_interval),
                  history=history, results=results, instruction=batch['instruction'][0],
                  interpretation='controlled observation sensitivity; unchanged Power/positions/GT; not a new simulator rollout')
    write_json(output, report)
    return report
