"""Paired full Oracle-Risk rollout on validation, using a frozen prior audit.

Grounder, radar/ego context, instruction and decoding are held fixed. Only the
five-dimensional risk condition changes. Oracle is never a deployment setting.
Missing risk labels exclude a sample from BOTH sides of the comparison.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch


def read_rows(path):
    result = {}
    if not Path(path).exists():
        return result
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['sample_id'] in result:
                raise ValueError('Duplicate sample ID')
            result[row['sample_id']] = row
    return result


def oracle_eligible(row):
    return (len(row['risk_mask']) == 5 and all(row['risk_mask'])
            and len(row['risk_target']) == 5
            and bool(np.isfinite(row['risk_target']).all()))


def score_trajectory(plan, truth, teacher_mode):
    times = np.asarray(truth['future_times_s'], float)
    gt = np.asarray(truth['ego_future'], float)
    valid = np.asarray(truth['ego_future_mask'], bool) & np.isfinite(gt).all(-1)
    required = valid & ((times <= 1.000001) if teacher_mode == 'short' else True)
    cv = times[:, None] * np.asarray(truth['ego_velocity'], float)
    errors, cv_errors = {}, {}
    if plan.get('valid'):
        for point, t in zip(plan.get('ego_trajectory', []), plan.get('ego_times_s', [])):
            found = np.flatnonzero(np.isclose(times, t, atol=1e-5, rtol=0))
            if len(found) != 1 or point is None:
                continue
            i = int(found[0])
            p = np.asarray(point, float)
            if valid[i] and p.shape == (2,) and np.isfinite(p).all():
                errors[str(i)] = float(np.linalg.norm(p - gt[i]))
                cv_errors[str(i)] = float(np.linalg.norm(cv[i] - gt[i]))
    needed = [str(i) for i in np.flatnonzero(required)]
    selected = [i for i in needed if i in errors]
    return dict(required_indices=needed, errors_by_index_m=errors,
                required_points=len(needed), required_covered=len(selected),
                required_errors_m=[errors[i] for i in selected],
                paired_cv_errors_m=[cv_errors[i] for i in selected])


def paired_scores(predicted, oracle, truth, teacher_mode):
    p = score_trajectory(predicted, truth, teacher_mode)
    o = score_trajectory(oracle, truth, teacher_mode)
    shared = [i for i in p['required_indices']
              if i in p['errors_by_index_m'] and i in o['errors_by_index_m']]
    return dict(predicted=p, oracle=o, common_required_indices=shared,
                predicted_common_errors_m=[p['errors_by_index_m'][i] for i in shared],
                oracle_common_errors_m=[o['errors_by_index_m'][i] for i in shared])


def main(a):
    from radar_vla.data import RadarDataset, collate_batch
    from radar_vla.evaluation import evaluate_plan
    from radar_vla.pipeline import (autocast_context, file_digest, load_models,
                                    select_risk, source_fingerprint, to_device,
                                    validate_input_shapes, write_json)
    from radar_vla.task_supervision import stage_available

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    a.output.mkdir(parents=True, exist_ok=True)
    baseline_protocol = json.loads((a.baseline / 'protocol.json').read_text())
    prior = read_rows(a.baseline / 'samples.jsonl')
    source_before = source_fingerprint()
    if source_before != baseline_protocol['source_files']:
        raise ValueError('Model source differs from baseline; finish paired audit before editing src')
    if file_digest(a.checkpoint) != baseline_protocol['checkpoint_sha256']:
        raise ValueError('Checkpoint differs from paired baseline')
    if file_digest(a.manifest) != baseline_protocol['manifest_sha256']:
        raise ValueError('Manifest differs from paired baseline')
    if a.device.startswith('cuda'):
        torch.cuda.set_device(a.device)
        free, _ = torch.cuda.mem_get_info(a.device)
        if free < a.start_free_gib * 1024**3:
            raise RuntimeError('Insufficient GPU headroom before loading; choose another GPU')
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction, a.device)
    print('Loading frozen models; no training or downloads', flush=True)
    grounder, planner, state = load_models(a.checkpoint, a.device)
    if planner is None or planner.config.risk_source != 'predicted' or not planner.config.use_risk_token:
        raise ValueError('Requires the predicted-Risk baseline planner')
    precision = state['protocol'].get('precision', 'float32')
    dataset = RadarDataset(a.manifest, 'val', max_agents=None)
    dataset.records = [r for r in dataset.records if stage_available(r, 'sft')]
    validate_input_shapes(dataset, state['protocol']['config'])
    ids = [r['sample_id'] for r in dataset.records]
    if set(ids) != set(prior):
        raise ValueError('Full SFT validation cohort differs from baseline')
    eligible = [i for i, identifier in enumerate(ids) if oracle_eligible(prior[identifier])]
    shard = [i for j, i in enumerate(eligible) if j % a.shard_count == a.shard_index]
    shard.sort(key=lambda i: (prior[ids[i]]['teacher_mode'] != 'short', i))
    protocol = dict(checkpoint=str(a.checkpoint.resolve()),
        checkpoint_sha256=baseline_protocol['checkpoint_sha256'], epoch=state['epoch'],
        manifest=str(a.manifest.resolve()), manifest_sha256=baseline_protocol['manifest_sha256'],
        baseline_samples_sha256=file_digest(a.baseline / 'samples.jsonl'),
        baseline_protocol_sha256=file_digest(a.baseline / 'protocol.json'),
        source_files=source_before, script_sha256=file_digest(__file__),
        split='val', full_sft_samples=len(ids), full_paired_samples=len(eligible),
        excluded_incomplete_risk=[s for s in ids if not oracle_eligible(prior[s])],
        eligible_ids=[ids[i] for i in shard], shard_index=a.shard_index, shard_count=a.shard_count,
        batch_size=1, seed=42, precision=precision, max_new_tokens=512,
        intervention='replace all five risk scalars only; never partial substitution',
        generation='original full-prefix greedy, no KV cache; unchanged radar/ego/instruction',
        baseline_reuse='frozen prior native batch-one predictions; verify representative tokens before rollout',
        metrics='teacher-required coverage and common timestamp errors; physics on each own horizon',
        limitations='oracle diagnosis only; collision tangent-yaw approximation; no closed-loop safety claim',
        memory_fraction=a.memory_fraction, cpu_threads=1, gpu_shared=True,
        min_runtime_free_gib=a.runtime_free_gib,
        gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'))
    path = a.output / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError('Resume protocol differs')
    write_json(path, protocol)
    # Verify across teacher modes and a moving example before reusing any prior output.
    verify = list(dict.fromkeys(
        [next((i for i in shard if prior[ids[i]]['teacher_mode'] == m), shard[0]) for m in ('short', 'long')]
        + [next((i for i in shard if prior[ids[i]]['speed_mps'] >= 1.), shard[0])]))
    checks = []
    with torch.no_grad(), autocast_context(a.device, precision):
        for i in verify:
            batch = to_device(collate_batch([dataset[i]]), a.device)
            prediction = grounder(batch)
            generated = planner.generate(prediction['radar_tokens'], prediction['risk'], batch['instruction'], max_new_tokens=512)[0]
            identical = generated == prior[ids[i]]['generated_ids']
            risk_delta = float(np.max(np.abs(np.asarray(prior[ids[i]]['risk']) - prediction['risk'][0].float().cpu().numpy())))
            checks.append(dict(sample_id=ids[i], tokens_identical=identical, risk_max_abs_delta=risk_delta))
            if not identical or risk_delta > 1e-5:
                write_json(a.output / 'baseline_verification.json', checks)
                raise ValueError('Native baseline reproduction failed')
    write_json(a.output / 'baseline_verification.json', checks)
    del batch, prediction
    completed = read_rows(a.output / 'samples.jsonl')
    if set(completed) - set(protocol['eligible_ids']):
        raise ValueError('Resume has samples outside shard')
    remaining = [i for i in shard if ids[i] not in completed]
    if a.limit:
        remaining = remaining[:a.limit]
    began = time.monotonic()
    with (a.output / 'samples.jsonl').open('a', buffering=1) as stream, torch.no_grad():
        for ordinal, i in enumerate(remaining, 1):
            if a.device.startswith('cuda') and torch.cuda.mem_get_info(a.device)[0] < a.runtime_free_gib * 1024**3:
                write_json(a.output / 'status.json', dict(state='paused_low_free_memory', completed=len(completed), total=len(shard)))
                return
            record, base = dataset.records[i], prior[ids[i]]
            sample = dataset[i]
            batch = to_device(collate_batch([sample]), a.device)
            expected = np.asarray(base['risk_target'])
            if not torch.equal(batch['risk_mask'][0].cpu(), torch.tensor(base['risk_mask'])):
                raise ValueError('Risk mask differs from baseline')
            if not np.array_equal(batch['risk_target'][0].cpu().numpy(), expected.astype(np.float32)):
                raise ValueError('Risk values differ from baseline')
            with autocast_context(a.device, precision):
                prediction = grounder(batch)
                risk = select_risk(prediction['risk'], batch, 'oracle')
                if a.device.startswith('cuda'):
                    torch.cuda.synchronize(a.device)
                t0 = time.perf_counter()
                generated = planner.generate(prediction['radar_tokens'], risk, batch['instruction'], max_new_tokens=512)[0]
                if a.device.startswith('cuda'):
                    torch.cuda.synchronize(a.device)
                generation_s = time.perf_counter() - t0
            plan = planner.decode(generated, record['future_times_s'])
            if plan.get('mode') != base['mode_oracle']['mode']:
                raise ValueError('Full oracle mode differs from prior oracle first-token probe')
            if not np.allclose(prediction['risk'][0].float().cpu().numpy(), base['risk'], atol=1e-5, rtol=0):
                raise ValueError('Risk prediction drifted from paired baseline')
            scores = paired_scores(base['plan'], plan, base['truth'], base['teacher_mode'])
            local_record = dict(record, risk_agents=record['agents'],
                tracking_coverage_valid=[bool(record.get('agent_supervision_available', True))] * len(record['future_times_s']))
            row = dict(sample_id=ids[i], episode_id=record['episode_id'],
                teacher_mode=base['teacher_mode'], category=base['category'], speed_mps=base['speed_mps'],
                risk_predicted=base['risk'], risk_oracle=base['risk_target'], risk_mask=base['risk_mask'],
                predicted_mode=base['plan'].get('mode'), oracle_mode=plan.get('mode'),
                generated_ids=generated, plan=plan, scores=scores,
                physics_predicted_own_horizon=base['physics_local'],
                physics_oracle_own_horizon=evaluate_plan(local_record, plan),
                generation_seconds=generation_s, generated_tokens=len(generated),
                peak_allocated_mib=torch.cuda.max_memory_allocated(a.device) / 1024**2)
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            completed[ids[i]] = True
            elapsed = time.monotonic() - began
            status = dict(state='running', completed=len(completed), total=len(shard),
                elapsed_s=elapsed, eta_s=(len(shard)-len(completed))*elapsed/ordinal,
                last_sample=ids[i], pid=os.getpid())
            write_json(a.output / 'status.json', status)
            if ordinal <= 3 or ordinal % 20 == 0:
                print(json.dumps(status), flush=True)
            del batch, prediction, sample
    if source_fingerprint() != source_before:
        raise ValueError('Model source changed during audit')
    write_json(a.output / 'status.json', dict(state='complete' if len(completed) == len(shard) else 'pilot_complete',
        completed=len(completed), total=len(shard), source_unchanged=True, pid=os.getpid()))


def self_test():
    truth = dict(future_times_s=[.5, 1., 1.5, 2., 2.5, 3.], ego_future=[[0., 0.]] * 6,
                 ego_future_mask=[True] * 6, ego_velocity=[0., 0.])
    long = dict(valid=True, ego_times_s=truth['future_times_s'], ego_trajectory=[[1., 0.]] * 6)
    short = dict(valid=True, ego_times_s=[.5, 1.], ego_trajectory=[[0., 0.]] * 2)
    scores = paired_scores(long, short, truth, 'long')
    assert scores['predicted']['required_covered'] == 6
    assert scores['oracle']['required_covered'] == 2
    assert scores['predicted_common_errors_m'] == [1., 1.]
    assert scores['oracle_common_errors_m'] == [0., 0.]
    assert not oracle_eligible(dict(risk_target=[0.] * 5, risk_mask=[True] * 4 + [False]))
    assert not oracle_eligible(dict(risk_target=[0., 0., 0., 0., float('nan')], risk_mask=[True] * 5))
    print('PASS: incomplete Oracle excluded; shorter horizon is missing coverage, paired timestamps match')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'manifest', 'baseline', 'output'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--shard-count', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--memory-fraction', type=float, default=.22)
    p.add_argument('--start-free-gib', type=float, default=24.)
    p.add_argument('--runtime-free-gib', type=float, default=8.)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            main(args)
        except BaseException as error:
            if args.output:
                from radar_vla.pipeline import write_json
                write_json(args.output / 'failure.json', dict(error=repr(error), pid=os.getpid()))
            raise
