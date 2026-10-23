"""Paired Grounder experiment: full RA versus a calibrated 50 m input crop.

No production modules or labels are modified. Both models consume the same
raw batches, with independent AdamW states and bitwise identical initialization.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from radar_vla import data as data_module
from radar_vla.data import RadarDataset, collate_batch
from radar_vla.detection import decode_detections
from radar_vla.detection_metrics import DetectionMetrics
from radar_vla.losses import loss_grounding
from radar_vla.metrics import binary_metrics
from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.pipeline import (autocast_context, data_fingerprint, file_digest,
                               make_loader, source_fingerprint, to_device,
                               validate_input_shapes, write_json)
from radar_vla.task_supervision import stage_available

ARMS = ('full118', 'crop50')


def crop_batch(batch, limit_m=50.):
    """Crop native range bins, synchronizing every per-pixel input field.

    Targets, per-agent radar associations and extrinsics retain the canonical
    labels; this is deliberately an input-only intervention.
    """
    axes = batch['range_m']
    keep = axes <= limit_m
    counts = keep.sum(-1)
    if axes.ndim != 2 or not torch.all(axes.diff(dim=-1) > 0):
        raise ValueError('Expected increasing [B,R] native range axes')
    if not bool((counts == counts[0]).all()) or int(counts[0]) < 2:
        raise ValueError('Crop must retain equal, nonempty native range prefixes')
    n = int(counts[0])
    result = dict(batch)
    if not bool(keep[:, :n].all()) or bool(keep[:, n:].any()):
        raise ValueError('Range crop must be a prefix')
    for key in ('radar', 'radar_doppler_valid', 'radar_cartesian', 'radar_los'):
        value = batch[key]
        dim = 3 if key == 'radar' else 2
        if value.shape[dim] != axes.shape[1]:
            raise ValueError(f'{key} is not aligned with the range axis')
        result[key] = value.narrow(dim, 0, n).contiguous()
    result['range_m'] = axes[:, :n].contiguous()
    return result


def state_hash(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def critical_matches(decoded, item, record):
    """Class-agnostic 2 m one-to-one matching, matching canonical metrics."""
    agents = sorted(record['agents'], key=lambda a: (np.linalg.norm(a['position']), str(a['id'])))
    gt = item['agent_state'][0].detach().cpu().numpy()[:len(agents)]
    source = set(map(str, record['local_risk']['pcol_source_agent_ids'][2]))
    found = {}
    if decoded and len(agents):
        distance = np.linalg.norm(np.asarray([d['position_m'] for d in decoded])[:, None] - gt[None, :, :2], axis=-1)
        cost = np.where(distance <= 2., distance, (len(decoded) + len(agents) + 1) * 3.)
        pi, gi = linear_sum_assignment(cost)
        found = {int(g): decoded[int(p)] for p, g in zip(pi, gi) if distance[p, g] <= 2.}
    result = []
    for i, agent in enumerate(agents):
        if str(agent['id']) not in source:
            continue
        row = dict(id=str(agent['id']), matched=i in found,
                   range_m=float(np.linalg.norm(agent['position'])), velocity_error_mps=None,
                   radial_error_mps=None, future_ade_m=None)
        if i in found:
            d = found[i]
            sm = item['agent_state_mask'][0, i].bool().cpu().numpy()
            if sm[3:5].all():
                row['velocity_error_mps'] = float(np.linalg.norm(np.asarray(d['velocity_mps']) - gt[i, 3:5]))
            if sm[2]:
                row['radial_error_mps'] = abs(float(d['radial_velocity_mps']) - float(gt[i, 2]))
            fm = item['agent_future_mask'][0, i].bool().cpu().numpy()
            if fm.any():
                error = np.linalg.norm(np.asarray(d['future_xy']) - item['agent_future'][0, i].cpu().numpy(), axis=-1)
                row['future_ade_m'] = float(error[fm].mean())
        result.append(row)
    return result


@torch.no_grad()
def evaluate(models, dataset, device, output, epoch, status):
    for model in models.values():
        model.eval()
    metrics = {a: DetectionMetrics(2., 'agent_v1') for a in ARMS}
    rows = {a: [] for a in ARMS}
    losses = {a: 0. for a in ARMS}
    records = {r['sample_id']: r for r in dataset.records}
    for i, raw in enumerate(make_loader(dataset, 1, 0, workers=0)):
        raw = to_device(raw, device)
        for arm in ARMS:
            batch = crop_batch(raw) if arm == 'crop50' else raw
            with autocast_context(device, 'bfloat16'):
                pred = models[arm](batch)
                loss = loss_grounding(pred, batch)['total']
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite validation loss')
            losses[arm] += float(loss)
            decoded = decode_detections(pred, models[arm].config, batch['future_times_s'])
            metrics[arm].update(decoded, batch)
            sid = batch['sample_id'][0]
            rows[arm].append(dict(sample_id=sid, risk=pred['risk'][0].float().cpu().tolist(),
                target=batch['risk_target'][0].cpu().tolist(), mask=batch['risk_mask'][0].cpu().tolist(),
                critical=critical_matches(decoded[0]['detections'], batch, records[sid])))
        if (i + 1) % 100 == 0:
            status('validation', epoch=epoch, done=i + 1, total=len(dataset))
    summaries = {}
    for arm in ARMS:
        values = rows[arm]
        risk = np.asarray([r['risk'] for r in values])
        target = np.asarray([r['target'] for r in values])
        mask = np.asarray([r['mask'] for r in values], bool)
        physical = {}
        for j in range(3):
            take = mask[:, j]
            p, y = risk[take, j], target[take, j]
            physical[f'pcol{j+1}'] = binary_metrics(p, y)
            physical[f'pcol{j+1}'].update(tp=int(((p >= .5) & (y == 1)).sum()),
                fn=int(((p < .5) & (y == 1)).sum()), fp=int(((p >= .5) & (y == 0)).sum()),
                threshold=.5, note='Risk event threshold; not LLM SHORT/LONG recall')
        for j, name in [(3, 'dmin'), (4, 'areq')]:
            error = abs(risk[mask[:, j], j] - target[mask[:, j], j])
            physical[name] = dict(count=len(error), mae=float(error.mean()) if len(error) else None)
        critical = [c for r in values for c in r['critical']]
        matched = sum(c['matched'] for c in critical)
        cm = dict(objects=len(critical), matched=matched, recall=matched / len(critical) if critical else None)
        for key in ['velocity_error_mps', 'radial_error_mps', 'future_ade_m']:
            observed = [c[key] for c in critical if c[key] is not None]
            cm[key] = dict(count=len(observed), mean=float(np.mean(observed)) if observed else None)
        summaries[arm] = dict(loss=losses[arm] / len(dataset), samples=len(dataset),
            detection=metrics[arm].compute(), risk=physical, critical_agents=cm)
        (output / f'val_{arm}_epoch_{epoch:03d}.jsonl').write_text(
            ''.join(json.dumps(r, allow_nan=False) + '\n' for r in values))
    return summaries


def smoke(args):
    dataset = RadarDataset(args.smoke_manifest, 'val', max_agents=None)
    raw = collate_batch([dataset[0]])
    models = []
    for arm in ARMS:
        torch.manual_seed(42)
        model = RadarVLAGrounder(ModelConfig(**json.loads(args.config.read_text())['model']))
        model.train()
        batch = crop_batch(raw) if arm == 'crop50' else raw
        pred = model(batch)
        loss = loss_grounding(pred, batch)['total']
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        models.append(model)
        print(json.dumps(dict(arm=arm, shape=list(batch['radar'].shape), loss=float(loss),
                             candidates=int(pred['detection_logits'].shape[1]))), flush=True)
    assert state_hash(models[0]) == state_hash(models[1])
    print('CPU_SMOKE_PASS identical initialization; finite forward/backward in both arms')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--smoke-manifest', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.smoke_manifest:
        smoke(args)
        return
    assert args.epochs == 5, 'Predeclared budget is five epochs per arm'
    args.output.mkdir(parents=True, exist_ok=True)
    args.assets.mkdir(parents=True, exist_ok=True)
    assert not (args.output / 'protocol.json').exists(), 'Never overwrite a started experiment'
    device = 'cuda:0'
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.12, 0)
    assert torch.cuda.mem_get_info()[0] >= 32 * 1024**3, 'Insufficient starting free VRAM'
    finished = threading.Event()
    def status(phase, **extra):
        record = dict(state='running', phase=phase, pid=os.getpid(), **extra)
        write_json(args.output / 'status.json', record)
        print(json.dumps(record), flush=True)
    def guard():
        while not finished.wait(10):
            if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
                write_json(args.output / 'resource_guard.json', dict(pid=os.getpid(), reason='Self exit below 8 GiB free VRAM'))
                os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=guard, daemon=True).start()
    try:
        status('validating_manifest_once')
        loader = data_module.load_manifest
        records = loader(args.manifest)
        # Reuse the same validated metadata, without retaining duplicate 1.3 GB parses.
        data_module.load_manifest = lambda *a, **kw: records
        try:
            train = RadarDataset(args.manifest, 'train', max_agents=None)
            val = RadarDataset(args.manifest, 'val', max_agents=None)
        finally:
            data_module.load_manifest = loader
        train.records = [r for r in train.records if stage_available(r, 'grounding')]
        val.records = [r for r in val.records if stage_available(r, 'grounding')]
        episodes = {s: sorted({r['episode_id'] for r in ds.records}) for s, ds in [('train', train), ('val', val)]}
        assert not set(episodes['train']) & set(episodes['val'])
        assert len(train) == 6847 and len(val) == 1472
        del records
        config = json.loads(args.config.read_text())
        validate_input_shapes(train, config)
        validate_input_shapes(val, config)
        assert all(sum(x <= 50. for x in r['sensors']['radar']['range_m']) == 109
                   for ds in (train, val) for r in ds.records)
        torch.manual_seed(42)
        initial = RadarVLAGrounder(ModelConfig(**config['model']))
        initial_hash = state_hash(initial)
        models = {a: copy.deepcopy(initial).to(device) for a in ARMS}
        assert all(state_hash(m) == initial_hash for m in models.values())
        del initial
        status('fingerprinting_sensor_files_once')
        protocol = dict(schema='paired_range_control_v1', config=config, epochs=5, seed=42,
            batch_size=1, accumulation_steps=8, effective_batch_size=8, precision='bfloat16',
            lr=.0003, optimizer='AdamW', weight_decay=.0001, gradient_clip=5., workers=0,
            initialized_from='random identical weights; no legacy checkpoint; no LLM instantiated',
            initial_state_sha256=initial_hash, episodes=episodes, counts=dict(train=len(train), val=len(val)),
            transforms=dict(full118='identity, 256 native range bins', crop50='range_m <= 50, 109 native bins; no resize'),
            labels='Identical canonical labels/masks/agent associations; boundary positives retained',
            causal_scope='Whole input-crop effect, including changed normalization, pooling and anchor grid',
            original_ddp_difference='Both arms use one GPU and accumulation=8; original run used two GPUs and accumulation=4',
            test_evaluated=False, selection='Fixed epoch 5 primary; all epochs reported, no validation-tuned thresholds',
            resources=dict(memory_fraction=.12, cpu_threads=1, min_start_free_gib=32, min_runtime_free_gib=8),
            source_files=source_fingerprint(), experiment_script_sha256=file_digest(__file__),
            data=data_fingerprint(args.manifest))
        write_json(args.output / 'protocol.json', protocol)
        optimizers = {a: torch.optim.AdamW(models[a].parameters(), lr=.0003, weight_decay=.0001) for a in ARMS}
        history = []
        for epoch in range(1, 6):
            started = time.monotonic()
            for a in ARMS:
                models[a].train()
                optimizers[a].zero_grad(set_to_none=True)
            totals = {a: 0. for a in ARMS}
            order = hashlib.sha256()
            for step, raw in enumerate(make_loader(train, 1, 42 + epoch, shuffle=True, workers=0)):
                order.update((raw['sample_id'][0] + '\n').encode())
                raw = to_device(raw, device)
                window = min(8, len(train) - (step // 8) * 8)
                for a in ARMS:
                    batch = crop_batch(raw) if a == 'crop50' else raw
                    with autocast_context(device, 'bfloat16'):
                        pred = models[a](batch)
                        loss = loss_grounding(pred, batch)['total']
                    if not bool(torch.isfinite(loss)):
                        raise ValueError(f'Nonfinite loss {a} epoch {epoch} step {step}')
                    (loss / window).backward()
                    totals[a] += float(loss.detach())
                    if (step + 1) % 8 == 0 or step + 1 == len(train):
                        torch.nn.utils.clip_grad_norm_(models[a].parameters(), 5., error_if_nonfinite=True)
                        optimizers[a].step()
                        optimizers[a].zero_grad(set_to_none=True)
                if (step + 1) % 100 == 0:
                    status('paired_training', epoch=epoch, done=step + 1, total=len(train),
                           running_loss={a: totals[a] / (step + 1) for a in ARMS})
            validation = evaluate(models, val, device, args.output, epoch, status)
            row = dict(epoch=epoch, train_loss={a: totals[a] / len(train) for a in ARMS},
                       validation=validation, sample_order_sha256=order.hexdigest(), seconds=time.monotonic() - started)
            history.append(row)
            for a in ARMS:
                state = dict(schema='paired_range_control_checkpoint_v1', protocol=protocol, arm=a, epoch=epoch,
                             grounder=models[a].state_dict(), optimizer=optimizers[a].state_dict(),
                             history=history, input_range_m=50. if a == 'crop50' else None)
                torch.save(state, args.assets / f'{a}_epoch_{epoch:03d}.pt')
            write_json(args.output / 'metrics.json', history)
            status('epoch_complete', epoch=epoch, seconds=row['seconds'])
        assert source_fingerprint() == protocol['source_files']
        assert file_digest(__file__) == protocol['experiment_script_sha256']
        assert file_digest(args.manifest) == protocol['data']['manifest_sha256']
        write_json(args.output / 'results.json', dict(protocol=protocol, history=history, primary=history[-1]['validation'],
            checks=dict(source_unchanged=True, manifest_unchanged=True, matched_initialization=True,
                        shared_raw_batches=True, train_val_episodes_disjoint=True, test_evaluated=False),
            note='One paired seed; no LLM/mode/rollout result; crop also changes pooling/anchor layout'))
        write_json(args.output / 'status.json', dict(state='complete', epoch=5, pid=os.getpid()))
    except BaseException as error:
        write_json(args.output / 'status.json', dict(state='failed', error=str(error), pid=os.getpid()))
        raise
    finally:
        finished.set()


if __name__ == '__main__':
    main()
