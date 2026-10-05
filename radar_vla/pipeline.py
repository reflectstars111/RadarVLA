"""Reproducible two-stage training, explicit evaluation, and offline prediction."""

from dataclasses import asdict
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import RadarDataset, collate_batch
from .losses import loss_grounding
from .metrics import MetricAccumulator, planning_metrics
from .model import ModelConfig, RadarVLAGrounder


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def save_checkpoint(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(manifest):
    manifest = Path(manifest).resolve()
    files = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        def sensor_paths(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in ('path', 'power', 'unfolded_doppler', 'folded_doppler', 'doppler_valid', 'doppler_prior',
                               'raw_points', 'raw_cube', 'pointcloud', 'points', 'cube') and isinstance(child, str):
                        path = (manifest.parent / child).resolve()
                        if not path.is_file():
                            raise ValueError(f'Missing sensor source: {path}')
                        if str(path) not in files:
                            files[str(path)] = file_digest(path)
                    else:
                        sensor_paths(child)
            elif isinstance(value, list):
                for child in value:
                    sensor_paths(child)
        sensor_paths(record['sensors'])
    return dict(manifest_sha256=file_digest(manifest), sensor_files=files)


def source_fingerprint():
    return {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def resolve_config(config=None):
    from .planner import PlannerConfig
    config = config or {}
    if set(config) - {'model', 'planner', 'data', 'language_model'}:
        raise ValueError('Unrecognized configuration section')
    model = ModelConfig(**config.get('model', {}))
    planner_options = dict(hidden_dim=model.hidden_dim, radar_dim=model.hidden_dim, num_heads=model.num_heads,
                           max_agents=model.max_agents, horizon_steps=model.horizon_steps)
    planner_options.update(config.get('planner', {}))
    if planner_options.get('risk_source') == 'none':
        planner_options['use_risk_token'] = False
    planner = PlannerConfig(**planner_options)
    if planner.radar_dim != model.hidden_dim:
        raise ValueError('Planner radar_dim must match model hidden_dim')
    for name in ('hidden_dim', 'max_agents', 'horizon_steps'):
        if getattr(model, name) != getattr(planner, name):
            raise ValueError(f'Model and planner disagree on {name}')
    data = dict(config.get('data', {}))
    if set(data) - {'radar_shape', 'history_frames', 'strict_supervision'}:
        raise ValueError('Unrecognized data configuration')
    language = {'backend': 'hf', **config.get('language_model', {})}
    if language['backend'] not in ('tiny', 'hf'):
        raise ValueError('language_model.backend must be tiny or hf')
    return dict(model=asdict(model), planner=asdict(planner), data=data, language_model=language)


def make_planner(config):
    from .planner import PlannerConfig, RiskConditionedPlanner
    options = dict(config.get('language_model', {'backend': 'tiny'}))
    backend = options.pop('backend')
    planner_config = PlannerConfig(**config['planner'])
    if backend == 'hf':
        if not options.get('model_path'):
            raise ValueError('Production SFT requires local Qwen model_path; set backend=tiny explicitly only for tests')
        from .hf_planner import HFRiskConditionedPlanner
        return HFRiskConditionedPlanner(planner_config, **options)
    if options:
        raise ValueError('Tiny backend does not accept HF loading options')
    return RiskConditionedPlanner(planner_config)


def planner_state(planner):
    if planner is None:
        return None
    return planner.checkpoint_state() if hasattr(planner, 'checkpoint_state') else planner.state_dict()


def restore_planner(planner, state):
    if hasattr(planner, 'load_checkpoint_state'):
        planner.load_checkpoint_state(state)
    else:
        planner.load_state_dict(state)


def validate_input_shapes(dataset, config):
    shapes = set()
    for record in dataset.records:
        radar = record['sensors']['radar']
        shape = (2, len(radar['range_m']), len(radar['azimuth_rad']))
        frames = len(radar['time_offsets_s'])
        shapes.add((frames, *shape))
        if 'radar_shape' in config['data'] and list(shape) != config['data']['radar_shape']:
            raise ValueError(f'Input radar_shape {shape} differs from configured radar_shape')
        if 'history_frames' in config['data'] and frames != config['data']['history_frames']:
            raise ValueError('Input history_frames differs from configuration')
        if len(record['future_times_s']) != config['model']['horizon_steps']:
            raise ValueError('Dataset horizon differs from model horizon_steps')
    if len(shapes) != 1:
        raise ValueError('All samples in a split must have the same radar_shape/history_frames')


def language_fingerprint(config):
    if config['language_model']['backend'] != 'hf':
        return None
    root = Path(config['language_model'].get('model_path', '')).resolve()
    if not (root / 'config.json').is_file():
        raise ValueError('HF backend requires a local model_path with config.json')
    files = sorted({*root.glob('*.json'), *root.glob('*.safetensors'), *root.glob('*.bin'),
                    *root.glob('*.model'), *root.glob('*.txt')})
    return {p.name: file_digest(p) for p in files}


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def make_loader(dataset, batch_size, seed, shuffle=False, sampler=None, workers=0):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, num_workers=workers,
                      sampler=sampler,
                      collate_fn=collate_batch, generator=torch.Generator().manual_seed(seed))


def _rng_state(device='cpu'):
    state = dict(torch=torch.get_rng_state(), python=random.getstate())
    if str(device).startswith('cuda'):
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def _restore_rng(state, device='cpu'):
    torch.set_rng_state(state['torch'].cpu())
    random.setstate(state['python'])
    if 'cuda' in state and str(device).startswith('cuda'):
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


class TrainingSystem(torch.nn.Module):
    """A single DDP unit ensures one coordinated backward for both stages."""
    def __init__(self, grounder, planner):
        super().__init__()
        self.grounder, self.planner = grounder, planner

    def forward(self, batch, gt_risk_mix=0.):
        return _loss(self.grounder, self.planner, batch, gt_risk_mix)[0]


def autocast_context(device, precision):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16) if precision == 'bfloat16' else nullcontext()


def select_risk(predicted, batch, source='predicted'):
    if source in ('predicted', 'none'):
        return predicted
    if source != 'oracle':
        raise ValueError('Risk source must be predicted, none, or oracle')
    if ('risk_target' not in batch or 'risk_mask' not in batch
            or not batch['risk_mask'].bool().all() or not torch.isfinite(batch['risk_target']).all()):
        raise ValueError('Oracle risk requires a shared evaluation cohort with all five risk labels known')
    return batch['risk_target']


def _loss(grounder, planner, batch, gt_risk_mix=0.):
    from .planner import create_supervision, token_cross_entropy, planner_physics_loss
    prediction = grounder(batch)
    components = loss_grounding(prediction, batch)
    if planner is None:
        return components['total'], components, prediction
    risk = select_risk(prediction['risk'], batch, planner.config.risk_source)
    if gt_risk_mix and planner.config.risk_source == 'predicted':
        valid = batch['risk_mask'].bool()
        use_gt = (torch.rand(risk.shape[0], 1, device=risk.device) < gt_risk_mix) & valid
        risk = torch.where(use_gt, batch['risk_target'], risk)
    supervision = create_supervision(batch, planner.tokenizer, planner.config)
    targets = supervision['target_ids'].to(risk.device)
    logits = planner(prediction['radar_tokens'], risk, batch['instruction'], targets)
    token = token_cross_entropy(logits, targets, planner.tokenizer,
                                continuous_targets=supervision['continuous_targets'])
    physics = planner_physics_loss(logits, supervision, batch, planner.tokenizer, planner.config)
    components = {**components, 'token': token, **{'planner_' + k: v for k, v in physics.items()}}
    return token + components['total'] + physics['total'], components, prediction


@torch.no_grad()
def validate(grounder, planner, dataset, batch_size, device, precision='float32'):
    grounder.eval()
    if planner is not None:
        planner.eval()
    accumulator = MetricAccumulator()
    total, count = 0., 0
    for batch in make_loader(dataset, batch_size, 0):
        batch = to_device(batch, device)
        with autocast_context(device, precision):
            loss, _, prediction = _loss(grounder, planner, batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        n = len(batch['radar'])
        total += loss.item() * n
        count += n
        accumulator.update(prediction, batch)
    if not count:
        raise ValueError('Validation split is empty')
    return {**accumulator.compute(), 'loss': total / count}


def run_training(manifest, output, stage='grounding', epochs=5, batch_size=2,
                 lr=3e-4, seed=42, device='cpu', config=None, init_grounding=None,
                 resume=False, stop_after_epoch=None, gt_risk_mix=0., precision='float32',
                 accumulation_steps=1, workers=0, language_model_path=None):
    from .planner import create_supervision
    if stage not in ('grounding', 'sft') or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid training stage, budget, or learning rate')
    if precision not in ('float32', 'bfloat16') or accumulation_steps < 1 or workers < 0:
        raise ValueError('Invalid precision, accumulation_steps or workers')
    if not math.isfinite(lr) or not 0 <= gt_risk_mix <= 1:
        raise ValueError('Invalid learning rate or GT risk mixing probability')
    if stop_after_epoch is not None and not 1 <= stop_after_epoch <= epochs:
        raise ValueError('stop_after_epoch must be within the planned budget')
    output, manifest = Path(output).resolve(), Path(manifest).resolve()
    if resume and not (output / 'last.pt').exists():
        raise ValueError('Resume requires last.pt')
    if not resume and output.exists() and any(output.iterdir()):
        raise ValueError('Output directory is not empty; use resume or a new directory')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    distributed = world_size > 1
    if device.startswith('cuda'):
        device = f'cuda:{int(os.environ.get("LOCAL_RANK", "0"))}' if distributed else device
        torch.cuda.set_device(torch.device(device))
        if precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError('Device does not support requested bfloat16')
    if distributed and not dist.is_initialized():
        dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo')
    config = resolve_config(config)
    if language_model_path:
        config['language_model']['model_path'] = str(Path(language_model_path).resolve())
    train = RadarDataset(manifest, 'train', max_agents=config['model']['max_agents'])
    val = RadarDataset(manifest, 'val', max_agents=config['model']['max_agents'])
    if not len(train) or not len(val):
        raise ValueError('Both train and val splits are required')
    validate_input_shapes(train, config)
    validate_input_shapes(val, config)
    initial_digest = file_digest(init_grounding) if init_grounding else None
    fingerprints = [dict(data=data_fingerprint(manifest), source_files=source_fingerprint(),
                         language_model_files=language_fingerprint(config) if stage == 'sft' else None)
                    if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(fingerprints, src=0)
    protocol = dict(schema='radar_vla_pipeline_v2', stage=stage, config=config,
                    epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                    gt_risk_mix=gt_risk_mix, **fingerprints[0], init_grounding_sha256=initial_digest,
                    world_size=world_size, precision=precision, accumulation_steps=accumulation_steps,
                    workers=workers, effective_batch_size=world_size * batch_size * accumulation_steps,
                    distributed_sampling='epoch-seeded DistributedSampler; pads up to world_size-1 examples',
                    selection='lowest validation total loss; no test during training')
    if resume:
        old = json.loads((output / 'protocol.json').read_text())
        if not init_grounding:
            protocol['init_grounding_sha256'] = old.get('init_grounding_sha256')
        if protocol != old:
            raise ValueError('Resume protocol/provenance fingerprint differs (data, source, or configuration)')
    torch.manual_seed(seed)
    random.seed(seed)
    grounder = RadarVLAGrounder(ModelConfig(**config['model'])).to(device)
    planner = make_planner(config).to(device) if stage == 'sft' else None
    if stage == 'sft' and not resume:
        if not init_grounding:
            raise ValueError('SFT requires --init-grounding from Stage 1')
        initial = torch.load(init_grounding, map_location='cpu', weights_only=True)
        if (initial['protocol']['stage'] != 'grounding'
                or initial['protocol']['config']['model'] != config['model']
                or initial['protocol']['data'] != protocol['data']):
            raise ValueError('Grounding checkpoint model or data provenance differs')
        grounder.load_state_dict(initial['grounder'])
    if planner:
        # Scan the whole supervised cohort before the first optimizer step;
        # an isolated valid sample must not conceal missing targets elsewhere.
        for dataset in (train, val):
            eligible = False
            for i in range(len(dataset)):
                sample = collate_batch([dataset[i]])
                if planner.config.risk_source == 'oracle':
                    select_risk(sample['risk_target'], sample, 'oracle')
                supervision = create_supervision(sample, planner.tokenizer, planner.config)
                row = supervision['rows'][0]
                usable = row['mode'] is not None and row['ego'] is not None and bool(row['ego']['fitted'])
                if config['data'].get('strict_supervision', True):
                    identifier = dataset.records[i]['sample_id']
                    if not usable:
                        raise ValueError(f'SFT sample {identifier} lacks a known mode or a fit-able ego trajectory')
                    if 'agent_supervision_mask' in sample and not bool(sample['agent_supervision_mask'].all()):
                        raise ValueError(f'SFT sample {identifier} has unknown agent annotation coverage')
                    if row['mode'] == 'long' and not (row['road']['fitted'] and row['road']['width_mask']):
                        raise ValueError(f'LONG SFT sample {identifier} needs observed map centerline and linked lane boundaries')
                    if row['mode'] == 'long' and any(not a['trajectory']['fitted'] for a in row['agents']):
                        raise ValueError(f'LONG SFT sample {identifier} needs observed future endpoints for every agent')
                if usable:
                    eligible = True
            if not eligible:
                raise ValueError('SFT train/val require a known bootstrap mode and observed ego future')
    parameters = [p for p in list(grounder.parameters()) + (list(planner.parameters()) if planner else [])
                  if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    start, best, history = 1, float('inf'), []
    if resume:
        state = torch.load(output / 'last.pt', map_location=device, weights_only=True)
        if state['protocol'] != protocol:
            raise ValueError('Checkpoint provenance differs from protocol')
        grounder.load_state_dict(state['grounder'])
        if planner:
            restore_planner(planner, state['planner'])
        optimizer.load_state_dict(state['optimizer'])
        _restore_rng(state.get('rank_rng', [state['rng']])[rank], device)
        start, best, history = state['epoch'] + 1, state['best_loss'], state['history']
        if [row['epoch'] for row in history] != list(range(1, start)):
            raise ValueError('Checkpoint history is inconsistent')
        if rank == 0 and state['best_epoch'] == state['epoch']:
            save_checkpoint(output / 'best.pt', state)
        if not (output / 'best.pt').exists():
            raise ValueError('Previous best checkpoint is missing')
    else:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / 'protocol.json', protocol)
    if distributed:
        dist.barrier()
    system = TrainingSystem(grounder, planner)
    if distributed:
        system = torch.nn.parallel.DistributedDataParallel(
            system, device_ids=[torch.device(device).index] if device.startswith('cuda') else None,
            find_unused_parameters=True)
    best_epoch = state['best_epoch'] if resume else None
    try:
        final_epoch = min(epochs, stop_after_epoch or epochs)
        for epoch in range(start, final_epoch + 1):
            grounder.train()
            if planner:
                planner.train()
            total, count = 0., 0
            mixture = gt_risk_mix * max(0., 1 - (epoch - 1) / (epochs - 1)) if epochs > 1 else 0.
            if rank == 0:
                write_json(output / 'status.json', dict(state='training', stage=stage, epoch=epoch,
                                                        epochs=epochs, world_size=world_size, pid=os.getpid()))
            began = time.monotonic()
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True,
                                         seed=seed, drop_last=False) if distributed else None
            if sampler:
                sampler.set_epoch(epoch)
            loader = make_loader(train, batch_size, seed + epoch, shuffle=True, sampler=sampler, workers=workers)
            local_samples = len(sampler) if sampler is not None else len(train)
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(loader):
                batch = to_device(batch, device)
                n = len(batch['radar'])
                window_start = (step // accumulation_steps) * accumulation_steps * batch_size
                window_samples = min(accumulation_steps * batch_size, local_samples - window_start)
                update = (step + 1) % accumulation_steps == 0 or step + 1 == len(loader)
                sync_context = system.no_sync() if distributed and not update else nullcontext()
                with sync_context, autocast_context(device, precision):
                    loss = system(batch, mixture)
                    if not torch.isfinite(loss):
                        raise ValueError(f'Nonfinite training loss at epoch {epoch}')
                    (loss * n / window_samples).backward()
                if update:
                    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item() * n
                count += n
            if distributed:
                totals = torch.tensor([total, count], dtype=torch.float64, device=device)
                dist.all_reduce(totals)
                total, count = totals.tolist()
            validation_box = [validate(grounder, planner, val, batch_size, device, precision) if rank == 0 else None]
            if distributed:
                dist.broadcast_object_list(validation_box, src=0)
            validation = validation_box[0]
            row = dict(epoch=epoch, train_loss=total / count, validation=validation,
                       gt_risk_mix=mixture, seconds=time.monotonic() - began)
            peak = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.startswith('cuda') else None
            peaks = [None] * world_size
            if distributed:
                dist.all_gather_object(peaks, peak)
            else:
                peaks[0] = peak
            row['peak_allocated_mib_by_rank'] = peaks
            history.append(row)
            if validation['loss'] < best:
                best, best_epoch = validation['loss'], epoch
            rng = _rng_state(device)
            rank_rng = [None] * world_size
            if distributed:
                dist.all_gather_object(rank_rng, rng)
            else:
                rank_rng[0] = rng
            if rank == 0:
                state = dict(protocol=protocol, epoch=epoch, grounder=grounder.state_dict(),
                             planner=planner_state(planner), optimizer=optimizer.state_dict(),
                             rng=rng, rank_rng=rank_rng, best_loss=best, best_epoch=best_epoch, history=history)
                save_checkpoint(output / 'last.pt', state)
                if best_epoch == epoch:
                    save_checkpoint(output / 'best.pt', state)
                (output / 'metrics.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in history))
                print(json.dumps(dict(stage=stage, epoch=epoch, train_loss=total / count,
                                      val_loss=validation['loss'])), flush=True)
            if distributed:
                dist.barrier()
        status = dict(state='complete' if history[-1]['epoch'] == epochs else 'paused',
                      stage=stage, epoch=history[-1]['epoch'], epochs=epochs,
                      best_epoch=best_epoch, best_validation_loss=best,
                      test_evaluated=False, world_size=world_size, pid=os.getpid())
        if rank == 0:
            write_json(output / 'status.json', status)
        return status
    except BaseException as error:
        if rank == 0:
            write_json(output / 'status.json', dict(state='failed', stage=stage, error=str(error), pid=os.getpid()))
        raise


def load_models(checkpoint, device='cpu'):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state['protocol'].get('schema') != 'radar_vla_pipeline_v2':
        raise ValueError('Checkpoint predates the full v2 data/trajectory protocol; incompatible checkpoints require retraining')
    config = state['protocol']['config']
    grounder = RadarVLAGrounder(ModelConfig(**config['model'])).to(device)
    grounder.load_state_dict(state['grounder'])
    grounder.eval()
    planner = None
    if state['planner'] is not None:
        if language_fingerprint(config) != state['protocol'].get('language_model_files'):
            raise ValueError('Local language model weights differ from checkpoint provenance')
        planner = make_planner(config).to(device)
        restore_planner(planner, state['planner'])
        planner.eval()
    return grounder, planner, state


def evaluate_checkpoint(checkpoint, manifest, output, split='test', batch_size=2, device='cpu', risk_source=None):
    grounder, planner, state = load_models(checkpoint, device)
    if data_fingerprint(manifest) != state['protocol']['data']:
        raise ValueError('Evaluation data fingerprint differs from frozen training manifest')
    dataset = RadarDataset(manifest, split, max_agents=state['protocol']['config']['model']['max_agents'])
    validate_input_shapes(dataset, state['protocol']['config'])
    report = validate(grounder, planner, dataset, batch_size, device, state['protocol'].get('precision', 'float32'))
    if planner is not None:
        source = risk_source or planner.config.risk_source
        if (source == 'none') != (not planner.config.use_risk_token):
            raise ValueError('Risk-token presence must match the trained checkpoint; use the paired no-risk model')
        rows = prediction_rows(grounder, planner, dataset, state, batch_size, device, risk_source=source)
        report['free_generation'] = planning_metrics(rows, (dataset[i] for i in range(len(dataset))))
        from .evaluation import evaluate_plan, aggregate_plan_metrics
        records = {r['sample_id']: r for r in dataset.records}
        report['physical_planning'] = aggregate_plan_metrics([
            evaluate_plan(records[row['sample_id']], row['plan']) for row in rows])
        from .instruction_metrics import evaluate_instruction, aggregate_instruction_metrics
        report['instruction_following'] = aggregate_instruction_metrics([
            evaluate_instruction(records[row['sample_id']], row['plan']) for row in rows])
        report['risk_source'] = source
        report['amortized_inference_seconds'] = sum(r['amortized_inference_seconds'] for r in rows) / len(rows)
    report.update(split=split, checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=file_digest(checkpoint), epoch=state['epoch'],
                  evaluation='offline risk/physical metrics; no closed-loop safety claim')
    write_json(output, report)
    return report


@torch.no_grad()
def predict_checkpoint(checkpoint, manifest, output, split='test', batch_size=2,
                       device='cpu', max_new_tokens=1024):
    grounder, planner, state = load_models(checkpoint, device)
    dataset = RadarDataset(manifest, split, max_agents=state['protocol']['config']['model']['max_agents'], supervision=False)
    validate_input_shapes(dataset, state['protocol']['config'])
    rows = prediction_rows(grounder, planner, dataset, state, batch_size, device, max_new_tokens)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    return rows


@torch.no_grad()
def prediction_rows(grounder, planner, dataset, state, batch_size, device, max_new_tokens=1024, risk_source='predicted'):
    rows = []
    for batch in make_loader(dataset, batch_size, 0):
        batch = to_device(batch, device)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with autocast_context(device, state['protocol'].get('precision', 'float32')):
            prediction = grounder(batch)
            risk = select_risk(prediction['risk'], batch, risk_source)
            generated = planner.generate(prediction['radar_tokens'], risk,
                                         batch['instruction'], max_new_tokens=max_new_tokens) if planner else None
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        batch_seconds = time.perf_counter() - inference_start
        for b, sample_id in enumerate(batch['sample_id']):
            row = dict(sample_id=sample_id, scene_id=batch['scene_id'][b], risk_source=risk_source,
                       risk=prediction['risk'][b].cpu().tolist(),
                       conditioning_risk=risk[b].cpu().tolist() if planner and planner.config.use_risk_token else None,
                       object_probabilities=prediction['object_logits'][b].sigmoid().cpu().tolist(),
                       agent_state=prediction['agent_state'][b].cpu().tolist(),
                       agent_future=prediction['agent_future'][b].cpu().tolist(), plan=None,
                       amortized_inference_seconds=batch_seconds / len(batch['sample_id']))
            if generated is not None:
                row['plan'] = planner.decode(generated[b], future_times_s=batch['future_times_s'][b].cpu().tolist())
                row['generated_tokens'] = len(generated[b])
            rows.append(row)
    return rows


def run_smoke(output):
    from .synthetic import write_synthetic_frames
    from .records import prepare_frames
    from .data import prepare_labels
    from .cohort import build_cohort, absolute_sensor_paths
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Smoke output must be a new or empty directory')
    torch.set_num_threads(1)
    raw_manifest = write_synthetic_frames(output / 'raw', scenes_per_split=1, frames_per_scene=5, seed=42,
                                          range_bins=256, azimuth_bins=107)
    prepared = prepare_frames(raw_manifest, output / 'data', history_frames=4)
    labeled = prepare_labels(prepared, output / 'labeled.jsonl')
    config = dict(data=dict(radar_shape=[2, 256, 107], history_frames=4), language_model=dict(backend='tiny'),
                  model=dict(hidden_dim=16, num_heads=2, num_queries=4, max_agents=4, horizon_steps=6),
                  planner=dict(hidden_dim=16, num_heads=2, num_layers=1, max_agents=4, horizon_steps=6))
    manifest = output / 'cohort.jsonl'
    build_cohort(labeled, manifest, config=config)
    run_training(manifest, output / 'grounding', epochs=1, config=config)
    run_training(manifest, output / 'sft', stage='sft', epochs=1, config=config,
                 init_grounding=output / 'grounding/best.pt')
    evaluate_checkpoint(output / 'sft/best.pt', manifest, output / 'test_metrics.json')
    predictions = predict_checkpoint(output / 'sft/best.pt', manifest, output / 'predictions.jsonl')
    observations = []
    for line in raw_manifest.read_text().splitlines():
        record = json.loads(line)
        if record['split'] != 'test':
            continue
        for field in ('agents', 'map', 'tracking', 'tracking_coverage_valid'):
            record.pop(field, None)
        record['ego'].pop('future_trajectory', None)
        record['sensors'] = absolute_sensor_paths(record['sensors'], raw_manifest.parent)
        observations.append(record)
    observation_raw = output / 'observations.jsonl'
    observation_raw.write_text(''.join(json.dumps(r)+'\n' for r in observations))
    observation_manifest = prepare_frames(observation_raw, output / 'observations', history_frames=4, supervision=False)
    unannotated_predictions = predict_checkpoint(output / 'sft/best.pt', observation_manifest, output / 'unannotated_predictions.jsonl')
    report = dict(synthetic=True, status='complete', stages=['raw_frame_import', 'risk_labels', 'common_cohort', 'grounding', 'sft', 'evaluation', 'observation_only_prediction'],
                  samples_predicted=len(predictions), output=str(output),
                  unannotated_samples_predicted=len(unannotated_predictions),
                  note='Synthetic integration check with randomly initialized tiny decoder; not model quality evidence')
    write_json(output / 'smoke_summary.json', report)
    return report
