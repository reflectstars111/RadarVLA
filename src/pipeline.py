"""Reproducible two-stage training, explicit evaluation, and offline prediction."""

from dataclasses import asdict
from contextlib import nullcontext
from datetime import timedelta
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
from tqdm.auto import tqdm

from .data import RadarDataset, collate_batch
from .losses import loss_grounding
from .metrics import MetricAccumulator
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
    with manifest.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            # RadarDataset only loads these arrays. The manifest digest covers labels,
            # metadata and paths; hashing unused camera/lidar assets can delay DDP's
            # first broadcast long enough for the waiting ranks to time out.
            radar = record['sensors']['radar']
            for key in ('power', 'unfolded_doppler', 'doppler_valid'):
                if key not in radar:
                    continue
                values = radar[key] if isinstance(radar[key], list) else [radar[key]]
                for value in values:
                    if not isinstance(value, str):
                        raise ValueError(f'Sensor path {key} must be a string')
                    path = (manifest.parent / value).resolve()
                    if not path.is_file():
                        raise ValueError(f'Missing sensor source: {path}')
                    if str(path) not in files:
                        files[str(path)] = file_digest(path)
    return dict(manifest_sha256=file_digest(manifest), sensor_files=files)


def source_fingerprint():
    return {p.name: file_digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def _validated_loss_weights(config):
    loss_weights = config.get('loss_weights', {})
    from .losses import DEFAULT_WEIGHTS
    if (not isinstance(loss_weights, dict)
            or set(loss_weights) - (set(DEFAULT_WEIGHTS) - {k for k in DEFAULT_WEIGHTS if k.startswith('scale_')})):
        raise ValueError('Unknown loss weight; physical normalization scales are fixed')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
           for value in loss_weights.values()):
        raise ValueError('Loss weights must be finite nonnegative numbers')
    return dict(loss_weights)


def resolve_config(config=None):
    from .planner import PlannerConfig
    config = config or {}
    if set(config) - {'model', 'planner', 'data', 'language_model', 'loss_weights'}:
        raise ValueError('Unrecognized configuration section')
    loss_weights = _validated_loss_weights(config)
    model = ModelConfig(**config.get('model', {}))
    planner_options = dict(hidden_dim=model.hidden_dim, radar_dim=model.hidden_dim, num_heads=model.num_heads,
                           max_agents=PlannerConfig().max_agents if model.use_detection else model.max_agents,
                           horizon_steps=model.horizon_steps,risk_horizon_schema=model.risk_horizon_schema)
    planner_options.update(config.get('planner', {}))
    if planner_options.get('risk_source') == 'none':
        planner_options['use_risk_token'] = False
    planner = PlannerConfig(**planner_options)
    if planner.risk_horizon_schema != model.risk_horizon_schema:
        raise ValueError('Grounder and Planner Risk horizons differ')
    if planner.radar_dim != model.hidden_dim:
        raise ValueError('Planner radar_dim must match model hidden_dim')
    for name in ('hidden_dim', 'horizon_steps'):
        if getattr(model, name) != getattr(planner, name):
            raise ValueError(f'Model and planner disagree on {name}')
    if not model.use_detection and model.max_agents != planner.max_agents:
        raise ValueError('Legacy model and planner disagree on max_agents')
    data = dict(config.get('data', {}))
    if set(data) - {'radar_shape', 'history_frames', 'strict_supervision', 'require_detection_classes', 'agent_ignore_policy', 'agent_selection_policy', 'risk_supervision_schema', 'supervision_protocol'}:
        raise ValueError('Unrecognized data configuration')
    if 'require_detection_classes' in data and type(data['require_detection_classes']) is not bool:
        raise ValueError('require_detection_classes must be boolean')
    from .moving_agents import POLICY
    if data.get('agent_selection_policy') not in (None, POLICY):
        raise ValueError('Unknown Agent selection policy')
    if data.get('agent_selection_policy') == POLICY and (data.get('agent_ignore_policy') != 'none' or data.get('supervision_protocol') != 'taskwise_local_v1' or not model.use_detection or model.detection_max_objects is not None or model.detection_max_candidates is not None):
        raise ValueError('Motion policy requires taskwise dense detection without count caps or ignored regions')
    from .local_risk import LOCAL_RISK_SCHEMA, LOCAL_RISK_SCHEMAS, LEGACY_RISK_SCHEMA
    if data.get('risk_supervision_schema', LEGACY_RISK_SCHEMA) not in (*LOCAL_RISK_SCHEMAS, LEGACY_RISK_SCHEMA):
        raise ValueError('Unrecognized Risk supervision schema')
    from .risk_contract import horizon_schema
    if horizon_schema(data.get('risk_supervision_schema',LEGACY_RISK_SCHEMA)) != model.risk_horizon_schema:
        raise ValueError('Configuration data/model Risk horizons differ')
    from .task_supervision import TASKWISE_PROTOCOL, LEGACY_PROTOCOL
    if data.get('supervision_protocol',LEGACY_PROTOCOL) not in (TASKWISE_PROTOCOL,LEGACY_PROTOCOL):raise ValueError('Unknown task supervision protocol')
    if data.get('supervision_protocol') == TASKWISE_PROTOCOL:
        if data.get('risk_supervision_schema') not in LOCAL_RISK_SCHEMAS or model.detection_class_schema != 'agent_v1':raise ValueError('Taskwise protocol requires Agent/Local Risk schemas')
        if planner.reasoning_policy != 'adaptive' or (planner.short_collision_threshold,planner.short_clearance_threshold_m,planner.short_brake_threshold_mps2) != (.5,.5,3.):raise ValueError('Taskwise Mode policy thresholds differ')
    language = {'backend': 'hf', **config.get('language_model', {})}
    if language['backend'] not in ('tiny', 'hf'):
        raise ValueError('language_model.backend must be tiny or hf')
    result = dict(model=asdict(model), planner=asdict(planner), data=data, language_model=language)
    # Absent means the exact historical protocol; explicit overrides are pinned
    # in checkpoints and therefore participate in native resume validation.
    if 'loss_weights' in config:
        result['loss_weights'] = dict(loss_weights)
    return result


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
    config = {**config, 'data': config.get('data', {})}
    from .risk_contract import horizon_schema, LEGACY_HORIZONS
    if horizon_schema(dataset.risk_schema) != config['model'].get('risk_horizon_schema',LEGACY_HORIZONS):
        raise ValueError('Dataset and model Risk horizons differ')
    from .task_supervision import LEGACY_PROTOCOL
    if getattr(dataset,'supervision_protocol',LEGACY_PROTOCOL) != config['data'].get('supervision_protocol',LEGACY_PROTOCOL):
        raise ValueError('Dataset and configuration task supervision protocols differ')
    from .local_risk import LEGACY_RISK_SCHEMA
    if dataset.risk_schema != config.get('data', {}).get('risk_supervision_schema', LEGACY_RISK_SCHEMA):
        raise ValueError('Dataset and configuration Risk supervision schemas differ')
    if dataset.class_schema != config['model'].get('detection_class_schema', 'legacy4_v1'):
        raise ValueError('Dataset and model detection_class_schema differ; select the matching versioned config')
    shapes = set()
    for record in dataset.records:
        if record.get('agent_selection_policy') != config['data'].get('agent_selection_policy'):
            raise ValueError('Dataset Agent selection policy differs from configuration')
        if config['data'].get('agent_ignore_policy') is not None and record.get('agent_ignore_policy') != config['data']['agent_ignore_policy']:
            raise ValueError('Dataset agent_ignore_policy differs from configuration')
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
        if (dataset.supervision and config['model'].get('use_detection', False)
                and config['data'].get('require_detection_classes', False)):
            from .categories import category_name
            if any(category_name(a) == 'unknown' for a in record.get('agents', [])):
                raise ValueError(f'Sample {record["sample_id"]} has unknown agent category; provide explicit labels for multiclass detection')
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
    def __init__(self, grounder, planner, detach_risk_condition=False):
        super().__init__()
        self.grounder, self.planner = grounder, planner
        self.detach_risk_condition = detach_risk_condition

    def forward(self, batch, gt_risk_mix=0.):
        return _loss(self.grounder, self.planner, batch, gt_risk_mix, self.detach_risk_condition)[0]


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


def _loss(grounder, planner, batch, gt_risk_mix=0., detach_risk_condition=False):
    from .planner import create_supervision, token_cross_entropy, planner_physics_loss
    prediction = grounder(batch)
    components = loss_grounding(prediction, batch, weights=getattr(grounder, 'loss_weights', None))
    if planner is None:
        return components['total'], components, prediction
    risk = select_risk(prediction['risk'], batch, planner.config.risk_source)
    if gt_risk_mix and planner.config.risk_source == 'predicted':
        valid = batch['risk_mask'].bool()
        use_gt = (torch.rand(risk.shape[0], 1, device=risk.device) < gt_risk_mix) & valid
        risk = torch.where(use_gt, batch['risk_target'], risk)
    if detach_risk_condition:
        # Keep physical Risk supervised by its own loss. Radar tokens still
        # receive planner gradients; this is different from freezing Grounder.
        risk = risk.detach()
    supervision = create_supervision(batch, planner.tokenizer, planner.config)
    targets = supervision['target_ids'].to(risk.device)
    logits = planner(prediction['radar_tokens'], risk, batch['instruction'], targets)
    token = token_cross_entropy(logits, targets, planner.tokenizer,
                                continuous_targets=supervision['continuous_targets'],
                                mode_weight=planner.config.mode_loss_weight)
    physics = planner_physics_loss(logits, supervision, batch, planner.tokenizer, planner.config)
    components = {**components, 'token': token, **{'planner_' + k: v for k, v in physics.items()}}
    return token + components['total'] + physics['total'], components, prediction


@torch.no_grad()
def validate(grounder, planner, dataset, batch_size, device, precision='float32',
             progress_desc=None):
    grounder.eval()
    if planner is not None:
        planner.eval()
    accumulator = MetricAccumulator(grounder.config.risk_horizon_schema)
    if grounder.config.use_detection:
        from .detection import decode_detections
        from .detection_metrics import DetectionMetrics
        detection_metrics = DetectionMetrics(grounder.config.detection_match_distance_m, grounder.config.detection_class_schema)
    else:
        detection_metrics = None
    total, count = 0., 0
    loader = make_loader(dataset, batch_size, 0)
    for batch in tqdm(loader, total=len(loader), desc=progress_desc or 'validation',
                      unit='batch', disable=progress_desc is None):
        batch = to_device(batch, device)
        with autocast_context(device, precision):
            loss, _, prediction = _loss(grounder, planner, batch)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite validation loss')
        n = len(batch['radar'])
        total += loss.item() * n
        count += n
        accumulator.update(prediction, batch)
        if detection_metrics is not None:
            detection_metrics.update(decode_detections(prediction, grounder.config, batch['future_times_s']), batch)
    if not count:
        raise ValueError('Validation split is empty')
    return {**accumulator.compute(), 'loss': total / count,
            'detection': detection_metrics.compute() if detection_metrics is not None else None}


@torch.no_grad()
def validate_generation(grounder, planner, dataset, indices, teacher_modes,
                        batch_size, device, precision='float32'):
    """Greedy validation on a fixed audit subset, with no GT in model inputs."""
    from .training_quality import generation_diagnostics
    grounder.eval()
    planner.eval()
    rows = []
    subset = torch.utils.data.Subset(dataset, indices)
    for batch in make_loader(subset, batch_size, 0):
        batch = to_device(batch, device)
        with autocast_context(device, precision):
            prediction = grounder(batch)
            risk = select_risk(prediction['risk'], batch, planner.config.risk_source)
            ids = planner.generate(prediction['radar_tokens'], risk, batch['instruction'],
                                   max_new_tokens=planner.config.max_length)
        for b, tokens in enumerate(ids):
            rows.append(dict(sample_id=batch['sample_id'][b], generated_tokens=len(tokens),
                             plan=planner.decode(tokens, batch['future_times_s'][b].cpu().tolist())))
    report = generation_diagnostics(rows, [dataset[i] for i in indices],
        [teacher_modes[i] for i in indices], planner.config.short_horizon_s)
    report.update(selection_split='val', sample_ids=[dataset.records[i]['sample_id'] for i in indices],
                  subset='fixed seeded mode-stratified audit; not a prevalence estimate')
    return report


def run_training(manifest, output, stage='grounding', epochs=5, batch_size=2,
                 lr=3e-4, seed=42, device='cpu', config=None, init_grounding=None,
                 resume=False, stop_after_epoch=None, gt_risk_mix=0., precision='float32',
                 accumulation_steps=1, workers=0, language_model_path=None,
                 validation_generation_samples=32, freeze_grounder=False,
                 detach_risk_condition=False, lr_schedule='constant',
                 lr_decay_start_epoch=5, lr_min_ratio=.1, lr_schedule_unit='epoch'):
    from .training_schedule import learning_rate_for_epoch, learning_rate_for_update
    if lr_schedule_unit not in ('epoch', 'update'):
        raise ValueError('lr_schedule_unit must be epoch or update')
    if lr_schedule_unit == 'epoch':
        learning_rate_for_epoch(1, epochs, lr, lr_schedule, lr_decay_start_epoch, lr_min_ratio)
    elif (type(lr_decay_start_epoch) is not int or lr_decay_start_epoch < 0
            or type(epochs) is not int or type(batch_size) is not int
            or type(accumulation_steps) is not int):
        raise ValueError('Invalid update schedule epoch/batch budget')
    from .planner import create_supervision
    if type(freeze_grounder) is not bool or type(detach_risk_condition) is not bool:
        raise ValueError('Risk training controls must be bool')
    if (freeze_grounder or detach_risk_condition) and stage != 'sft':
        raise ValueError('Risk training controls require the sft stage')
    if freeze_grounder and detach_risk_condition:
        raise ValueError('Use separate freeze and detached-Risk controls to avoid confounding')
    if stage not in ('grounding', 'sft') or epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid training stage, budget, or learning rate')
    if precision not in ('float32', 'bfloat16') or accumulation_steps < 1 or workers < 0:
        raise ValueError('Invalid precision, accumulation_steps or workers')
    if not math.isfinite(lr) or not 0 <= gt_risk_mix <= 1:
        raise ValueError('Invalid learning rate or GT risk mixing probability')
    if type(validation_generation_samples) is not int or validation_generation_samples < 0:
        raise ValueError('validation_generation_samples must be a nonnegative integer')
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
        if distributed:
            device = f'cuda:{int(os.environ.get("LOCAL_RANK", "0"))}'
        elif device == 'cuda':
            device = 'cuda:0'
        torch.cuda.set_device(torch.device(device))
        if precision == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise ValueError('Device does not support requested bfloat16')
    if distributed and not dist.is_initialized():
        # Rank zero validates the full split while other ranks wait for its
        # broadcast. A full validation pass can exceed the default 10 minutes.
        dist.init_process_group('nccl' if device.startswith('cuda') else 'gloo',
                                timeout=timedelta(hours=1))
    config = resolve_config(config)
    if language_model_path:
        config['language_model']['model_path'] = str(Path(language_model_path).resolve())
    # Dense grounding has variable target counts. Planner grammar capacity is
    # checked separately by create_supervision only when running Stage 2.
    capacity = None if config['model']['use_detection'] else config['model']['max_agents']
    if rank == 0:
        print('Loading and validating training manifest...', flush=True)
    train = RadarDataset(manifest, 'train', max_agents=capacity)
    val = RadarDataset(manifest, 'val', max_agents=capacity)
    stage_excluded_samples = {}
    from .task_supervision import stage_available
    for split_name, dataset in (('train', train), ('val', val)):
        stage_excluded_samples[split_name] = [r['sample_id'] for r in dataset.records if not stage_available(r,stage)]
        dataset.records = [r for r in dataset.records if stage_available(r,stage)]
    if not len(train) or not len(val):
        raise ValueError('Both train and val splits are required')
    samples_per_rank = math.ceil(len(train) / world_size)
    updates_per_epoch = math.ceil(math.ceil(samples_per_rank / batch_size) / accumulation_steps)
    total_updates = epochs * updates_per_epoch
    flat_updates = lr_decay_start_epoch * updates_per_epoch
    if lr_schedule_unit == 'update':
        learning_rate_for_update(1, total_updates, lr, lr_schedule, flat_updates, lr_min_ratio)
    validate_input_shapes(train, config)
    validate_input_shapes(val, config)
    initial_digest = file_digest(init_grounding) if init_grounding else None
    if rank == 0:
        print('Hashing radar inputs for the training fingerprint...', flush=True)
    fingerprints = [dict(data=data_fingerprint(manifest), source_files=source_fingerprint(),
                         language_model_files=language_fingerprint(config) if stage == 'sft' else None)
                    if rank == 0 else None]
    if distributed:
        dist.broadcast_object_list(fingerprints, src=0)
    protocol = dict(schema='radar_vla_pipeline_v2', stage=stage, config=config,
                    epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
                    gt_risk_mix=gt_risk_mix, **fingerprints[0], init_grounding_sha256=initial_digest,
                    stage_excluded_samples=stage_excluded_samples,
                    world_size=world_size, precision=precision, accumulation_steps=accumulation_steps,
                    workers=workers, effective_batch_size=world_size * batch_size * accumulation_steps,
                    validation_generation_samples=validation_generation_samples,
                    freeze_grounder=freeze_grounder, detach_risk_condition=detach_risk_condition,
                    distributed_sampling='epoch-seeded DistributedSampler; pads up to world_size-1 examples',
                    selection='best.pt: validation loss; best_detection.pt: F1 then count MAE; '
                              'best_planning.pt: valid rate then teacher-required coverage then ADE '
                              'on fixed validation subset; no test during training')
    protocol['learning_rate_schedule'] = dict(name=lr_schedule, decay_start_epoch=lr_decay_start_epoch,
        min_ratio=lr_min_ratio, unit='epoch', base_lr=lr,
        semantics='Set optimizer LR before each epoch; stateless schedule fixed by planned budget')
    if lr_schedule_unit == 'update':
        protocol['learning_rate_schedule'].update(unit='optimizer_update', updates_per_epoch=updates_per_epoch,
            total_updates=total_updates, flat_updates=flat_updates if lr_schedule == 'flat_cosine' else 0,
            semantics='One-based optimizer updates; first LR=base, final LR=base*min_ratio; '
                      'flat_cosine holds complete initial epochs; tail accumulation counts as one update')
    if resume:
        old = json.loads((output / 'protocol.json').read_text())
        if not init_grounding:
            protocol['init_grounding_sha256'] = old.get('init_grounding_sha256')
        if protocol != old:
            raise ValueError('Resume protocol/provenance fingerprint differs (data, source, or configuration)')
    torch.manual_seed(seed)
    random.seed(seed)
    grounder = RadarVLAGrounder(ModelConfig(**config['model'])).to(device)
    grounder.loss_weights = dict(config.get('loss_weights', {}))
    planner = make_planner(config).to(device) if stage == 'sft' else None
    if stage == 'sft' and not resume:
        if not init_grounding:
            raise ValueError('SFT requires --init-grounding from Stage 1')
        initial = torch.load(init_grounding, map_location='cpu', weights_only=True)
        initial_model = dict(initial['protocol']['config']['model'])
        initial_model.setdefault('use_detection', False)
        if (initial['protocol']['stage'] != 'grounding'
                or asdict(ModelConfig(**initial_model)) != config['model']
                or initial['protocol']['data'] != protocol['data']):
            raise ValueError('Grounding checkpoint model or data provenance differs')
        from .local_risk import LEGACY_RISK_SCHEMA
        old_risk_schema = initial['protocol']['config'].get('data', {}).get('risk_supervision_schema', LEGACY_RISK_SCHEMA)
        if old_risk_schema != config['data'].get('risk_supervision_schema', LEGACY_RISK_SCHEMA):
            raise ValueError('Grounding checkpoint Risk supervision schema differs')
        from .task_supervision import LEGACY_PROTOCOL
        if initial['protocol']['config'].get('data',{}).get('supervision_protocol',LEGACY_PROTOCOL) != config['data'].get('supervision_protocol',LEGACY_PROTOCOL):
            raise ValueError('Grounding checkpoint task supervision protocol differs')
        if initial['protocol']['config'].get('data',{}).get('agent_selection_policy') != config['data'].get('agent_selection_policy'):
            raise ValueError('Grounding checkpoint Agent selection policy differs')
        grounder.load_state_dict(initial['grounder'])
    modes_by_split, generation_indices = {}, []
    if planner:
        # Scan the whole supervised cohort before the first optimizer step;
        # an isolated valid sample must not conceal missing targets elsewhere.
        for split_name, dataset in (('train', train), ('val', val)):
            eligible = False
            modes_by_split[split_name] = []
            for i in range(len(dataset)):
                sample = collate_batch([dataset[i]])
                if planner.config.risk_source == 'oracle':
                    select_risk(sample['risk_target'], sample, 'oracle')
                supervision = create_supervision(sample, planner.tokenizer, planner.config)
                row = supervision['rows'][0]
                modes_by_split[split_name].append(row['mode'])
                usable = (row['mode'] is not None if dataset.supervision_protocol == 'taskwise_local_v1' else
                          row['mode'] is not None and row['ego'] is not None and bool(row['ego']['fitted']))
                if config['data'].get('strict_supervision', True) and dataset.supervision_protocol != 'taskwise_local_v1':
                    identifier = dataset.records[i]['sample_id']
                    if not usable:
                        raise ValueError(f'SFT sample {identifier} lacks a known mode or a fit-able ego trajectory')
                    if 'agent_supervision_mask' in sample and not bool(sample['agent_supervision_mask'].all()):
                        raise ValueError(f'SFT sample {identifier} has unknown agent annotation coverage')
                    if row['mode'] == 'long' and not all(
                        s['fitted'] and s['width_mask'] for s in row['road'].get('segments', [row['road']])):
                        raise ValueError(f'LONG SFT sample {identifier} needs observed map centerline and linked lane boundaries')
                    if row['mode'] == 'long' and any(not a['trajectory']['fitted'] for a in row['agents']):
                        raise ValueError(f'LONG SFT sample {identifier} needs observed future endpoints for every agent')
                if usable:
                    eligible = True
            if not eligible:
                raise ValueError('SFT train/val require a known bootstrap mode and observed ego future')
        from .training_quality import select_generation_indices
        generation_indices = select_generation_indices(modes_by_split['val'], validation_generation_samples, seed)
    if freeze_grounder:
        grounder.requires_grad_(False)
        grounder.eval()
    parameters = [p for p in list(grounder.parameters()) + (list(planner.parameters()) if planner else [])
                  if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    start, best, history = 1, float('inf'), []
    best_quality, best_quality_epoch = {}, {}
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
        best_quality = state.get('best_quality', {})
        best_quality_epoch = state.get('best_quality_epoch', {})
        if [row['epoch'] for row in history] != list(range(1, start)):
            raise ValueError('Checkpoint history is inconsistent')
        if rank == 0 and state['best_epoch'] == state['epoch']:
            save_checkpoint(output / 'best.pt', state)
        if distributed:
            dist.barrier()
        if not (output / 'best.pt').exists():
            raise ValueError('Previous best checkpoint is missing')
        for name, selected_epoch in best_quality_epoch.items():
            path = output / f'best_{name}.pt'
            if rank == 0 and selected_epoch == state['epoch']:
                save_checkpoint(path, state)
            if distributed:
                dist.barrier()
            if not path.exists():
                raise ValueError(f'Previous best {name} checkpoint is missing')
    else:
        if rank == 0:
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / 'protocol.json', protocol)
    if rank == 0 and planner:
        write_json(output / 'supervision_summary.json', dict(
            teacher_modes={split: {str(mode): modes.count(mode) for mode in ('short', 'long', None)}
                           for split, modes in modes_by_split.items()},
            generation_validation_ids=[val.records[i]['sample_id'] for i in generation_indices],
            note='Mode counts come from teachers, not collision-positive counts.'))
    if distributed:
        dist.barrier()
    system = TrainingSystem(grounder, planner, detach_risk_condition=detach_risk_condition)
    if distributed:
        system = torch.nn.parallel.DistributedDataParallel(
            system, device_ids=[torch.device(device).index] if device.startswith('cuda') else None,
            find_unused_parameters=True)
    best_epoch = state['best_epoch'] if resume else None
    try:
        final_epoch = min(epochs, stop_after_epoch or epochs)
        for epoch in range(start, final_epoch + 1):
            epoch_lr = (learning_rate_for_epoch(epoch, epochs, lr, lr_schedule, lr_decay_start_epoch, lr_min_ratio)
                        if lr_schedule_unit == 'epoch' else learning_rate_for_update(
                            (epoch-1)*updates_per_epoch+1, total_updates, lr, lr_schedule, flat_updates, lr_min_ratio))
            for group in optimizer.param_groups:
                group['lr'] = epoch_lr
            grounder.train(not freeze_grounder)
            if planner:
                planner.train()
            total, count = 0., 0
            mixture = gt_risk_mix * max(0., 1 - (epoch - 1) / (epochs - 1)) if epochs > 1 else 0.
            if rank == 0:
                write_json(output / 'status.json', dict(state='training', stage=stage, epoch=epoch,
                                                        epochs=epochs, learning_rate=epoch_lr,
                                                        world_size=world_size, pid=os.getpid()))
            began = time.monotonic()
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            sampler = DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True,
                                         seed=seed, drop_last=False) if distributed else None
            if sampler:
                sampler.set_epoch(epoch)
            loader = make_loader(train, batch_size, seed + epoch, shuffle=True, sampler=sampler, workers=workers)
            local_samples = len(sampler) if sampler is not None else len(train)
            update_rates = []
            optimizer.zero_grad(set_to_none=True)
            batches = tqdm(loader, total=len(loader), desc=f'{stage} train {epoch}/{epochs}',
                           unit='batch', disable=rank != 0)
            for step, batch in enumerate(batches):
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
                    if lr_schedule_unit == 'update':
                        update_index = (epoch-1)*updates_per_epoch + len(update_rates) + 1
                        rate = learning_rate_for_update(update_index, total_updates, lr,
                                                        lr_schedule, flat_updates, lr_min_ratio)
                        for group in optimizer.param_groups:
                            group['lr'] = rate
                        update_rates.append(rate)
                    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                total += loss.item() * n
                count += n
            if distributed:
                totals = torch.tensor([total, count], dtype=torch.float64, device=device)
                dist.all_reduce(totals)
                total, count = totals.tolist()
            validation_box = [validate(grounder, planner, val, batch_size, device, precision,
                                       progress_desc=f'{stage} val {epoch}/{epochs}')
                              if rank == 0 else None]
            if rank == 0 and planner is not None and generation_indices:
                validation_box[0]['free_generation'] = validate_generation(
                    grounder, planner, val, generation_indices, modes_by_split['val'],
                    batch_size, device, precision)
            if distributed:
                dist.broadcast_object_list(validation_box, src=0)
            validation = validation_box[0]
            row = dict(epoch=epoch, learning_rate=epoch_lr, train_loss=total / count, validation=validation,
                       gt_risk_mix=mixture, seconds=time.monotonic() - began)
            if lr_schedule_unit == 'update':
                assert len(update_rates) == updates_per_epoch
                row.update(learning_rate=update_rates[-1], learning_rate_start=update_rates[0],
                           optimizer_learning_rates=update_rates, optimizer_updates=epoch*updates_per_epoch)
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
            from .training_quality import quality_scores
            for name, score in quality_scores(validation).items():
                if name not in best_quality or tuple(score) < tuple(best_quality[name]):
                    best_quality[name], best_quality_epoch[name] = score, epoch
            rng = _rng_state(device)
            rank_rng = [None] * world_size
            if distributed:
                dist.all_gather_object(rank_rng, rng)
            else:
                rank_rng[0] = rng
            if rank == 0:
                state = dict(protocol=protocol, epoch=epoch, grounder=grounder.state_dict(),
                             planner=planner_state(planner), optimizer=optimizer.state_dict(),
                             rng=rng, rank_rng=rank_rng, best_loss=best, best_epoch=best_epoch, history=history,
                             best_quality=best_quality, best_quality_epoch=best_quality_epoch)
                save_checkpoint(output / 'last.pt', state)
                save_checkpoint(output / f'epoch_{epoch:03d}.pt', state)
                if best_epoch == epoch:
                    save_checkpoint(output / 'best.pt', state)
                for name, selected_epoch in best_quality_epoch.items():
                    if selected_epoch == epoch:
                        save_checkpoint(output / f'best_{name}.pt', state)
                (output / 'metrics.jsonl').write_text(''.join(json.dumps(x, allow_nan=False) + '\n' for x in history))
                completed_this_run = epoch - start + 1
                mean_epoch_seconds = sum(item['seconds'] for item in history[start-1:]) / completed_this_run
                print(json.dumps(dict(stage=stage, epoch=epoch, train_loss=total / count,
                                      val_loss=validation['loss'],
                                      eta_remaining_s=round(mean_epoch_seconds * (final_epoch - epoch), 1))),
                      flush=True)
            if distributed:
                dist.barrier()
        status = dict(state='complete' if history[-1]['epoch'] == epochs else 'paused',
                      stage=stage, epoch=history[-1]['epoch'], epochs=epochs,
                      best_epoch=best_epoch, best_validation_loss=best,
                      best_quality_epoch=best_quality_epoch,
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
    # Earlier v2 checkpoints predate the detector. Keep their exact architecture;
    # never attach a randomly initialized detector and report its counts.
    model_options = dict(config['model'])
    model_options.setdefault('use_detection', False)
    grounder = RadarVLAGrounder(ModelConfig(**model_options)).to(device)
    # Loading outside the campaign runner must preserve the trained objective.
    grounder.loss_weights = _validated_loss_weights(config)
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


def _override_risk_projection(planner, projection):
    if projection is None:
        return
    if projection not in ('none', 'isotonic'):
        raise ValueError('risk_projection must be none or isotonic')
    if planner is None:
        if projection != 'none':
            raise ValueError('Risk condition projection requires a planner')
        return
    if projection != 'none' and not planner.config.use_risk_token:
        raise ValueError('Risk condition projection requires a trained Risk token')
    planner.config.risk_projection = projection


def evaluate_checkpoint(checkpoint, manifest, output, split='test', batch_size=2, device='cpu', risk_source=None,
                        risk_projection=None):
    grounder, planner, state = load_models(checkpoint, device)
    _override_risk_projection(planner, risk_projection)
    if data_fingerprint(manifest) != state['protocol']['data']:
        raise ValueError('Evaluation data fingerprint differs from frozen training manifest')
    capacity = None if grounder.config.use_detection else grounder.config.max_agents
    dataset = RadarDataset(manifest, split, max_agents=capacity)
    from .task_supervision import stage_available
    dataset.records=[r for r in dataset.records if stage_available(r,state['protocol']['stage'])]
    if not dataset.records:raise ValueError('No eligible evaluation samples for this stage')
    validate_input_shapes(dataset, state['protocol']['config'])
    report = validate(grounder, planner, dataset, batch_size, device, state['protocol'].get('precision', 'float32'))
    if planner is not None:
        source = risk_source or planner.config.risk_source
        if (source == 'none') != (not planner.config.use_risk_token):
            raise ValueError('Risk-token presence must match the trained checkpoint; use the paired no-risk model')
        rows = prediction_rows(grounder, planner, dataset, state, batch_size, device, risk_source=source)
        from .planner import _teacher_mode
        from .training_quality import generation_diagnostics
        samples, modes = [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            modes.append(_teacher_mode(collate_batch([sample]), 0, planner.config))
            # Do not retain the full radar/geometry tensors for an entire test
            # split merely to aggregate a few trajectory diagnostics.
            samples.append({key: sample[key] for key in ('sample_id', 'ego_future', 'ego_future_mask',
                            'future_times_s', 'ego_velocity', 'ego_velocity_mask') if key in sample})
        report['free_generation'] = generation_diagnostics(rows, samples, modes, planner.config.short_horizon_s)
        from .evaluation import evaluate_plan, aggregate_plan_metrics
        records = {r['sample_id']: r for r in dataset.records}
        report['physical_planning'] = aggregate_plan_metrics([
            evaluate_plan(records[row['sample_id']], row['plan']) for row in rows])
        from .instruction_metrics import evaluate_instruction, aggregate_instruction_metrics
        report['instruction_following'] = aggregate_instruction_metrics([
            evaluate_instruction(records[row['sample_id']], row['plan']) for row in rows])
        report['risk_source'] = source
        report['risk_projection'] = planner.config.risk_projection if planner.config.use_risk_token else 'none'
        report['risk_metrics_semantics'] = 'raw Grounder Risk; actual planner condition is exported as conditioning_risk'
        report['amortized_inference_seconds'] = sum(r['amortized_inference_seconds'] for r in rows) / len(rows)
    report.update(split=split, checkpoint=str(Path(checkpoint).resolve()),
                  checkpoint_sha256=file_digest(checkpoint), epoch=state['epoch'],
                  evaluation='offline risk/physical metrics; no closed-loop safety claim')
    write_json(output, report)
    return report


@torch.no_grad()
def predict_checkpoint(checkpoint, manifest, output, split='test', batch_size=2,
                       device='cpu', max_new_tokens=1024, risk_projection=None):
    grounder, planner, state = load_models(checkpoint, device)
    _override_risk_projection(planner, risk_projection)
    dataset = RadarDataset(manifest, split, supervision=False)
    validate_input_shapes(dataset, state['protocol']['config'])
    rows = prediction_rows(grounder, planner, dataset, state, batch_size, device, max_new_tokens)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
    return rows


@torch.no_grad()
def prediction_rows(grounder, planner, dataset, state, batch_size, device, max_new_tokens=1024, risk_source='predicted'):
    from .execution import measure_plan, execution_payload
    validate_input_shapes(dataset, state['protocol']['config'])
    rows = []
    records = {record['sample_id']: record for record in dataset.records}
    precision = state['protocol'].get('precision', 'float32')
    reward_options = state['protocol'].get('rl_config', {}).get('reward', {})
    execution_options = {key: reward_options[key] for key in
        ('max_acceleration_mps2', 'max_jerk_mps3', 'initial_velocity_tolerance_mps') if key in reward_options}
    for batch in make_loader(dataset, batch_size, 0):
        batch = to_device(batch, device)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()
        with autocast_context(device, precision):
            prediction = grounder(batch)
        if grounder.config.use_detection:
            from .detection import decode_detections
            detections = decode_detections(prediction, grounder.config, batch['future_times_s'])
        else:
            detections = [dict(detection_status='unavailable_legacy', detections=None,
                               num_agents=None, num_vehicles=None, class_counts=None)] * len(batch['radar'])
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        detection_report = time.perf_counter()
        detection_seconds = detection_report - inference_start
        risk = select_risk(prediction['risk'], batch, risk_source)
        batch_rows = []
        for b, sample_id in enumerate(batch['sample_id']):
            measured = None
            waiting = 0.
            if planner is not None:
                # All detections in this batch are reported together. Plans
                # then use the same batch-one runtime and explicit queue clock
                # as response replay. Each record remains an independent frame.
                waiting = max(0., time.perf_counter() - detection_report)
                with autocast_context(device, precision):
                    measured = measure_plan(planner, prediction['radar_tokens'][b:b+1], risk[b:b+1],
                        [batch['instruction'][b]], max_new_tokens=max_new_tokens,
                        future_times_s=batch['future_times_s'][b].cpu().tolist())
            generation_seconds = 0. if measured is None else measured['generation_s']
            record = records[sample_id]
            adoption_delay = detection_seconds + waiting + generation_seconds
            row = dict(sample_id=sample_id, scene_id=batch['scene_id'][b], risk_source=risk_source,
                       risk=prediction['risk'][b].cpu().tolist(),
                       conditioning_risk=(planner.condition_risk(risk[b:b+1])[0].cpu().tolist()
                                          if planner and planner.config.use_risk_token else None),
                       risk_projection=planner.config.risk_projection if planner and planner.config.use_risk_token else 'none',
                       plan=None,
                       executed_plan=None, detection_seconds=detection_seconds,
                       generation_seconds=generation_seconds, waiting_seconds=waiting,
                       detection_report_timestamp_s=float(record['timestamp_s']) + detection_seconds,
                       plan_adoption_timestamp_s=(float(record['timestamp_s']) + adoption_delay
                                                  if measured is not None else None),
                       runtime_semantics='batched detection; sequential per-frame planning; CV/CTRV before adoption; no scene execution state')
            row.update(detections[b])
            if grounder.config.risk_horizon_schema != '1_2_3_v1':
                from .risk_contract import contract
                horizons,geometry_horizon=contract(grounder.config.risk_horizon_schema)
                row.update(risk_horizons_s=list(horizons),risk_geometry_horizon_s=geometry_horizon)
            if grounder.config.use_detection:
                selected = detections[b]['detections']
                row['object_probabilities'] = [d['confidence'] for d in selected]
                row['agent_state'] = [[*d['position_m'], d['radial_velocity_mps'], *d['velocity_mps']] for d in selected]
                row['agent_future'] = [d['future_xy'] for d in selected]
            else:
                row.update(object_probabilities=prediction['object_logits'][b].sigmoid().cpu().tolist(),
                           agent_state=prediction['agent_state'][b].cpu().tolist(),
                           agent_future=prediction['agent_future'][b].cpu().tolist())
            if measured is not None:
                row['plan'] = measured['plan']
                row['generated_tokens'] = len(measured['ids'])
                row['executed_plan'] = execution_payload(record, measured['plan'], adoption_delay, **execution_options)
            batch_rows.append(row)
        batch_seconds = time.perf_counter() - inference_start
        for row in batch_rows:
            row['amortized_inference_seconds'] = batch_seconds / len(batch_rows)
        rows.extend(batch_rows)
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
