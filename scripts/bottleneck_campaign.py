"""Resolve the fixed bottleneck comparison into the existing training entry point.

This module deliberately uses only the standard library. Planning a command must
not import torch, load a model, or require the data/checkpoints to exist.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ARMS = ('baseline', 'assignment', 'risk', 'action', 'combined')
STAGES = ('grounding', 'risk-readout', 'sft')
_ENV_FIELDS = {
    'epochs': 'EPOCHS', 'batch_size': 'BATCH_SIZE',
    'accumulation_steps': 'ACCUMULATION_STEPS', 'lr': 'LR',
    'lr_schedule': 'LR_SCHEDULE', 'lr_schedule_unit': 'LR_SCHEDULE_UNIT',
    'lr_decay_start_epoch': 'LR_DECAY_START_EPOCH', 'lr_min_ratio': 'LR_MIN_RATIO',
    'seed': 'SEED', 'workers': 'WORKERS', 'precision': 'PRECISION',
    'validation_generation_samples': 'VALIDATION_GENERATION_SAMPLES',
}


def add_arguments(parser):
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--gpus', help='Physical GPU indices; defaults to the profile (eight L40S GPUs)')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--init-grounding', type=Path)
    parser.add_argument('--profile', type=Path, help='Campaign JSON; defaults to configs/bottleneck_8xl40s.json')


def _relative(base, value):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _gpu_ids(raw):
    if not isinstance(raw, str):
        raise ValueError('profile gpus must be a comma-separated string')
    ids = [item.strip() for item in raw.split(',')]
    if (not 1 <= len(ids) <= 8 or len(set(ids)) != len(ids) or
            any(not item.isascii() or not item.isdigit() or str(int(item)) != item for item in ids)):
        raise ValueError('--gpus requires 1–8 distinct nonnegative GPU indices, e.g. 1,3,6')
    return ids


def _validate_stage(stage, settings):
    if not isinstance(settings, dict):
        raise ValueError(f'profile stage {stage} must be an object')
    missing = set(_ENV_FIELDS) - set(settings)
    unknown = set(settings) - set(_ENV_FIELDS) - {'freeze_grounder', 'risk_readout_only'}
    if missing or unknown:
        raise ValueError(f'profile stage {stage}: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}')
    for name in ('epochs', 'batch_size', 'accumulation_steps', 'lr_decay_start_epoch'):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    for name in ('workers', 'seed', 'validation_generation_samples'):
        if type(settings[name]) is not int or settings[name] < 0:
            raise ValueError(f'{name} must be a nonnegative integer')
    for name in ('lr', 'lr_min_ratio'):
        number = settings[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'{name} must be a finite number')
    if settings['lr'] <= 0 or not 0 <= settings['lr_min_ratio'] <= 1:
        raise ValueError('lr must be positive and lr_min_ratio must be in [0,1]')
    for name, choices in {'lr_schedule': ('constant', 'cosine', 'flat_cosine'),
                          'lr_schedule_unit': ('epoch', 'update'),
                          'precision': ('float32', 'bfloat16')}.items():
        if settings[name] not in choices:
            raise ValueError(f'unsupported {name}: {settings[name]}')
    for name in ('freeze_grounder', 'risk_readout_only'):
        if name in settings and type(settings[name]) is not bool:
            raise ValueError(f'{name} must be boolean')
    if settings.get('risk_readout_only', False) != (stage == 'risk-readout'):
        raise ValueError('risk_readout_only must be true exactly for risk-readout stage')
    if settings.get('freeze_grounder', False) and stage != 'sft':
        raise ValueError('freeze_grounder applies only to sft')


def _source_checkpoint(profile, campaign_root, arm, stage):
    try:
        epochs = profile['stages'][stage]['epochs']
    except (KeyError, TypeError) as exc:
        raise ValueError(f'profile must declare source-stage epochs for {stage}') from exc
    if type(epochs) is not int or epochs < 1:
        raise ValueError(f'source-stage {stage} epochs must be a positive integer')
    return campaign_root / arm / stage / f'epoch_{epochs:03d}.pt'


def run_campaign(args, root, data_root, run_root):
    profile_path = args.profile.resolve() if args.profile else root / 'configs/bottleneck_8xl40s.json'
    profile_bytes = profile_path.read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get('schema') != 'radar_vla_bottleneck_campaign_v1':
        raise ValueError('unsupported bottleneck campaign profile schema')
    if args.stage == 'risk-readout' and args.arm not in ('baseline', 'risk', 'combined'):
        raise ValueError('risk-readout supports only baseline, risk, or combined comparisons')
    try:
        settings = profile['stages'][args.stage]
        config = _relative(root, profile['arms'][args.arm]['config'])
        campaign_root = _relative(run_root, profile['output_root'])
        manifest = args.manifest.resolve() if args.manifest else _relative(data_root, profile['manifest'])
        gpu_ids = _gpu_ids(args.gpus if args.gpus is not None else profile['gpus'])
    except (KeyError, TypeError) as exc:
        raise ValueError(f'incomplete campaign profile: {exc}') from exc
    _validate_stage(args.stage, settings)
    output = args.output.resolve() if args.output else campaign_root / args.arm / args.stage
    init = args.init_grounding.resolve() if args.init_grounding else None
    if init is None and args.stage == 'risk-readout':
        source_arm = 'assignment' if args.arm == 'combined' else 'baseline'
        init = _source_checkpoint(profile, campaign_root, source_arm, 'grounding')
    if init is None and args.stage == 'sft':
        if args.arm in ('risk', 'combined'):
            init = _source_checkpoint(profile, campaign_root, args.arm, 'risk-readout')
        else:
            source_arm = 'assignment' if args.arm == 'assignment' else 'baseline'
            init = _source_checkpoint(profile, campaign_root, source_arm, 'grounding')
    # The ordinary CLI remains the sole data validation and stage launch gate.
    # sys.executable runs that stdlib-only wrapper; the configured model Python
    # is invoked by launch_l40s.sh only after preflight (never during --dry-run).
    command = [sys.executable, str(root / 'scripts/radarvla_cli.py'),
               '--project-root', str(root), '--paths', str(args.paths.resolve()),
               'train-sft' if args.stage == 'sft' else 'train-grounding',
               '--gpus', ','.join(gpu_ids), '--manifest', str(manifest),
               '--config', str(config), '--output', str(output)]
    if init is not None:
        command += ['--init-grounding', str(init)]
    if settings.get('risk_readout_only', False):
        command.append('--risk-readout-only')
    if settings.get('freeze_grounder', False):
        command.append('--freeze-grounder')
    if args.dry_run:
        command.append('--dry-run')
    if args.resume:
        command.append('--resume')
    env = os.environ.copy()
    env.update({env_name: str(settings[name]) for name, env_name in _ENV_FIELDS.items()})
    env.update(RISK_READOUT_ONLY=str(int(settings.get('risk_readout_only', False))),
               FREEZE_GROUNDER=str(int(settings.get('freeze_grounder', False))))
    env.pop('STOP_AFTER_EPOCH', None)
    per_rank = settings['batch_size'] * settings['accumulation_steps']
    metadata = {
        'schema': profile['schema'], 'profile': str(profile_path),
        'profile_sha256': hashlib.sha256(profile_bytes).hexdigest(),
        'arm': args.arm, 'stage': args.stage, 'config': str(config),
        'manifest': str(manifest), 'output': str(output),
        'init_grounding': str(init) if init else None, 'resume': args.resume,
        'gpus': gpu_ids, 'ranks': len(gpu_ids),
        'effective_global_batch': len(gpu_ids) * per_rank,
        'reference_global_batch': 8 * per_rank, 'settings': settings,
        'comparison_note': 'Shared eight-GPU baseline; not strictly equivalent to historical single-GPU runs.',
    }
    print('CAMPAIGN ' + json.dumps(metadata, sort_keys=True), flush=True)
    if len(gpu_ids) != 8:
        print(f'GPU count changed: effective global batch {len(gpu_ids) * per_rank} '
              f'(eight-GPU reference {8 * per_rank}); this changes the experiment.', flush=True)
    return subprocess.call(command, cwd=root, env=env)
