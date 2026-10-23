"""Campaign commands are reproducible and dry runs never touch the model runtime."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / 'scripts/radarvla_cli.py'
PROFILE = ROOT / 'configs/bottleneck_8xl40s.json'


@pytest.fixture
def local_paths(tmp_path):
    marker = tmp_path / 'runtime-invoked'
    sentinel = tmp_path / 'sentinel-python'
    sentinel.write_text('#!/bin/sh\ntouch ' + shlex.quote(str(marker)) + '\nexit 73\n')
    sentinel.chmod(0o755)
    paths = tmp_path / 'paths.json'
    paths.write_text(json.dumps({
        'python': str(sentinel), 'data_root': str(tmp_path / 'data'),
        'collection_manifest': str(tmp_path / 'collection.json'),
        'model_config': str(ROOT / 'configs/l40s_qwen25_3b.json'),
        'qwen_model': str(tmp_path / 'qwen'), 'run_root': str(tmp_path / 'runs')}))
    return paths, marker


def run_cli(paths, *args, env=None):
    return subprocess.run([sys.executable, str(CLI), '--project-root', str(ROOT),
                           '--paths', str(paths), *args], cwd=paths.parent,
                          env=env, text=True, capture_output=True)


def native_args(result):
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line for line in result.stdout.splitlines() if line.startswith('cd '))
    tokens = shlex.split(line)
    return tokens[tokens.index('&&') + 1:]


def value(args, name):
    return args[args.index(name) + 1]


def metadata(result):
    return json.loads(next(line.removeprefix('CAMPAIGN ') for line in result.stdout.splitlines()
                           if line.startswith('CAMPAIGN ')))


@pytest.mark.parametrize('arm,stage,checkpoint', [
    ('baseline', 'grounding', None),
    ('assignment', 'grounding', None),
    ('baseline', 'risk-readout', 'baseline/grounding/epoch_020.pt'),
    ('risk', 'risk-readout', 'baseline/grounding/epoch_020.pt'),
    ('combined', 'risk-readout', 'assignment/grounding/epoch_020.pt'),
    ('baseline', 'sft', 'baseline/grounding/epoch_020.pt'),
    ('action', 'sft', 'baseline/grounding/epoch_020.pt'),
    ('assignment', 'sft', 'assignment/grounding/epoch_020.pt'),
    ('risk', 'sft', 'risk/risk-readout/epoch_010.pt'),
    ('combined', 'sft', 'combined/risk-readout/epoch_010.pt'),
])
def test_campaign_fixed_protocol_and_checkpoint_chain(local_paths, arm, stage, checkpoint):
    paths, marker = local_paths
    result = run_cli(paths, 'bottleneck', '--arm', arm, '--stage', stage, '--dry-run')
    args = native_args(result)
    assert '--nproc_per_node=8' in args
    assert value(args, '--config') == str(ROOT / f'configs/bottleneck_v1/{arm}.json')
    assert value(args, '--stage') == ('sft' if stage == 'sft' else 'grounding')
    assert value(args, '--manifest') == str(paths.parent / 'data/prepared/taskwise_moving_krsv2_0p5_1_2/manifest.jsonl')
    assert value(args, '--output') == str(paths.parent / f'runs/bottleneck_20261023/{arm}/{stage}')
    assert value(args, '--epochs') == {'grounding': '20', 'risk-readout': '10', 'sft': '5'}[stage]
    assert value(args, '--batch-size') == '1'
    assert value(args, '--accumulation-steps') == ('4' if stage == 'sft' else '1')
    assert value(args, '--lr-schedule-unit') == 'update'
    assert value(args, '--lr-schedule') == 'flat_cosine'
    assert value(args, '--validation-generation-samples') == '32'
    assert ('--freeze-grounder' in args) == (stage == 'sft')
    assert ('--risk-readout-only' in args) == (stage == 'risk-readout')
    if checkpoint:
        assert value(args, '--init-grounding') == str(paths.parent / 'runs/bottleneck_20261023' / checkpoint)
    else:
        assert '--init-grounding' not in args
    info = metadata(result)
    assert info['effective_global_batch'] == (32 if stage == 'sft' else 8)
    assert info['profile_sha256'] == hashlib.sha256(PROFILE.read_bytes()).hexdigest()
    assert 'historical single-GPU' in info['comparison_note']
    assert not marker.exists()
    assert not (paths.parent / 'runs').exists()


def test_campaign_overrides_and_resume_are_explicit(local_paths):
    paths, marker = local_paths
    profile = json.loads(PROFILE.read_text())
    profile['stages']['sft']['validation_generation_samples'] = 7
    profile_path = paths.parent / 'custom-profile.json'
    profile_path.write_text(json.dumps(profile))
    result = run_cli(paths, 'bottleneck', '--arm', 'combined', '--stage', 'sft', '--dry-run',
                     '--profile', str(profile_path), '--gpus', '1,3', '--manifest', 'custom.jsonl',
                     '--output', 'custom-run', '--init-grounding', 'custom.pt', '--resume',
                     env=dict(os.environ, STOP_AFTER_EPOCH='1', LR='900', RISK_READOUT_ONLY='1'))
    args = native_args(result)
    assert '--nproc_per_node=2' in args
    assert '--resume' in args and '--init-grounding' not in args
    assert '--stop-after-epoch' not in args
    assert '--risk-readout-only' not in args
    assert value(args, '--manifest') == str(paths.parent / 'custom.jsonl')
    assert value(args, '--output') == str(paths.parent / 'custom-run')
    assert float(value(args, '--lr')) == 5e-5
    assert value(args, '--validation-generation-samples') == '7'
    info = metadata(result)
    assert info['effective_global_batch'] == 8
    assert info['reference_global_batch'] == 32
    assert info['init_grounding'] == str(paths.parent / 'custom.pt')
    assert 'changed' in result.stdout.lower()
    assert not marker.exists()


@pytest.mark.parametrize('arm,stage,source', [
    ('risk', 'risk-readout', 'baseline/grounding/epoch_007.pt'),
    ('combined', 'risk-readout', 'assignment/grounding/epoch_007.pt'),
    ('baseline', 'sft', 'baseline/grounding/epoch_007.pt'),
    ('action', 'sft', 'baseline/grounding/epoch_007.pt'),
    ('assignment', 'sft', 'assignment/grounding/epoch_007.pt'),
    ('risk', 'sft', 'risk/risk-readout/epoch_003.pt'),
    ('combined', 'sft', 'combined/risk-readout/epoch_003.pt'),
])
def test_custom_epoch_budgets_determine_default_checkpoint_chain(local_paths, arm, stage, source):
    paths, marker = local_paths
    profile = json.loads(PROFILE.read_text())
    profile['stages']['grounding']['epochs'] = 7
    profile['stages']['risk-readout']['epochs'] = 3
    profile_path = paths.parent / 'short-profile.json'
    profile_path.write_text(json.dumps(profile))
    result = run_cli(paths, 'bottleneck', '--arm', arm, '--stage', stage,
                     '--profile', str(profile_path), '--dry-run')
    args = native_args(result)
    assert value(args, '--init-grounding') == str(paths.parent / 'runs/bottleneck_20261023' / source)
    assert not marker.exists()


@pytest.mark.parametrize('extra,expected', [
    (['--arm', 'assignment', '--stage', 'risk-readout'], 'risk-readout'),
    (['--arm', 'action', '--stage', 'risk-readout'], 'risk-readout'),
    (['--arm', 'baseline', '--stage', 'grounding', '--gpus', '1,1'], 'distinct'),
])
def test_campaign_rejects_invalid_comparisons(local_paths, extra, expected):
    paths, marker = local_paths
    result = run_cli(paths, 'bottleneck', *extra, '--dry-run')
    assert result.returncode == 2
    assert expected in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize('status,digest', [('FAIL', 'correct'), ('PASS', 'wrong')])
def test_real_campaign_requires_pass_and_matching_hash_before_runtime(local_paths, status, digest):
    paths, marker = local_paths
    manifest = paths.parent / 'manifest.jsonl'
    manifest.write_text('{}\n')
    (paths.parent / 'validation.json').write_text(json.dumps({
        'status': status,
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if digest == 'correct' else 'wrong'}))
    result = run_cli(paths, 'bottleneck', '--arm', 'baseline', '--stage', 'grounding',
                     '--manifest', str(manifest))
    assert result.returncode == 2
    assert 'PASS validation' in result.stderr
    assert not marker.exists()


def test_readout_requires_existing_grounding_checkpoint_before_runtime(local_paths):
    paths, marker = local_paths
    manifest = paths.parent / 'manifest.jsonl'
    manifest.write_text('{}\n')
    (paths.parent / 'validation.json').write_text(json.dumps({
        'status': 'PASS', 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    result = run_cli(paths, 'train-grounding', '--risk-readout-only', '--manifest', str(manifest),
                     '--gpus', '0', '--init-grounding', str(paths.parent / 'missing.pt'))
    assert result.returncode == 2
    assert 'checkpoint' in result.stderr
    assert not marker.exists()


def test_pass_report_reaches_runtime_gate_without_launching_training(local_paths):
    paths, marker = local_paths
    manifest = paths.parent / 'manifest.jsonl'
    manifest.write_text('{}\n')
    (paths.parent / 'validation.json').write_text(json.dumps({
        'status': 'PASS', 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}))
    result = run_cli(paths, 'bottleneck', '--arm', 'baseline', '--stage', 'grounding',
                     '--manifest', str(manifest))
    # The configured sentinel is only reached for the actual hardware preflight;
    # it exits before torchrun. This also proves valid reports are not all blocked.
    assert result.returncode == 73, result.stdout + result.stderr
    assert marker.exists()
    assert not (paths.parent / 'runs').exists()


def test_public_shell_entry_dry_run(local_paths):
    paths, _ = local_paths
    config = json.loads(paths.read_text())
    config['python'] = sys.executable
    paths.write_text(json.dumps(config))
    result = subprocess.run(['bash', str(ROOT / 'scripts/radarvla.sh'), 'bottleneck',
                             '--arm', 'assignment', '--stage', 'grounding', '--dry-run'],
                            cwd=paths.parent, env=dict(os.environ, RADARVLA_PATHS_CONFIG=str(paths)),
                            text=True, capture_output=True)
    assert '--nproc_per_node=8' in native_args(result)


def test_real_launcher_checks_available_cuda_before_training(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('{}\n')
    env = dict(os.environ, MANIFEST=str(manifest), OUTPUT=str(tmp_path / 'run'), STAGE='grounding',
               CUDA_VISIBLE_DEVICES='999', NPROC_PER_NODE='1', RADAR_VLA_PYTHON=sys.executable,
               RADAR_VLA_CONFIG=str(ROOT / 'configs/l40s_qwen25_3b.json'),
               RISK_READOUT_ONLY='0', FREEZE_GROUNDER='0', RESUME='0')
    result = subprocess.run(['bash', str(ROOT / 'launch_l40s.sh')], env=env,
                            text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode != 0
    assert 'visible CUDA' in result.stderr
    assert not (tmp_path / 'run').exists()
