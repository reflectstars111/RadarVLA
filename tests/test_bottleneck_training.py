"""Native training honors experimental assignment and frozen risk readout."""
import copy

import pytest
import torch
from test_radar_vla_hf import tiny_local_qwen

from radar_vla import pipeline
from radar_vla.synthetic import write_synthetic_dataset


@pytest.mark.parametrize('arm', ['baseline', 'assignment', 'risk', 'action', 'combined'])
def test_server_arm_configs_resolve_all_model_interfaces(arm):
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    config = pipeline.resolve_config(json.loads((root / 'configs/bottleneck_v1' / f'{arm}.json').read_text()))
    assert (config['model']['risk_input'] == 'predicted_geometry_v1') == (arm in ('risk', 'combined'))
    assert (config['planner']['action_representation'] == 'parallel_acceleration_v1') == (arm in ('action', 'combined'))


def fixture(tmp_path):
    torch.set_num_threads(1)
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
        frames_per_scene=2, range_bins=8, azimuth_bins=8)
    config = dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2),
                  data=dict(radar_shape=[2, 8, 8], history_frames=3),
                  language_model=dict(backend='tiny'))
    return manifest, config


def test_assignment_warmup_is_epoch_bound_and_records_actual_order(tmp_path, monkeypatch):
    manifest, config = fixture(tmp_path)
    config['model'].update(detection_assignment_policy='local_quality_v1',
                           detection_assignment_warmup_epochs=2)
    original, calls = pipeline.loss_grounding, []
    def observed(*args, **kwargs):
        calls.append(kwargs['assignment_policy'])
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'loss_grounding', observed)
    pipeline.run_training(manifest, tmp_path / 'run', epochs=3, batch_size=2, config=config)
    # One train batch and one val batch per epoch; validation always uses the
    # configured quality policy, training changes only after the two warmups.
    assert calls == ['anchor_v1', 'local_quality_v1', 'anchor_v1', 'local_quality_v1',
                     'local_quality_v1', 'local_quality_v1']
    state = torch.load(tmp_path / 'run/last.pt', weights_only=True)
    for row in state['history']:
        order = row['training_order_by_rank'][0]
        assert order['samples'] == 2 and len(order['sha256']) == 64
    assert state['history'][2]['assignment_diagnostics_by_rank'][0]['local_quality_frames'] == 2


def test_frozen_risk_readout_explicitly_transfers_baseline_and_keeps_detector(tmp_path):
    manifest, config = fixture(tmp_path)
    pipeline.run_training(manifest, tmp_path / 'base', epochs=1, batch_size=2, config=config)
    initial = torch.load(tmp_path / 'base/last.pt', weights_only=True)
    changed = copy.deepcopy(config)
    changed['model']['risk_input'] = 'predicted_geometry_v1'
    pipeline.run_training(manifest, tmp_path / 'risk', epochs=1, batch_size=2, config=changed,
                          init_grounding=tmp_path / 'base/last.pt', risk_readout_only=True)
    state = torch.load(tmp_path / 'risk/last.pt', weights_only=True)
    assert state['protocol']['risk_readout_only']
    for name, tensor in initial['grounder'].items():
        if not name.startswith(('risk_head.', 'risk_attention.', 'risk_query')):
            torch.testing.assert_close(state['grounder'][name], tensor, rtol=0, atol=0)
    assert any(not torch.equal(state['grounder'][name], tensor)
               for name, tensor in initial['grounder'].items() if name.startswith('risk_head.'))
    model, _, loaded = pipeline.load_models(tmp_path / 'risk/last.pt')
    assert model.config.risk_input == 'predicted_geometry_v1'
    assert loaded['protocol']['init_grounding_sha256'] == pipeline.file_digest(tmp_path / 'base/last.pt')


def test_readout_migration_rejects_missing_weights_in_same_schema():
    from radar_vla.model import ModelConfig, RadarVLAGrounder
    config = pipeline.resolve_config(dict(model=dict(hidden_dim=8, num_heads=2,
                                                    risk_input='predicted_geometry_v1')))
    source = RadarVLAGrounder(ModelConfig(**config['model'])).state_dict()
    del source['risk_fusion.candidate_encoder.0.weight']
    initial = dict(protocol=dict(stage='grounding', config=config, data={}), grounder=source)
    with pytest.raises(ValueError, match='[Ss]ame|[Mm]issing|[Cc]omplete'):
        pipeline._initialize_risk_readout(RadarVLAGrounder(ModelConfig(**config['model'])),
                                        initial, config, {})


@pytest.mark.parametrize('options', [dict(risk_readout_only=True),
    dict(stage='sft', risk_readout_only=True), dict(risk_readout_only='yes')])
def test_readout_requires_explicit_grounding_initialization_before_loading_data(tmp_path, options):
    with pytest.raises(ValueError, match='readout'):
        pipeline.run_training(tmp_path / 'missing', tmp_path / 'never', **options)
    assert not (tmp_path / 'never').exists()


def test_parallel_action_runs_native_sft_reload_and_prediction(tmp_path, monkeypatch):
    from radar_vla.data import RadarDataset
    manifest, config = fixture(tmp_path)
    pipeline.run_training(manifest, tmp_path / 'ground', epochs=1, batch_size=2, config=config)
    config['planner'] = dict(action_representation='parallel_acceleration_v1', num_layers=1)
    pipeline.run_training(manifest, tmp_path / 'action', stage='sft', epochs=1,
        batch_size=2, config=config, init_grounding=tmp_path / 'ground/last.pt',
        freeze_grounder=True, validation_generation_samples=2)
    grounder, planner, state = pipeline.load_models(tmp_path / 'action/last.pt')
    def forbidden(*args, **kwargs):
        raise AssertionError('Action readiness waited for autoregressive description')
    monkeypatch.setattr(planner, 'generate', forbidden)
    data = RadarDataset(manifest, 'val', supervision=False)
    rows = pipeline.prediction_rows(grounder, planner, data, state, batch_size=2, device='cpu')
    assert len(rows) == 2
    for row in rows:
        assert row['plan']['schema'] == 'radar_vla_parallel_acceleration_v1'
        assert row['plan']['valid'] and row['plan']['ego_complete']
        assert row['generated_tokens'] == 0
        assert row['generation_seconds'] > 0
        assert row['executed_plan'] is not None
    assert state['history'][0]['validation']['free_generation']['schema_valid_rate'] == 1.
    assert state['history'][0]['validation']['action_components']['description_token'] > 0
    calls = []
    original_action = planner.plan_action
    def action(*args, **kwargs):
        calls.append('action')
        return original_action(*args, **kwargs)
    def description(*args, **kwargs):
        calls.append('description')
        return [[]]
    monkeypatch.setattr(planner, 'plan_action', action)
    monkeypatch.setattr(planner, 'generate', description)
    monkeypatch.setattr(planner, 'decode', lambda *a, **k: dict(valid=True, mode='long',
        road={'segments': []}, agent_states=[], ego_trajectory=[[999.,999.]]))
    described = pipeline.prediction_rows(grounder, planner, data, state, 2, 'cpu', include_description=True)
    assert calls == ['action', 'action', 'description', 'description']
    assert described[0]['description']['road'] == {'segments': []}
    assert 'ego_trajectory' not in described[0]['description']
    assert described[0]['description_seconds'] >= 0
    with pytest.raises(ValueError, match='Oracle'):
        pipeline.prediction_rows(grounder, planner, data, state, 2, 'cpu', risk_source='oracle')


def test_parallel_training_rejects_oracle_mixing_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match='[Pp]arallel|[Oo]racle|[Pp]redicted'):
        pipeline.run_training(tmp_path / 'missing', tmp_path / 'never', stage='sft',
            config={'planner':{'action_representation':'parallel_acceleration_v1'}}, gt_risk_mix=.5)
    assert not (tmp_path / 'never').exists()


def test_combined_hf_lora_two_rank_training_and_resume(tmp_path, tiny_local_qwen):
    """Real DDP/backbone: shared action + teacher branches, frozen fused Risk."""
    import json
    import os
    import subprocess
    import sys
    manifest, config = fixture(tmp_path)
    config['model'].update(detection_assignment_policy='local_quality_v1',
                           detection_assignment_warmup_epochs=0)
    pipeline.run_training(manifest, tmp_path / 'ground', epochs=1, batch_size=2, config=config)
    config['model']['risk_input'] = 'predicted_geometry_v1'
    pipeline.run_training(manifest, tmp_path / 'risk', epochs=1, batch_size=2, config=config,
        init_grounding=tmp_path / 'ground/last.pt', risk_readout_only=True)
    config['planner'] = dict(action_representation='parallel_acceleration_v1', num_layers=1)
    config['language_model'] = dict(backend='hf', model_path=str(tiny_local_qwen),
        lora=True, dtype='bfloat16', gradient_checkpointing=True, generation_use_cache=True)
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    runner = tmp_path / 'ddp.py'
    runner.write_text('import json, sys, torch\n'
        'from radar_vla.pipeline import run_training\n'
        'torch.set_num_threads(1)\n'
        'run_training(sys.argv[1], sys.argv[2], stage="sft", epochs=2, batch_size=1, '
        'device="cpu", config=json.load(open(sys.argv[3])), init_grounding=sys.argv[4], '
        'freeze_grounder=True, precision="bfloat16", validation_generation_samples=2, '
        'resume=sys.argv[5]=="resume", stop_after_epoch=1 if sys.argv[5]=="fresh" else None)\n'
        'torch.distributed.destroy_process_group()\n')
    env = dict(os.environ, OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TOKENIZERS_PARALLELISM='false')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(runner), str(manifest), str(tmp_path / 'sft'), str(config_path), str(tmp_path / 'risk/last.pt')]
    for mode in ('fresh', 'resume'):
        result = subprocess.run(command + [mode], env=env, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    state = torch.load(tmp_path / 'sft/last.pt', weights_only=True)
    assert state['epoch'] == 2
    assert len(state['history'][1]['training_order_by_rank']) == 2
    assert state['history'][1]['validation']['free_generation']['schema_valid_rate'] == 1.
    initial = torch.load(tmp_path / 'risk/last.pt', weights_only=True)
    for name, value in initial['grounder'].items():
        torch.testing.assert_close(state['grounder'][name], value, rtol=0, atol=0)
