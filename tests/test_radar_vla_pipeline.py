"""Pipeline contracts: reproducible resume, provenance, and held-out metrics."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


def test_probability_metrics_handle_ties_and_undefined_classes():
    from radar_vla.metrics import binary_metrics

    actual = binary_metrics([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1])
    assert actual['auroc'] == pytest.approx(0.5)
    assert actual['average_precision'] == pytest.approx(0.5)
    assert actual['brier'] == pytest.approx(0.25)
    assert binary_metrics([0.1, 0.3], [0, 0])['auroc'] is None
    assert binary_metrics([0.1, 0.3], [0, 0])['average_precision'] is None
    assert binary_metrics([], [])['brier'] is None


def test_short_plan_is_not_mistaken_for_full_horizon_fde():
    from radar_vla.metrics import planning_metrics

    truth = [dict(sample_id='one', ego_future=torch.tensor([[0., 0.], [3., 4.], [6., 8.]]),
                  ego_future_mask=torch.tensor([True, True, True]),
                  future_times_s=torch.tensor([1., 2., 3.]))]
    rows = [dict(sample_id='one', generated_tokens=12, plan=dict(valid=True, mode='short',
                 ego_trajectory=[[3., 4.], None], ego_times_s=[1., 2.]))]
    result = planning_metrics(rows, truth)
    assert result['ego_ade_m'] == 5.
    assert result['ego_final_horizon_fde_m'] is None
    assert result['ego_point_coverage'] == pytest.approx(1 / 3)
    assert result['schema_valid_rate'] == 1.


def test_metrics_exclude_missing_components_instead_of_emitting_nan():
    from radar_vla.metrics import MetricAccumulator

    prediction = dict(risk=torch.zeros(1, 5), agent_state=torch.zeros(1, 1, 5),
                      agent_future=torch.zeros(1, 1, 2, 2))
    target = dict(risk_target=torch.full((1, 5), float('nan')), risk_mask=torch.zeros(1, 5, dtype=torch.bool),
                  agent_state=torch.tensor([[[1., 0., float('nan'), float('nan'), float('nan')]]]),
                  agent_mask=torch.ones(1, 1, dtype=torch.bool),
                  agent_state_mask=torch.tensor([[[True, True, False, False, False]]]),
                  agent_future=torch.full((1, 1, 2, 2), float('nan')),
                  agent_future_mask=torch.zeros(1, 1, 2, dtype=torch.bool))
    metrics = MetricAccumulator()
    metrics.update(prediction, target)
    result = metrics.compute()
    assert result['position_mae'] == pytest.approx(1.)
    assert result['radial_mae'] is None
    assert result['velocity_rmse_mps'] is None
    json.dumps(result, allow_nan=False)


def test_native_radar_shape_and_history_are_checked(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import run_training

    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    cfg = {'data': {'radar_shape': [2, 256, 107], 'history_frames': 4}}
    with pytest.raises(ValueError, match='radar_shape|history_frames'):
        run_training(manifest, tmp_path / 'run', config=cfg, epochs=1)


def test_cpu_torchrun_two_ranks_save_one_complete_checkpoint(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset

    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=2)
    cfg = tmp_path / 'config.json'
    cfg.write_text(json.dumps({'model': {'hidden_dim': 16, 'num_heads': 2, 'num_queries': 4,
                                         'max_agents': 4, 'horizon_steps': 6},
                                'planner': {'hidden_dim': 16, 'num_heads': 2, 'num_layers': 1}}))
    output = tmp_path / 'distributed'
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2',
               '-m', 'radar_vla', 'train', '--manifest', str(manifest), '--output', str(output),
               '--config', str(cfg), '--epochs', '1', '--device', 'cpu', '--batch-size', '1',
               '--accumulation-steps', '2', '--precision', 'float32']
    result = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1'),
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, result.stdout[-5000:]
    checkpoint = torch.load(output / 'last.pt', weights_only=True)
    assert checkpoint['epoch'] == 1
    assert checkpoint['protocol']['world_size'] == 2
    assert len(checkpoint['rank_rng']) == 2
    assert json.loads((output / 'status.json').read_text())['state'] == 'complete'


def test_training_resume_reproduces_uninterrupted_run_and_rejects_changed_data(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import run_training

    torch.set_num_threads(1)
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1,
                                       frames_per_scene=2, seed=4)
    cfg = {'model': {'hidden_dim': 16, 'num_heads': 2, 'num_queries': 4,
                     'max_agents': 4, 'horizon_steps': 6},
           'planner': {'hidden_dim': 16, 'num_heads': 2, 'num_layers': 1,
                       'max_agents': 4, 'horizon_steps': 6}}
    common = dict(manifest=manifest, stage='grounding', epochs=2, batch_size=2,
                  seed=7, config=cfg, device='cpu')
    run_training(output=tmp_path / 'full', **common)
    run_training(output=tmp_path / 'resumed', stop_after_epoch=1, **common)
    run_training(output=tmp_path / 'resumed', resume=True, **common)
    full = torch.load(tmp_path / 'full/last.pt', weights_only=True)
    resumed = torch.load(tmp_path / 'resumed/last.pt', weights_only=True)
    assert resumed['epoch'] == 2
    for key, value in full['grounder'].items():
        torch.testing.assert_close(resumed['grounder'][key], value, rtol=0, atol=0)
    assert len((tmp_path / 'resumed/metrics.jsonl').read_text().splitlines()) == 2
    assert not (tmp_path / 'resumed/test_metrics.json').exists()
    first = json.loads(Path(manifest).read_text().splitlines()[0])
    path = Path(manifest).parent / first['sensors']['radar']['power']
    power = np.load(path)
    np.save(path, power + 0.01)
    with pytest.raises(ValueError, match='provenance|fingerprint|data'):
        run_training(output=tmp_path / 'resumed', resume=True, **common)


def test_smoke_runs_both_stages_and_writes_prediction_and_metrics(tmp_path):
    from radar_vla.pipeline import run_smoke

    torch.set_num_threads(1)
    result = run_smoke(tmp_path / 'smoke')
    assert result['synthetic'] is True
    assert (tmp_path / 'smoke/grounding/last.pt').is_file()
    assert (tmp_path / 'smoke/sft/last.pt').is_file()
    report = json.loads((tmp_path / 'smoke/test_metrics.json').read_text())
    assert report['samples'] > 0
    assert report['split'] == 'test'
    rows = [json.loads(x) for x in (tmp_path / 'smoke/predictions.jsonl').read_text().splitlines()]
    assert len(rows) == report['samples']
    assert rows[0]['risk_source'] == 'predicted'
    assert len(rows[0]['risk']) == 5
    assert 'plan' in rows[0]
