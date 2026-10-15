"""Training launch selection is checked without allocating GPUs."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_cli_selected_gpus_set_ddp_rank_count(tmp_path):
    paths = tmp_path/'paths.json'
    paths.write_text(json.dumps({
        'python':sys.executable, 'data_root':str(tmp_path),
        'collection_manifest':str(tmp_path/'collection.yaml'),
        'model_config':str(ROOT/'configs/l40s_qwen25_3b.json'),
        'qwen_model':str(tmp_path/'qwen'), 'run_root':str(tmp_path/'runs')}))
    command = [sys.executable, str(ROOT/'scripts/radarvla_cli.py'),
               '--project-root', str(ROOT), '--paths', str(paths),
               'train-grounding', '--dry-run', '--gpus', '1,3']
    result = subprocess.run(command, text=True, capture_output=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'CUDA_VISIBLE_DEVICES=1,3' in result.stdout
    assert '--nproc_per_node=2' in result.stdout
    invalid = subprocess.run([*command[:-1], '1,1'], text=True, capture_output=True,
                             cwd=tmp_path)
    assert invalid.returncode == 2
    assert 'distinct' in invalid.stderr


def test_direct_launcher_respects_visible_gpu_count_and_rejects_mismatch(tmp_path):
    env = dict(os.environ, MANIFEST=str(tmp_path/'manifest.jsonl'),
               OUTPUT=str(tmp_path/'run'), STAGE='grounding',
               CUDA_VISIBLE_DEVICES='2,6', RADAR_VLA_PYTHON=sys.executable)
    env.pop('NPROC_PER_NODE', None)
    command = ['bash', str(ROOT/'launch_l40s.sh'), '--dry-run']
    selected = subprocess.run(command, env=env, text=True, capture_output=True, cwd=tmp_path)
    assert selected.returncode == 0, selected.stderr
    assert '--nproc_per_node=2' in selected.stdout
    wrong = subprocess.run(command, env=dict(env, NPROC_PER_NODE='3'),
                           text=True, capture_output=True, cwd=tmp_path)
    assert wrong.returncode == 2
    assert 'must equal 2 selected GPUs' in wrong.stderr


def test_training_and_validation_show_rank_zero_progress(tmp_path, capsys):
    import torch
    from radar_vla.pipeline import run_training
    from radar_vla.synthetic import write_synthetic_dataset

    torch.set_num_threads(1)
    manifest = write_synthetic_dataset(tmp_path/'data', scenes_per_split=1,
                                       frames_per_scene=2)
    config = {'model': {'hidden_dim':16, 'num_heads':2, 'num_queries':4,
                        'max_agents':4, 'horizon_steps':6},
              'planner': {'hidden_dim':16, 'num_heads':2, 'num_layers':1}}
    status = run_training(manifest, tmp_path/'run', config=config, epochs=1,
                          batch_size=1, device='cpu')
    assert status['state'] == 'complete'
    progress = capsys.readouterr().err
    assert 'grounding train 1/1' in progress
    assert 'grounding val 1/1' in progress
    assert 'batch/s' in progress or 'it/s' in progress
