"""Executable research protocol and intervention behavior, without training jobs."""
import copy
import pytest
import torch


def test_ablation_plan_pairs_seeds_and_freezes_budget_without_launching(tmp_path):
    from radar_vla.experiments import build_experiment_plan
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    plan = build_experiment_plan({'model': {}, 'planner': {}}, manifest, tmp_path,
                                 seeds=[42, 43], epochs=10, batch_size=1, accumulation_steps=4)
    tasks = plan['tasks']
    assert len(tasks) == 20
    assert all(t['epochs'] == 10 and t['batch_size'] == 1 and t['accumulation_steps'] == 4 for t in tasks)
    oracle = next(t for t in tasks if t['variant'] == 'risk_oracle' and t['seed'] == 42)
    predicted = next(t for t in tasks if t['variant'] == 'risk_predicted' and t['seed'] == 42)
    assert oracle['dependencies'] == predicted['dependencies']
    no_risk = next(t for t in tasks if t['variant'] == 'risk_none')
    assert no_risk['config']['planner']['use_risk_token'] is False
    assert not tmp_path.joinpath('risk_oracle').exists()


def test_doppler_intervention_changes_only_selected_observations():
    from radar_vla.experiments import intervene_doppler
    batch = {'radar': torch.zeros(1, 2, 2, 3, 4),
             'range_m': torch.tensor([[1., 2., 3.]]),
             'azimuth_rad': torch.tensor([[-.3, -.1, .1, .3]]),
             'time_offsets_s': torch.tensor([[-.1, 0.]]),
             'instruction': ['stay in lane']}
    batch['radar'][:, :, 0] = 2.
    original = copy.deepcopy(batch)
    changed = intervene_doppler(batch, -8., (1.5, 2.5), (-.2, .2), history='current')
    torch.testing.assert_close(changed['radar'][:, :, 0], original['radar'][:, :, 0])
    assert changed['radar'][0, -1, 1, 1, 1:3].tolist() == [-8., -8.]
    assert changed['radar'][:, 0, 1].sum() == 0.
    assert torch.equal(batch['radar'], original['radar'])
    assert changed['instruction'] == original['instruction']
    with pytest.raises(ValueError, match='empty'):
        intervene_doppler(batch, -8., (5., 6.), (-.2, .2))


def test_oracle_never_silently_uses_unknown_risk_targets():
    from radar_vla.pipeline import select_risk
    predicted = torch.ones(2, 5)
    batch = {'risk_target': torch.zeros(2, 5), 'risk_mask': torch.ones(2, 5, dtype=torch.bool)}
    assert torch.equal(select_risk(predicted, batch, 'oracle'), torch.zeros(2, 5))
    batch['risk_mask'][1, 4] = False
    with pytest.raises(ValueError, match='oracle|Oracle'):
        select_risk(predicted, batch, 'oracle')
    assert torch.equal(select_risk(predicted, batch, 'predicted'), predicted)


def test_common_cohort_reports_missing_agent_future_and_preserves_original(tmp_path):
    import json
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.cohort import build_cohort
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=2)
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    rows[0]['agents'][0]['future_valid'][-1] = False
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original = manifest.read_bytes()
    output = tmp_path / 'cohort.jsonl'
    result = build_cohort(manifest, output)
    assert result['retained_samples'] == 5
    assert result['excluded'][0]['sample_id'] == rows[0]['sample_id']
    assert 'agent future' in result['excluded'][0]['reason']
    assert manifest.read_bytes() == original
    assert output.with_suffix('.jsonl.cohort.json').is_file()
    assert all(r['sample_id'] != rows[0]['sample_id'] for r in map(json.loads, output.read_text().splitlines()))


def test_changed_data_aborts_frozen_queue_before_launch(tmp_path, monkeypatch):
    import json
    import numpy as np
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.experiments import build_experiment_plan, run_experiment_plan
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    plan = build_experiment_plan({}, manifest, tmp_path / 'runs', seeds=[42])
    plan_path = tmp_path / 'plan.json'; plan_path.write_text(json.dumps(plan))
    row = json.loads(manifest.read_text().splitlines()[0])
    source = manifest.parent / row['sensors']['radar']['power']
    np.save(source, np.load(source)+.1)
    monkeypatch.setattr('radar_vla.experiments.subprocess.run', lambda *a, **k: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='Dataset changed'):
        run_experiment_plan(plan_path, device='cpu')
    assert not (tmp_path / 'runs').exists()
