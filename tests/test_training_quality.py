"""Regressions for mode collapse and selecting checkpoints by generated quality."""
import copy
import math

import pytest
import torch

from radar_vla.tokenizer import PhysicalTokenizer, token_cross_entropy


def test_mode_auxiliary_loss_is_not_diluted_by_longer_descriptions():
    tok = PhysicalTokenizer(bins=9)
    def extra(length):
        target = torch.full((1, length), tok.eos_id)
        target[0, 0] = tok.token('<SHORT>')
        logits = torch.zeros(1, length, tok.vocab_size, requires_grad=True)
        base = token_cross_entropy(logits, target, tok)
        loss = token_cross_entropy(logits, target, tok, mode_weight=1.)
        (loss - base).backward()
        return (loss - base).item(), logits.grad[0, 0, tok.token('<SHORT>')].item()
    for length in (8, 100):
        value, gradient = extra(length)
        assert value == pytest.approx(math.log(2), abs=1e-6)
        assert gradient == pytest.approx(-.5, abs=1e-6)


def test_unknown_mode_has_no_auxiliary_loss():
    tok = PhysicalTokenizer(bins=9)
    target = torch.tensor([[tok.pad_id, tok.eos_id]])
    logits = torch.zeros(1, 2, tok.vocab_size, requires_grad=True)
    assert token_cross_entropy(logits, target, tok, mode_weight=1.) == token_cross_entropy(logits, target, tok)


def samples_and_rows():
    samples = [dict(sample_id='safe', ego_future=torch.tensor([[5., 0.], [10., 0.], [15., 0.]]),
                    ego_future_mask=torch.ones(3, dtype=torch.bool),
                    future_times_s=torch.tensor([1., 2., 3.]), ego_velocity=torch.tensor([5., 0.]))]
    rows = [dict(sample_id='safe', plan=dict(valid=True, mode='long',
                 ego_trajectory=[[5., 0.], [10., 0.], [15., 0.]], ego_times_s=[1., 2., 3.]))]
    return samples, rows


def test_shortcut_plan_cannot_win_by_omitting_difficult_late_points():
    from radar_vla.training_quality import generation_diagnostics, quality_scores
    samples, full = samples_and_rows()
    full[0]['plan']['ego_trajectory'][2] = [18., 0.]
    short = copy.deepcopy(full)
    short[0]['plan'].update(mode='short', ego_trajectory=[[5., 0.]], ego_times_s=[1.])
    a = generation_diagnostics(full, samples, ['long'], short_horizon_s=1.)
    b = generation_diagnostics(short, samples, ['long'], short_horizon_s=1.)
    assert a['required_point_coverage'] == 1.
    assert b['required_point_coverage'] == pytest.approx(1/3)
    assert b['required_ego_ade_m'] == 0.
    assert quality_scores({'free_generation': a})['planning'] < quality_scores({'free_generation': b})['planning']


def test_generation_reports_short_recall_and_constant_velocity_baseline():
    from radar_vla.training_quality import generation_diagnostics
    samples, rows = samples_and_rows()
    result = generation_diagnostics(rows, samples, ['short'], short_horizon_s=1.)
    assert result['mode_confusion']['short']['long'] == 1
    assert result['short_recall'] == 0.
    assert result['required_points'] == 1
    assert result['constant_velocity']['ego_ade_m'] == 0.
    assert result['constant_velocity']['ego_final_horizon_fde_m'] == 0.
    assert result['constant_velocity']['required_points'] == 1
    assert result['constant_velocity']['required_ego_ade_m'] == 0.


def test_generation_rejects_missing_or_duplicate_predictions():
    from radar_vla.training_quality import generation_diagnostics
    samples, rows = samples_and_rows()
    for incomplete in ([], rows + rows):
        with pytest.raises(ValueError, match='one prediction'):
            generation_diagnostics(incomplete, samples, ['long'])


def test_detection_checkpoint_score_uses_operating_quality_not_loss():
    from radar_vla.training_quality import quality_scores
    bad = dict(loss=.01, detection=dict(f1=.03, count_mae=195.))
    good = dict(loss=10., detection=dict(f1=.6, count_mae=2.))
    assert quality_scores(good)['detection'] < quality_scores(bad)['detection']


def test_validation_subset_includes_rare_modes_and_is_repeatable():
    from radar_vla.training_quality import select_generation_indices
    modes = ['long'] * 100 + ['short'] * 2
    first = select_generation_indices(modes, limit=8, seed=7)
    assert len(first) == len(set(first)) == 8
    assert any(modes[i] == 'short' for i in first)
    assert first == select_generation_indices(modes, limit=8, seed=7)
    assert select_generation_indices(modes, limit=0, seed=7) == []


def test_training_preserves_detection_best_when_lower_loss_model_has_more_false_positives(tmp_path, monkeypatch):
    from radar_vla import pipeline
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    validations = iter([
        dict(loss=10., detection=dict(f1=.6, count_mae=2.)),
        dict(loss=1., detection=dict(f1=.03, count_mae=195.)),
    ])
    # Controlled validation results isolate selection from stochastic learning;
    # training and checkpoint writes still execute the real pipeline.
    monkeypatch.setattr(pipeline, 'validate', lambda *a, **k: next(validations))
    output = tmp_path / 'run'
    pipeline.run_training(manifest, output, epochs=2,
        config=dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2)))
    assert torch.load(output / 'best.pt', weights_only=True)['epoch'] == 2
    assert torch.load(output / 'best_detection.pt', weights_only=True)['epoch'] == 1
    last = torch.load(output / 'last.pt', weights_only=True)
    assert last['best_quality_epoch']['detection'] == 1


def test_repaired_objectives_train_and_evaluate_together(tmp_path):
    from radar_vla import pipeline
    from radar_vla.synthetic import write_synthetic_dataset
    manifest = write_synthetic_dataset(tmp_path / 'data', scenes_per_split=1, frames_per_scene=1)
    config = dict(model=dict(hidden_dim=8, num_heads=2, num_queries=2, planner_ego_context=True),
                  planner=dict(num_layers=1, bins=65, ego_initial_velocity_weight=1., mode_loss_weight=1.),
                  language_model=dict(backend='tiny'))
    pipeline.run_training(manifest, tmp_path / 'grounding', epochs=1, config=config)
    pipeline.run_training(manifest, tmp_path / 'sft', stage='sft', epochs=1, config=config,
                          init_grounding=tmp_path / 'grounding/best_detection.pt')
    checkpoint = tmp_path / 'sft/best_planning.pt'
    grounder, planner, state = pipeline.load_models(checkpoint)
    assert grounder.config.planner_ego_context
    assert planner.config.ego_initial_velocity_weight == 1.
    assert state['best_quality_epoch']['planning'] == 1
    report = pipeline.evaluate_checkpoint(checkpoint, manifest, tmp_path / 'test.json')
    assert report['free_generation']['constant_velocity']['samples'] == 1
    assert report['free_generation']['teacher_mode_counts']['long'] + report['free_generation']['teacher_mode_counts']['short'] == 1
