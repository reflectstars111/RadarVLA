"""Quality assignment changes supervision, without pruning objects or candidates."""
import math

import pytest
import torch

from radar_vla.losses import loss_grounding


def scene(anchors, predicted, targets, shape):
    anchors = torch.tensor(anchors, dtype=torch.float32)[None]
    predicted = torch.tensor(predicted, dtype=torch.float32)[None]
    targets = torch.tensor(targets, dtype=torch.float32).reshape(1, -1, 2)
    count, objects = anchors.shape[1], targets.shape[1]
    state = torch.cat((predicted, torch.zeros(1, count, 3)), -1).requires_grad_()
    output = dict(risk=torch.zeros(1, 5), risk_logits=torch.zeros(1, 3),
                  detection_logits=torch.zeros(1, count, requires_grad=True),
                  detection_class_logits=torch.zeros(1, count, 4),
                  detection_anchors=anchors, detection_grid_shape=torch.tensor(shape),
                  detection_state=state, detection_size=torch.tensor([2., 1.]).expand(1, count, 2).clone(),
                  detection_heading=torch.zeros(1, count),
                  detection_future=predicted[:, :, None],
                  detection_future_velocity=torch.zeros(1, count, 1, 2))
    batch = dict(agent_state=torch.cat((targets, torch.zeros(1, objects, 3)), -1),
                 agent_mask=torch.ones(1, objects, dtype=torch.bool),
                 agent_supervision_mask=torch.ones(1, dtype=torch.bool),
                 agent_size=torch.tensor([2., 1.]).expand(1, objects, 2).clone(),
                 agent_heading=torch.zeros(1, objects), agent_box_mask=torch.zeros(1, objects, dtype=torch.bool))
    return output, batch


def assign(output, batch, policy='local_quality_v1'):
    diagnostics = {}
    losses = loss_grounding(output, batch, assignment_policy=policy, assignment_diagnostics=diagnostics)
    gradient = torch.autograd.grad(losses['detection_objectness'], output['detection_logits'], retain_graph=True)[0]
    return (gradient[0] < 0).nonzero().flatten().tolist(), losses, diagnostics, gradient


def test_quality_assigns_accurate_neighbour_instead_of_wrong_nearest_anchor():
    output, batch = scene([[0, 0], [1, 0], [2, 0]], [[1.5, 0], [0, 0], [0, 0]], [[0, 0]], (1, 3))
    old, old_loss, _, _ = assign(output, batch, 'anchor_v1')
    chosen, new_loss, counts, _ = assign(output, batch)
    assert old == [0] and chosen == [1]
    assert old_loss['detection_position'] > 0 and new_loss['detection_position'] == 0
    assert set(old_loss) == set(new_loss)  # diagnostics must not become weighted losses.
    assert counts == dict(assignment_frames=1, local_quality_frames=1,
                          anchor_fallback_frames=0, anchor_fallback_targets=0)
    # Assignment is detached: objectness does not backpropagate through geometry.
    gradient = torch.autograd.grad(new_loss['detection_objectness'], output['detection_state'], allow_unused=True)[0]
    assert gradient is None or not gradient.any()


def test_rotated_iou_uses_orientation_when_centres_tie():
    output, batch = scene([[0, 0], [1, 0]], [[0, 0], [0, 0]], [[0, 0]], (1, 2))
    output['detection_size'][:] = torch.tensor([8., 2.])
    output['detection_heading'][0, 0] = math.pi/2
    batch['agent_size'][:] = torch.tensor([8., 2.])
    batch['agent_box_mask'][:] = True
    chosen, losses, _, _ = assign(output, batch)
    assert chosen == [1] and losses['detection_heading'] == 0


def test_unknown_boxes_do_not_use_nan_geometry_or_a_fabricated_iou():
    output, batch = scene([[0, 0], [1, 0]], [[.7, 0], [.1, 0]], [[0, 0]], (1, 2))
    batch['agent_size'][:] = float('nan')
    batch['agent_heading'][:] = float('nan')
    chosen, losses, _, _ = assign(output, batch)
    assert chosen == [1] and torch.isfinite(losses['total'])
    assert losses['detection_box_size'] == losses['detection_heading'] == 0


def test_neighbourhood_is_two_dimensional_and_does_not_wrap_at_row_end():
    anchors = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1]]
    predicted = [[8, 8], [8, 8], [2.6, 0], [2, 0], [2.2, 0], [8, 8]]
    output, batch = scene(anchors, predicted, [[2, 0]], (2, 3))
    assert assign(output, batch)[0] == [4]  # index 3 is perfect but not in this 3x3 neighbourhood.


def test_two_objects_in_one_cell_keep_distinct_slot_positives():
    output, batch = scene([[0, 0], [0, 0]], [[.2, 0], [-.2, 0]], [[0, 0], [.1, 0]], (1, 1))
    chosen, _, counts, _ = assign(output, batch)
    assert chosen == [0, 1]
    assert counts['anchor_fallback_frames'] == 0


def test_local_collision_falls_back_to_global_anchor_assignment_for_all_targets():
    output, batch = scene([[0, 0], [1, 0], [2, 0]], [[0, 0]]*3, [[0, 0]]*3, (1, 3))
    chosen, losses, counts, _ = assign(output, batch)
    old_chosen, old_loss, _, _ = assign(output, batch, 'anchor_v1')
    assert chosen == old_chosen == [0, 1, 2]
    for key in losses:
        torch.testing.assert_close(losses[key], old_loss[key], rtol=0, atol=0)
    assert counts['anchor_fallback_frames'] == 1
    assert counts['anchor_fallback_targets'] == 3


def test_candidate_coverage_is_respected_during_fallback():
    output, batch = scene([[1, 0], [2, 0], [3, 0]], [[1, 0], [3, 0], [3, 0]], [[3, 0]], (1, 3))
    limits = torch.tensor([[1.5, -.8, .8]])
    output.update(detection_candidate_mask=torch.tensor([[True, False, False]]),
                  detection_candidate_limits=limits)
    batch.update(detection_annotation_domain=limits, detection_annotation_domain_mask=torch.tensor([True]),
                 radar_pose=torch.eye(4)[None, None])
    chosen, _, counts, gradient = assign(output, batch)
    assert chosen == [0] and counts['anchor_fallback_frames'] == 1
    assert not gradient[0, 1:].any()


def test_ignored_background_does_not_remove_a_matched_positive():
    output, batch = scene([[0, 0], [1, 0], [2, 0]], [[1, 0], [0, 0], [2, 0]], [[0, 0]], (1, 3))
    batch.update(detection_ignore_boxes=torch.tensor([[[0., 0., 8., 2., 0.]]]),
                 detection_ignore_mask=torch.tensor([[True]]))
    chosen, _, _, gradient = assign(output, batch)
    assert chosen == [1]
    assert gradient[0, 0] == gradient[0, 2] == 0


def test_ground_truth_permutation_keeps_quality_matching_and_losses_unchanged():
    output, batch = scene([[0, 0], [0, 0], [1, 0], [1, 0]],
                          [[.1, 0], [0, 0], [.5, 0], [1, 0]], [[0, 0], [.1, 0]], (1, 2))
    batch['agent_state'][0, :, 3] = torch.tensor([1., 2.])
    batch['agent_size'][0] = torch.tensor([[2., 1.], [4., 2.]])
    batch['agent_box_mask'][:] = True
    first, losses, _, _ = assign(output, batch)
    reversed_batch = {key: value[:, [1, 0]] if value.ndim > 1 and key.startswith('agent_') else value
                      for key, value in batch.items()}
    second, other, _, _ = assign(output, reversed_batch)
    assert first == second
    for key in losses:
        torch.testing.assert_close(losses[key], other[key], rtol=0, atol=0)


def test_matching_does_not_depend_on_objectness_or_predicted_future():
    output, batch = scene([[0, 0], [1, 0]], [[1, 0], [0, 0]], [[0, 0]], (1, 2))
    assert assign(output, batch)[0] == [1]
    output['detection_logits'] = torch.tensor([[3., -3.]], requires_grad=True)
    output['detection_future'] = torch.full((1, 2, 1, 2), 1000.)
    output['detection_future_velocity'] = torch.full((1, 2, 1, 2), -1000.)
    assert assign(output, batch)[0] == [1]


@pytest.mark.parametrize('unknown', ['state', 'class', 'heading', 'size'])
def test_local_tie_order_ignores_unknown_placeholders_and_uses_stable_ids(unknown):
    output, batch = scene([[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0, 0]], (1, 1))
    batch['agent_ids'] = [['alpha', 'beta']]
    batch['agent_future'] = torch.tensor([[[[0., 0.]], [[10., 0.]]]])
    batch['agent_future_mask'] = torch.ones(1, 2, 1, dtype=torch.bool)
    output['detection_future'] = batch['agent_future'].clone()
    if unknown == 'state':
        batch['agent_state_mask'] = torch.ones_like(batch['agent_state'], dtype=torch.bool)
        batch['agent_state_mask'][..., 3] = False
        placeholder = batch['agent_state'][0, :, 3]
    elif unknown == 'class':
        batch['agent_class'] = torch.zeros(1, 2, dtype=torch.long)
        batch['agent_class_mask'] = torch.zeros(1, 2, dtype=torch.bool)
        placeholder = batch['agent_class'][0]
    elif unknown == 'heading':
        placeholder = batch['agent_heading'][0]
    else:
        placeholder = batch['agent_size'][0, :, 0]
    placeholder[:] = torch.tensor([1, 2])
    _, first, _, _ = assign(output, batch)
    _, legacy_first, _, _ = assign(output, batch, 'anchor_v1')
    placeholder[:] = torch.tensor([2, 1])
    _, second, _, _ = assign(output, batch)
    _, legacy_second, _, _ = assign(output, batch, 'anchor_v1')
    assert first['detection_agent_future'] == second['detection_agent_future'] == 0
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)
    # Retain the old protocol's ordering exactly, even at this historical edge.
    assert legacy_first['detection_agent_future'] == 0
    assert legacy_second['detection_agent_future'] > 0
    reversed_batch = {
        key: value[:, [1, 0]] if isinstance(value, torch.Tensor) and value.ndim > 1 else value
        for key, value in batch.items()}
    reversed_batch['agent_ids'] = [['beta', 'alpha']]
    _, reversed_losses, _, _ = assign(output, reversed_batch)
    for key in first:
        torch.testing.assert_close(first[key], reversed_losses[key], atol=0, rtol=0)


@pytest.mark.parametrize('annotated,targets', [(False, [[0, 0]]), (True, [])])
def test_unknown_and_empty_scenes_preserve_their_objectness_semantics(annotated, targets):
    output, batch = scene([[0, 0], [1, 0]], [[1, 0], [0, 0]], targets, (1, 2))
    batch['agent_supervision_mask'][:] = annotated
    chosen, losses, counts, gradient = assign(output, batch)
    assert chosen == [] and counts['assignment_frames'] == 0
    assert torch.isfinite(losses['total'])
    if annotated:
        assert (gradient > 0).all()
    else:
        assert not gradient.any()


def test_default_anchor_policy_is_exactly_legacy_and_diagnostics_are_optional():
    output, batch = scene([[0, 0], [1, 0]], [[1, 0], [0, 0]], [[0, 0]], (1, 2))
    output.pop('detection_grid_shape')
    default = loss_grounding(output, batch)
    explicit = loss_grounding(output, batch, assignment_policy='anchor_v1')
    assert default.keys() == explicit.keys()
    for key in default:
        torch.testing.assert_close(default[key], explicit[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match='grid'):
        loss_grounding(output, batch, assignment_policy='local_quality_v1')
    with pytest.raises(ValueError, match='assignment_policy'):
        loss_grounding(output, batch, assignment_policy='unknown')
