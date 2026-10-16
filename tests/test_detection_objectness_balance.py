"""Dense background mistakes must not disappear into a background-only mean."""
import pytest
import torch

from radar_vla.losses import loss_grounding


def dense_fixture(logits, positive_indices=(0,)):
    """Keep geometry exact so these tests isolate detector confidence learning."""
    b, q = logits.shape
    anchors = torch.stack((torch.arange(1, q + 1, dtype=logits.dtype),
                           torch.zeros(q, dtype=logits.dtype)), -1)[None].expand(b, -1, -1)
    state = torch.cat((anchors, torch.zeros(b, q, 3)), -1)
    output = dict(risk=torch.zeros(b, 5), risk_logits=torch.zeros(b, 3),
                  detection_logits=logits, detection_class_logits=torch.zeros(b, q, 4),
                  detection_state=state, detection_size=torch.ones(b, q, 2),
                  detection_heading=torch.zeros(b, q), detection_anchors=anchors,
                  detection_future=anchors[:, :, None],
                  detection_future_velocity=torch.zeros(b, q, 1, 2))
    batch = dict(agent_state=state[:, list(positive_indices)].clone(),
                 agent_mask=torch.ones(b, len(positive_indices), dtype=torch.bool),
                 agent_supervision_mask=torch.ones(b, dtype=torch.bool))
    return output, batch


def objectness(logits, positive_indices=(0,)):
    output, batch = dense_fixture(logits, positive_indices)
    return loss_grounding(output, batch)['detection_objectness']


def test_more_ambiguous_background_candidates_increase_false_positive_penalty():
    # A larger image with the same one object and more equally bad background
    # predictions must not have the same confidence loss.
    small = torch.full((1, 8), .6)
    large = torch.full((1, 256), .6)
    small[:, 0] = large[:, 0] = .9
    assert objectness(torch.logit(large)) > 20 * objectness(torch.logit(small))


def test_sparse_scene_pushes_an_uninformative_shared_score_below_detection_threshold():
    shared_logit = torch.tensor(0., requires_grad=True)
    loss = objectness(shared_logit.expand(1, 3456), (0, 1, 2, 3))
    loss.backward()
    # Positive gradient means gradient descent lowers the shared score. Separate
    # positive/negative means cancelled exactly here despite 3452 empty cells.
    assert shared_logit.grad > 1.


def test_hundreds_of_false_positives_cannot_have_negligible_confidence_loss():
    probabilities = torch.full((1, 3456), .01)
    probabilities[:, :4] = .9
    probabilities[:, 4:204] = .6
    # This has 4 TP and 200 FP at the public 0.5 threshold (1.96% precision).
    assert objectness(torch.logit(probabilities), (0, 1, 2, 3)) > 1.


def test_sparse_scene_still_increases_matched_positive_scores():
    logits = torch.zeros(1, 128, requires_grad=True)
    objectness(logits, (3, 17)).backward()
    assert (logits.grad < 0).nonzero().tolist() == [[0, 3], [0, 17]]
    assert (logits.grad > 0).sum() == 126


def test_empty_annotated_scene_trains_background_without_nonfinite_losses():
    logits = torch.zeros(1, 16, requires_grad=True)
    loss = objectness(logits, ())
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    assert torch.isfinite(logits.grad).all() and (logits.grad > 0).all()


def test_unannotated_frames_change_neither_loss_nor_supervised_gradient():
    reference_logits = torch.zeros(1, 16, requires_grad=True)
    reference_loss = objectness(reference_logits)
    reference_loss.backward()
    logits = torch.zeros(2, 16, requires_grad=True)
    output, batch = dense_fixture(logits)
    batch['agent_supervision_mask'][1] = False
    loss = loss_grounding(output, batch)['detection_objectness']
    loss.backward()
    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(logits.grad[:1], reference_logits.grad)
    assert not logits.grad[1].any()


def test_entirely_unannotated_batch_keeps_graph_with_zero_gradient():
    logits = torch.zeros(2, 16, requires_grad=True)
    output, batch = dense_fixture(logits)
    batch['agent_supervision_mask'][:] = False
    loss = loss_grounding(output, batch)['detection_objectness']
    loss.backward()
    assert loss.item() == 0.
    assert logits.grad is not None and not logits.grad.any()


@pytest.mark.parametrize('positive_index', [0, 3])
def test_annotation_domain_masks_only_unmatched_background(positive_index):
    logits = torch.zeros(1, 4, requires_grad=True)
    output, batch = dense_fixture(logits, (positive_index,))
    # Anchors are x=1,2,3,4. A boundary-intersecting target may have its center
    # outside the 2.5m annotation range and must retain positive supervision.
    batch.update(detection_annotation_domain=torch.tensor([[2.5, -.8, .8]]),
                 detection_annotation_domain_mask=torch.tensor([True]),
                 radar_pose=torch.eye(4)[None, None])
    loss = loss_grounding(output, batch)['detection_objectness']
    loss.backward()
    assert logits.grad[0, positive_index] < 0
    assert logits.grad[0, 1] > 0
    assert logits.grad[0, 2] == 0
    if positive_index == 0:
        assert logits.grad[0, 3] == 0


def test_adding_candidates_outside_annotation_domain_does_not_dilute_loss():
    losses = []
    for q in (4, 128):
        output, batch = dense_fixture(torch.zeros(1, q))
        batch.update(detection_annotation_domain=torch.tensor([[2.5, -.8, .8]]),
                     detection_annotation_domain_mask=torch.tensor([True]),
                     radar_pose=torch.eye(4)[None, None])
        losses.append(loss_grounding(output, batch)['detection_objectness'])
    torch.testing.assert_close(*losses)
