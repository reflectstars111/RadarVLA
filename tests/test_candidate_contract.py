"""Train/decode share anchor coverage while boundary targets remain positive."""
import math
import pytest
import torch

from radar_vla.model import ModelConfig, RadarVLAGrounder
from radar_vla.losses import loss_grounding
from radar_vla.detection import decode_detections


def config(**kw):
    return ModelConfig(hidden_dim=16, detection_class_schema='agent_v1',
        detection_candidate_policy='supervised_sector_v1', **kw)


def fixture(targets=1):
    state = torch.tensor([[[50.5,0.,0.,0.,0.], [50.5,0.,0.,0.,0.]]])
    out = dict(risk=torch.zeros(1,5), risk_logits=torch.zeros(1,3),
        detection_logits=torch.tensor([[.5,8.]],requires_grad=True),
        detection_class_logits=torch.zeros(1,2,1), detection_state=state,
        detection_size=torch.tensor([[[4.,2.],[4.,2.]]]), detection_heading=torch.zeros(1,2),
        detection_future=state[:,:,:2,None].transpose(-1,-2),
        detection_future_velocity=torch.zeros(1,2,1,2),
        detection_anchors=torch.tensor([[[49.,0.],[50.5,0.]]]),
        detection_candidate_mask=torch.tensor([[True,False]]),
        detection_candidate_limits=torch.tensor([[50.,-math.radians(53),math.radians(53)]]))
    batch = dict(agent_mask=torch.ones(1,targets,dtype=torch.bool),
        agent_state=state[:,:targets].clone(), agent_supervision_mask=torch.tensor([True]),
        agent_future=out['detection_future'][:,:targets].clone(),
        agent_future_mask=torch.ones(1,targets,1,dtype=torch.bool),
        detection_annotation_domain=out['detection_candidate_limits'].clone(),
        detection_annotation_domain_mask=torch.tensor([True]),radar_pose=torch.eye(4)[None,None])
    return out,batch


def test_boundary_gt_is_assigned_to_valid_inner_anchor_and_outer_gradient_is_zero():
    out,batch=fixture()
    grad,=torch.autograd.grad(loss_grounding(out,batch)['detection_objectness'],out['detection_logits'])
    assert grad[0,0]<0 and grad[0,1]==0
    # The boundary GT and its full trajectory are retained.
    decoded=decode_detections(out,config(),torch.tensor([[1.]]))[0]
    assert decoded['num_agents']==1 and decoded['detections'][0]['position_m']==[50.5,0.]


def test_invalid_high_score_candidate_is_removed_before_nms_and_candidate_cap():
    out,_=fixture()
    rows=decode_detections(out,config(detection_max_candidates=1),torch.tensor([[1.]]))
    assert len(rows[0]['detections'])==1
    assert rows[0]['detections'][0]['confidence']==pytest.approx(torch.sigmoid(torch.tensor(.5)).item())
    assert not rows[0]['detection_truncated']


@pytest.mark.parametrize('fault',['missing','unknown','different'])
def test_loss_rejects_missing_or_mismatched_annotation_coverage(fault):
    out,batch=fixture()
    if fault=='missing':batch.pop('detection_annotation_domain')
    elif fault=='unknown':batch['detection_annotation_domain_mask'][:]=False
    else:batch['detection_annotation_domain'][:,0]=45.
    with pytest.raises(ValueError,match='candidate|coverage|domain'):
        loss_grounding(out,batch)


def test_insufficient_valid_candidate_capacity_fails_without_dropping_targets():
    out,batch=fixture(2)
    with pytest.raises(ValueError,match='candidate|capacity'):
        loss_grounding(out,batch)


def test_empty_scene_with_no_valid_candidates_has_finite_zero_detection_loss():
    out,batch=fixture(0);out['detection_candidate_mask'][:]=False
    out['detection_anchors'][:,:,0]=100.
    result=loss_grounding(out,batch)
    assert torch.isfinite(result['total']) and result['detection_objectness']==0
    grad,=torch.autograd.grad(result['total'],out['detection_logits'])
    assert (grad==0).all()
    assert decode_detections(out,config())[0]['num_agents']==0


def test_forward_coverage_uses_observations_not_gt_and_checks_rotated_sensor_pose():
    from test_radar_vla_model import example_batch
    data=example_batch(1)
    data['radar_pose']=torch.eye(4).repeat(1,3,1,1)
    data['radar_pose'][:,:,:2,:2]=torch.tensor([[0.,-1.],[1.,0.]])
    data['radar_pose'][:,:,:2,3]=torch.tensor([3.,4.])
    model=RadarVLAGrounder(config()).eval()
    with torch.no_grad():
        before=model(data)
        after=model(dict(data,agent_state=data['agent_state']*1000,
            detection_annotation_domain=torch.tensor([[5.,-.1,.1]]),
            detection_annotation_domain_mask=torch.tensor([True])))
    from radar_vla.annotation_domain import annotation_domain_mask
    expected=annotation_domain_mask(before['detection_anchors'],dict(data,
        detection_annotation_domain=before['detection_candidate_limits'],
        detection_annotation_domain_mask=torch.tensor([True])))
    assert torch.equal(before['detection_candidate_mask'],expected)
    for key in before:torch.testing.assert_close(before[key],after[key],rtol=0,atol=0)


def test_checkpoint_rejects_cross_policy_and_different_sector_even_with_strict_false():
    old=RadarVLAGrounder(ModelConfig(hidden_dim=16,detection_class_schema='agent_v1'))
    new=RadarVLAGrounder(config())
    with pytest.raises(ValueError,match='(?i)candidate'):
        new.load_state_dict(old.state_dict(),strict=False)
    with pytest.raises(ValueError,match='candidate'):
        old.load_state_dict(new.state_dict(),strict=False)
    other=RadarVLAGrounder(config(detection_candidate_range_m=40.))
    with pytest.raises(ValueError,match='candidate'):
        other.load_state_dict(new.state_dict(),strict=False)
    RadarVLAGrounder(config()).load_state_dict(new.state_dict())
    assert '_candidate_sector' not in old.state_dict()


@pytest.mark.parametrize('fault',['missing','nonbool','shape'])
def test_decode_rejects_missing_or_malformed_validity_mask(fault):
    out,_=fixture()
    if fault=='missing':out.pop('detection_candidate_mask')
    elif fault=='nonbool':out['detection_candidate_mask']=out['detection_candidate_mask'].float()
    else:out['detection_candidate_mask']=torch.ones(1,1,dtype=torch.bool)
    with pytest.raises(ValueError,match='(?i)candidate'):
        decode_detections(out,config())


def test_more_than_eight_boundary_targets_receive_distinct_valid_anchors():
    out,batch=fixture()
    for key,value in list(out.items()):
        if value.ndim>=2 and value.shape[1]==2:
            out[key]=value.repeat_interleave(6,dim=1).detach()
    out['detection_candidate_mask'][:]=True
    out['detection_anchors'][:,:,0]=49.
    out['detection_logits'].requires_grad_(True)
    batch['agent_state']=batch['agent_state'].repeat(1,9,1)
    batch['agent_mask']=torch.ones(1,9,dtype=torch.bool)
    batch['agent_future']=batch['agent_future'].repeat(1,9,1,1)
    batch['agent_future_mask']=torch.ones(1,9,1,dtype=torch.bool)
    result=loss_grounding(out,batch)
    grad,=torch.autograd.grad(result['detection_objectness'],out['detection_logits'])
    assert (grad<0).sum()==9 and (grad>0).sum()==3


def test_decode_rejects_config_with_a_different_sector():
    out,_=fixture()
    with pytest.raises(ValueError,match='(?i)candidate'):
        decode_detections(out,config(detection_candidate_range_m=45.))


def test_raw_physical_metrics_do_not_match_to_an_ineligible_perfect_prediction():
    from radar_vla.metrics import MetricAccumulator
    out,batch=fixture()
    out['detection_state'][0,0,0]=49.
    out['agent_state']=out['detection_state'];out['agent_future']=out['detection_future']
    batch['risk_target']=torch.zeros(1,5);batch['risk_mask']=torch.ones(1,5,dtype=torch.bool)
    metric=MetricAccumulator();metric.update(out,batch)
    assert metric.compute()['position_mae']==1.5


def test_unrestricted_decoder_cannot_silently_consume_restricted_output():
    out,_=fixture()
    with pytest.raises(ValueError,match='(?i)candidate'):
        decode_detections(out,ModelConfig(detection_class_schema='agent_v1'))
