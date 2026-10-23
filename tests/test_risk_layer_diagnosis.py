import importlib.util
from pathlib import Path
import numpy as np
import torch
from radar_vla.losses import loss_grounding

spec=importlib.util.spec_from_file_location('layer_diagnosis',Path(__file__).parents[1]/'scripts/diagnose_risk_layers.py')
diagnosis=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnosis)

def test_per_record_decomposition_matches_canonical_loss_and_gradients():
    raw=torch.tensor([[.2,-1.,.5,2.,-2.],[-.3,.1,-.7,.2,1.]],requires_grad=True)
    risk=torch.cat([raw[:,:3].sigmoid(),torch.nn.functional.softplus(raw[:,3:])],-1)
    target=torch.tensor([[0.,1.,1.,0.,2.],[1.,0.,0.,5.,float('nan')]])
    mask=torch.tensor([[True]*5,[True,True,True,True,False]])
    components=diagnosis.component_losses(risk,raw,target,mask)
    for i in range(2):
        expected=loss_grounding(dict(risk=risk[i:i+1],risk_logits=raw[i:i+1,:3]),
            dict(risk_target=target[i:i+1],risk_mask=mask[i:i+1]))['total']
        torch.testing.assert_close(components[i].sum(),expected)
        a=torch.autograd.grad(components[i].sum(),raw,retain_graph=True)[0]
        b=torch.autograd.grad(expected,raw,retain_graph=True)[0]
        torch.testing.assert_close(a,b)
    assert torch.isfinite(components).all() and components[1,4]==0
    for j in range(5):
        gradient=torch.autograd.grad(components[0,j],raw,retain_graph=True)[0]
        assert gradient[0,j]!=0 and torch.count_nonzero(gradient[0])==1

def test_rotated_gt_box_audit_and_zero_gradient_cosines():
    agent=dict(position=[2.,3.],heading=np.pi/2,size=[4.,2.])
    points=np.array([[2.,4.9],[2.9,3.],[3.1,3.]])
    assert diagnosis.inside_box(points,agent).tolist()==[True,True,False]
    values=diagnosis.cosine_matrix([torch.tensor([1.,0.]),torch.tensor([-1.,0.]),torch.zeros(2)])
    assert values[0][1]==-1 and values[0][2] is None

def test_evidence_accepts_actual_dataset_tensor_contract():
    agent=dict(id='a',position=[1.,0.],size=[4.,4.],heading=0.,velocity=[0.,0.],
        future_xy=[[1.,0.]]*6,future_valid=[True]*6)
    record=dict(sample_id='s',episode_id='e',agents=[agent],future_times_s=[.5,1.,1.5,2.,2.5,3.],
        ego=dict(velocity=[1.,0.],yaw_rate=0.,box_size=[2.,1.]),local_risk=dict(pcol_source_agent_ids=[['a']]*3))
    item=dict(radar=torch.stack([torch.ones(2,2),-torch.ones(2,2)])[None],
        radar_cartesian=torch.zeros(1,2,2,2),radar_doppler_valid=torch.ones(1,2,2,dtype=torch.bool),
        radar_los=torch.tensor([1.,0.]).expand(1,2,2,2),radar_sensor_velocity=torch.tensor([[1.,0.]]),
        time_offsets_s=torch.tensor([0.]),agent_radial_observed=torch.tensor([-1.]),
        agent_radial_mask=torch.tensor([True]),agent_state=torch.tensor([[1.,0.,-1.,0.,0.]]))
    evidence,mask=diagnosis.input_evidence(record,item)
    assert evidence['agents'][0]['valid_positive_power_cells']==4 and mask.all()
    assert evidence['agents'][0]['cell_weighted_radial_abs_error_mps']==0.

def test_raw_probe_sampling_preserves_all_positives_and_is_reproducible():
    cache=dict(risk_target=torch.zeros(20,5),risk_mask=torch.ones(20,5,dtype=torch.bool))
    cache['risk_target'][[2,8,19],2]=1.
    first=diagnosis.paired_train_indices(cache,5)
    assert first==diagnosis.paired_train_indices(cache,5)
    assert len(first)==8 and len(set(first))==8 and {2,8,19}.issubset(first)
