"""Reference math, gradients, and both real prefix implementations."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import isotonic_regression
import torch
from torch import nn

from radar_vla.hf_planner import HFRiskConditionedPlanner
from radar_vla.pipeline import _override_risk_projection
from radar_vla.planner import PlannerConfig,RiskConditionedPlanner
from radar_vla.risk_state import project_cumulative_risk
from radar_vla.tokenizer import PhysicalTokenizer


def test_projection_matches_independent_scipy_reference_and_preserves_regressions():
    risk=torch.tensor(np.random.default_rng(42).random((257,5)),dtype=torch.float64)
    actual=project_cumulative_risk(risk)
    reference=np.stack([isotonic_regression(row[:3].numpy()).x for row in risk])
    np.testing.assert_allclose(actual[:,:3].numpy(),reference,rtol=0,atol=1e-12)
    assert torch.equal(actual[:,3:],risk[:,3:])
    assert torch.equal(project_cumulative_risk(actual),actual)
    assert bool((actual[:,1:3]>=actual[:,:2]).all())
    for truth in [[0.,0.,0.],[0.,0.,1.],[0.,1.,1.],[1.,1.,1.]]:
        y=risk.new_tensor(truth)
        assert bool(((actual[:,:3]-y).square().sum(-1)<=(risk[:,:3]-y).square().sum(-1)+1e-12).all())


def test_projection_gradients_and_arbitrary_leading_dimensions():
    risk=torch.tensor([[.1,.3,.8,7.,1.],[.8,.2,.9,8.,2.],
                       [.1,.8,.4,9.,3.],[.9,.2,.3,10.,4.]],dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(project_cumulative_risk,(risk,),eps=1e-6,atol=1e-6)
    reshaped=project_cumulative_risk(risk.reshape(2,2,5))
    torch.testing.assert_close(reshaped.reshape(4,5),project_cumulative_risk(risk),rtol=0,atol=0)


class NativeEmbeddings(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(4,8)
    def get_input_embeddings(self):return self.embedding


def prefix_planner(backend,projection):
    config=PlannerConfig(hidden_dim=8,radar_dim=8,num_heads=2,num_layers=1,risk_projection=projection)
    if backend=='tiny':return RiskConditionedPlanner(config)
    # Exercise the actual HF encode_prefix without loading a language backbone.
    planner=HFRiskConditionedPlanner.__new__(HFRiskConditionedPlanner);nn.Module.__init__(planner)
    planner.config=config;planner.use_lora=False;planner.language_model=NativeEmbeddings()
    planner.tokenizer=PhysicalTokenizer(config.bins,config.coordinate_limit_m,config.velocity_limit_mps)
    planner.native_tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=0,encode=lambda *a,**k:[1])
    planner.radar_projector=nn.Linear(8,8);planner.risk_projector=nn.Linear(5,8)
    planner.physical_embedding=nn.Embedding(planner.tokenizer.vocab_size,8)
    planner.register_buffer('native_to_physical',torch.full((4,),-1,dtype=torch.long))
    return planner


@pytest.mark.parametrize('backend',['tiny','hf'])
@pytest.mark.parametrize('projection',['none','isotonic'])
def test_both_prefix_backends_apply_projection_before_exactly_one_unit_scaling(backend,projection):
    planner=prefix_planner(backend,projection)
    original=torch.tensor([[.9,.1,.2,8.,1.]])
    risk=original.clone();seen=[]
    hook=planner.risk_projector.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    planner.encode_prefix(torch.zeros(1,2,8),risk,['go'])
    hook.remove()
    expected=(original if projection=='none' else torch.tensor([[.4,.4,.4,8.,1.]]))/torch.tensor([1.,1.,1.,80.,10.])
    torch.testing.assert_close(seen[0],expected,rtol=0,atol=1e-7)
    assert torch.equal(original,risk)
    if projection=='none':assert planner.condition_risk(risk) is risk


@pytest.mark.parametrize('risk',[torch.zeros(2,4),torch.ones(2,5,dtype=torch.long),
    torch.tensor([[float('nan'),0.,1.,2.,3.]]),torch.tensor([[-.1,0.,1.,2.,3.]]),
    torch.tensor([[0.,1.,1.1,2.,3.]])])
def test_invalid_risk_is_rejected_instead_of_silently_clipped(risk):
    with pytest.raises(ValueError):project_cumulative_risk(risk)


def test_checkpoint_defaults_and_explicit_override_validation():
    planner=prefix_planner('tiny','none')
    _override_risk_projection(planner,None);assert planner.config.risk_projection=='none'
    _override_risk_projection(planner,'isotonic');assert planner.config.risk_projection=='isotonic'
    with pytest.raises(ValueError):_override_risk_projection(planner,'unknown')
    with pytest.raises(ValueError):_override_risk_projection(None,'isotonic')
    with pytest.raises(ValueError):PlannerConfig(risk_projection='unknown')
