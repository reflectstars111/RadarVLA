"""Loss coefficient provenance and resume behavior; model predictions stay unchanged."""
import copy
import pytest
import torch
from radar_vla import pipeline
from radar_vla.model import ModelConfig,RadarVLAGrounder
from test_radar_vla_model import example_batch


def test_loss_override_changes_only_requested_objective_and_keeps_default_protocol():
    torch.set_num_threads(1);torch.manual_seed(42)
    model=RadarVLAGrounder(ModelConfig(hidden_dim=8,num_heads=2))
    batch=example_batch()
    original,components,pred=pipeline._loss(model,None,batch)
    model.loss_weights={'distance':.25}
    changed,other,again=pipeline._loss(model,None,batch)
    torch.testing.assert_close(changed,original-.75*components['distance'])
    for k in pred:torch.testing.assert_close(pred[k],again[k],rtol=0,atol=0)
    for k in components:
        if k!='total':torch.testing.assert_close(components[k],other[k],rtol=0,atol=0)
    assert 'loss_weights' not in pipeline.resolve_config({})
    for invalid in [{'scale_distance':20},{'distance':-1},{'brake':True},{'distance':float('nan')}]:
        with pytest.raises(ValueError):pipeline.resolve_config({'loss_weights':invalid})


def test_resume_and_reload_keep_loss_weights_and_reject_changed_coefficient(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    torch.set_num_threads(1)
    manifest=write_synthetic_dataset(tmp_path/'data',scenes_per_split=1,frames_per_scene=2,seed=4)
    config={'model':{'hidden_dim':8,'num_heads':2},'loss_weights':{'distance':.25}}
    args=dict(manifest=manifest,epochs=2,batch_size=2,seed=42,device='cpu',config=config)
    output=tmp_path/'run'
    pipeline.run_training(output=output,stop_after_epoch=1,**args)
    model,_,state=pipeline.load_models(output/'last.pt')
    assert model.loss_weights=={'distance':.25}
    assert state['protocol']['config']['loss_weights']=={'distance':.25}
    changed=copy.deepcopy(config);changed['loss_weights']={'distance':1.}
    with pytest.raises(ValueError,match='protocol|provenance|fingerprint'):
        pipeline.run_training(output=output,resume=True,**{**args,'config':changed})
    pipeline.run_training(output=output,resume=True,**args)
    final=torch.load(output/'last.pt',weights_only=True)
    assert final['epoch']==2 and final['protocol']['config']['loss_weights']=={'distance':.25}
