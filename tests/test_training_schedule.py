"""Actual native-training resume and common-prefix controls for epoch LR schedules."""
import json
import math
import pytest
import torch


def test_flat_cosine_keeps_first_five_epochs_and_reaches_floor_at_twenty():
    from radar_vla.training_schedule import learning_rate_for_epoch
    rates=[learning_rate_for_epoch(i,20,1e-4,'flat_cosine',5,.1) for i in range(1,21)]
    assert rates[:5]==[1e-4]*5
    assert rates[-1]==pytest.approx(1e-5)
    assert all(a>b for a,b in zip(rates[4:],rates[5:]))
    assert learning_rate_for_epoch(10,20,1e-4,'constant',5,.1)==1e-4


@pytest.mark.parametrize('schedule,start,ratio',[('unknown',5,.1),('flat_cosine',20,.1),
    ('flat_cosine',-1,.1),('flat_cosine',True,.1),('flat_cosine',5,-.1),('flat_cosine',5,float('nan'))])
def test_invalid_schedule_is_rejected(schedule,start,ratio):
    from radar_vla.training_schedule import learning_rate_for_epoch
    with pytest.raises(ValueError):learning_rate_for_epoch(1,20,1e-4,schedule,start,ratio)


def test_native_schedule_resume_and_initial_constant_prefix_are_reproducible(tmp_path):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import run_training
    torch.set_num_threads(1)
    manifest=write_synthetic_dataset(tmp_path/'data',scenes_per_split=1,frames_per_scene=2,seed=4)
    cfg={'model':{'hidden_dim':8,'num_heads':2,'num_queries':2,'max_agents':4,'horizon_steps':6}}
    common=dict(manifest=manifest,stage='grounding',epochs=4,batch_size=2,seed=7,
        config=cfg,device='cpu',lr=1e-4,validation_generation_samples=0,
        lr_schedule='flat_cosine',lr_decay_start_epoch=2,lr_min_ratio=.1)
    run_training(output=tmp_path/'full',**common)
    run_training(output=tmp_path/'resume',stop_after_epoch=2,**common)
    run_training(output=tmp_path/'resume',resume=True,**common)
    full=torch.load(tmp_path/'full/last.pt',weights_only=True)
    resumed=torch.load(tmp_path/'resume/last.pt',weights_only=True)
    for k,v in full['grounder'].items():torch.testing.assert_close(v,resumed['grounder'][k],rtol=0,atol=0)
    assert [r['learning_rate'] for r in full['history']]==pytest.approx([1e-4,1e-4,5.5e-5,1e-5])
    assert full['optimizer']['param_groups'][0]['lr']==pytest.approx(1e-5)
    assert full['protocol']['learning_rate_schedule']['name']=='flat_cosine'
    with pytest.raises(ValueError,match='protocol|provenance'):
        run_training(output=tmp_path/'resume',resume=True,**dict(common,lr_min_ratio=.2))
    run_training(output=tmp_path/'constant',stop_after_epoch=2,**dict(common,lr_schedule='constant'))
    a=torch.load(tmp_path/'full/epoch_002.pt',weights_only=True)
    b=torch.load(tmp_path/'constant/epoch_002.pt',weights_only=True)
    for k,v in a['grounder'].items():torch.testing.assert_close(v,b['grounder'][k],rtol=0,atol=0)
