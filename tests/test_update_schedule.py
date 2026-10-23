import pytest
import torch
from radar_vla.training_schedule import learning_rate_for_update


def test_update_endpoints_plateau_and_monotonicity():
    for schedule, flat in [('flat_cosine', 6), ('cosine', 0)]:
        rates = [learning_rate_for_update(i, 24, 1e-4, schedule, flat) for i in range(1,25)]
        assert rates[0] == pytest.approx(1e-4)
        assert rates[-1] == pytest.approx(1e-5)
        assert all(a >= b for a,b in zip(rates,rates[1:]))
        if flat:
            assert rates[:flat] == [1e-4]*flat
            assert rates[flat] < 1e-4


@pytest.mark.parametrize('values', [(0,10,1e-4,'cosine',0,.1), (1,1,1e-4,'cosine',0,.1),
    (1,10,1e-4,'flat_cosine',10,.1), (1,10,1e-4,'bad',0,.1),
    (1,10,1e-4,'cosine',0,float('nan')), (True,10,1e-4,'cosine',0,.1)])
def test_invalid_update_schedule(values):
    with pytest.raises(ValueError):learning_rate_for_update(*values)


def test_native_actual_update_lrs_tail_and_resume(tmp_path, monkeypatch):
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.pipeline import run_training
    torch.set_num_threads(1)
    manifest=write_synthetic_dataset(tmp_path/'data',scenes_per_split=1,frames_per_scene=7,seed=4)
    cfg={'model':{'hidden_dim':8,'num_heads':2,'num_queries':2,'max_agents':4,'horizon_steps':6}}
    args=dict(manifest=manifest,stage='grounding',epochs=4,batch_size=2,accumulation_steps=2,
        seed=7,config=cfg,device='cpu',lr=1e-4,validation_generation_samples=0,
        lr_schedule='flat_cosine',lr_decay_start_epoch=1,lr_min_ratio=.1,lr_schedule_unit='update')
    captured=[];original=torch.optim.AdamW.step
    def observe(opt,*a,**kw):
        captured.append(opt.param_groups[0]['lr']);return original(opt,*a,**kw)
    monkeypatch.setattr(torch.optim.AdamW,'step',observe)
    run_training(output=tmp_path/'full',**args)
    full=torch.load(tmp_path/'full/last.pt',weights_only=True)
    schedule=full['protocol']['learning_rate_schedule']
    assert schedule['updates_per_epoch']==2 and schedule['total_updates']==8
    expected=[learning_rate_for_update(i,8,1e-4,'flat_cosine',2,.1) for i in range(1,9)]
    assert captured==pytest.approx(expected)
    assert [v for row in full['history'] for v in row['optimizer_learning_rates']]==pytest.approx(expected)
    assert full['history'][-1]['optimizer_updates']==8
    assert full['optimizer']['param_groups'][0]['lr']==pytest.approx(1e-5)
    captured.clear()
    run_training(output=tmp_path/'resumed',stop_after_epoch=2,**args)
    run_training(output=tmp_path/'resumed',resume=True,**args)
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=True)
    assert captured==pytest.approx(expected)
    for key,value in full['grounder'].items():torch.testing.assert_close(value,resumed['grounder'][key],rtol=0,atol=0)
    assert full['history'][-1]['optimizer_learning_rates']==resumed['history'][-1]['optimizer_learning_rates']
    for change in [dict(lr_schedule='cosine'),dict(lr_min_ratio=.2),dict(epochs=5),dict(lr_schedule_unit='epoch')]:
        with pytest.raises(ValueError,match='protocol|provenance'):
            run_training(output=tmp_path/'resumed',resume=True,**dict(args,**change))
