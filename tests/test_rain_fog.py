"""New-batch geometry, background supervision and quota invariants."""
import copy
import math
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'collection/collector'))
from agent_policy import POLICY, LEGACY_POLICY, full_bbox_inside, label_exclusion_reason, motion_metadata
from batch_selection import choose_with_quotas
from test_moving_agents import actor, inputs
from radar_vla.moving_agents import rebuild_agents, attach_supervision, validate_motion


def test_full_bbox_boundary_rotation_offset_and_sensor_origin():
    a = actor(1, x=49.)
    assert not full_bbox_inside(a, [0,0,0])
    a['pose']['position_m'][0] = math.sqrt(2500-1)-2
    assert not full_bbox_inside(a, [0,0,0])
    a['pose']['position_m'][0] -= 1e-5
    assert full_bbox_inside(a, [0,0,0])
    a = actor(1, x=48.5)
    assert not full_bbox_inside(a, [0,0,0])
    assert full_bbox_inside(a, [1.5,0,2.4])
    a['bbox']['center_actor_m'][0] = 2
    assert not full_bbox_inside(a, [1.5,0,2.4])
    a = actor(1, x=47.)
    a['bbox']['extent_m'] = [4,1,1]
    assert not full_bbox_inside(a,[0,0,0])
    a['bbox']['rotation_actor_deg'][2] = 90
    assert full_bbox_inside(a,[0,0,0])


def test_policy_version_and_historical_boundary_reading():
    a=actor(1,x=49.)
    assert label_exclusion_reason(a,[0,0,0]) == 'range_boundary'
    assert label_exclusion_reason(a,[0,0,0],LEGACY_POLICY) is None
    assert 'distance_boundary' not in motion_metadata(LEGACY_POLICY)
    assert motion_metadata()['policy'] == POLICY


def test_boundary_removes_only_actor_and_recomputes_all_teachers():
    old,current,meta,anchor=inputs()
    current['actors'][0]['pose']['position_m'][0]=49.
    before=copy.deepcopy(current)
    new=attach_supervision(rebuild_agents(old,current,meta,anchor))
    validate_motion(new)
    ids={a['id'] for a in new['agents']}
    assert '0' not in ids and len(ids)==11
    assert set(new['local_agent_risk_labels'])==ids
    assert {a['id'] for a in new['risk_agents']}==ids
    assert set(new['planner_agent_ids']).issubset(ids)
    assert any(a['id']=='0' and a['reason']=='range_boundary' for a in new['background_agents'])
    assert new['ignored_agents']==[] and new['agent_ignore_policy']=='none'
    assert current==before


def test_future_outside_range_is_preserved():
    old,current,meta,anchor=inputs()
    anchor['futures_by_actor_id']['0']['world_xyzyaw']=[[100.,0.,0.,0.]]*30
    new=rebuild_agents(old,current,meta,anchor)
    assert new['agents'][0]['future_xy']==[[100.,0.]]*6
    assert all(new['agents'][0]['future30_valid'])


def test_joint_quota_and_duplicate_invariants():
    candidates=[dict(episode_id='ep',carla_frame=i,category=c,weather=w,split='train')
                for i,(c,w) in enumerate([('normal','rain')]*4+[('normal','fog')]*4+[('scenario2','rain')]*4+[('scenario2','fog')]*4)]
    joint={c+'/'+w:2 for c in ['normal','scenario2'] for w in ['rain','fog']}
    selected=choose_with_quotas(candidates,[],dict(normal=4,scenario2=4),dict(train=8),dict(rain=4,fog=4),joint)
    assert len(selected)==8
    from collections import Counter
    assert Counter(r['category']+'/'+r['weather'] for r in selected)==joint
    assert choose_with_quotas(candidates,selected,dict(normal=4,scenario2=4),dict(train=8),dict(rain=4,fog=4),joint)==[]
    with pytest.raises(ValueError,match='duplicate'):
        choose_with_quotas([],selected+[selected[0]],dict(normal=4,scenario2=4),dict(train=8),dict(rain=4,fog=4),joint)


def test_new_and_legacy_model_configs_are_accepted():
    import json
    from radar_vla.pipeline import resolve_config
    root=Path(__file__).resolve().parents[1]
    for name in ['l40s_qwen25_3b_taskwise_moving_v1.json','l40s_qwen25_3b_rain_fog_fullbbox_v2.json']:
        resolve_config(json.loads((root/'configs'/name).read_text()))


def test_current_prepare_explicit_paths_resume_and_signature(tmp_path, monkeypatch):
    import json
    from radar_vla import collection_adapter as adapter
    from radar_vla import moving_agents as motion
    raw=tmp_path/'full/batch'
    raw.mkdir(parents=True)
    records=[dict(source_index=i,carla_frame=100+i,job_id='j',episode_id='e',split='train',category='normal',weather='rain') for i in (3,4,5)]
    (raw/'accepted_anchors.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    complete=raw/'jobs/j/radar/a/attempt_001/complete.json'
    complete.parent.mkdir(parents=True)
    complete.write_text('{}')
    ep=raw/'jobs/j/episode'
    ep.mkdir()
    (ep/'metadata.json').write_text('{}')
    monkeypatch.setattr(adapter,'_product_frames',lambda _: {3:object()})
    monkeypatch.setattr(adapter,'_anchor_at',lambda *args:{})
    monkeypatch.setattr(adapter,'_load_frame',lambda *args:{})
    monkeypatch.setattr(motion,'rebuild_agents',lambda r,*args:r)
    monkeypatch.setattr(motion,'attach_supervision',lambda r:r)
    visits=[]
    def build(row,*args,**kwargs):
        assert kwargs==dict(taskwise=True)
        visits.append(row['carla_frame'])
        if len(visits)==2:
            raise KeyboardInterrupt
        return dict(sample_id=str(row['carla_frame']),scene_id='j',split='train',category='normal',weather='rain',tracking_coverage_valid=[True]*6)
    monkeypatch.setattr(adapter,'_build_one',build)
    dest=tmp_path/'prepared/current'
    with pytest.raises(KeyboardInterrupt):
        adapter.prepare(tmp_path,sample=2,raw_root=raw,output_dir=dest)
    with (dest/'manifest.partial.jsonl').open('a') as stream:
        stream.write('uncommitted garbage')
    report=adapter.prepare(tmp_path,sample=2,raw_root=raw,output_dir=dest)
    assert visits==[103,104,104]
    assert report['prepared_rows']==2
    assert [r['sample_id'] for r in map(json.loads,(dest/'manifest.jsonl').read_text().splitlines())]==['103','104']
    with (raw/'accepted_anchors.jsonl').open('a') as stream:
        stream.write('\n')
    with pytest.raises(RuntimeError,match='version changed'):
        adapter.prepare(tmp_path,sample=2,raw_root=raw,output_dir=dest)


def test_all_boundary_targets_leave_a_valid_empty_detection_frame():
    old,current,meta,anchor=inputs()
    for a in current['actors']:
        a['pose']['position_m'][0]=49.
    new=attach_supervision(rebuild_agents(old,current,meta,anchor))
    validate_motion(new)
    assert new['agents']==new['risk_agents']==new['planner_agent_ids']==[]
    assert new['local_agent_risk_labels']=={}
    assert new['task_masks']['detection']


def pipeline_module():
    import importlib.util
    path=Path(__file__).resolve().parents[1]/'scripts/rain_fog_pipeline.py'
    spec=importlib.util.spec_from_file_location('_rain_fog_test_pipeline',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_run_requires_matching_smoke_and_untampered_evidence(tmp_path,monkeypatch):
    import json
    pipeline=pipeline_module()
    monkeypatch.setattr(pipeline,'signature',lambda _: 'current')
    evidence=tmp_path/'sensor_validation.json'
    evidence.write_text('{"status":"PASS"}')
    release=dict(status='PASS',signature='current',evidence_sha256={str(evidence):pipeline.sha(evidence)})
    (tmp_path/'smoke_release.json').write_text(json.dumps(release))
    assert pipeline.verify_release({},tmp_path/'config',tmp_path)['status']=='PASS'
    evidence.write_text('{"status":"FAIL"}')
    with pytest.raises(ValueError,match='evidence changed'):
        pipeline.verify_release({},tmp_path/'config',tmp_path)
    release['signature']='older'
    (tmp_path/'smoke_release.json').write_text(json.dumps(release))
    with pytest.raises(ValueError,match='Matching PASS smoke'):
        pipeline.verify_release({},tmp_path/'config',tmp_path)


def test_resume_rejects_changed_signature_before_collection(tmp_path,monkeypatch):
    import json
    pipeline=pipeline_module()
    raw=tmp_path/'full/batch'
    raw.mkdir(parents=True)
    (raw/'input_signature.json').write_text(json.dumps(dict(signature='older')))
    monkeypatch.setattr(pipeline,'verify_release',lambda *args: {})
    monkeypatch.setattr(pipeline,'signature',lambda _: 'current')
    def unexpected(*args,**kwargs):
        raise AssertionError('Must reject before any CARLA collection')
    monkeypatch.setattr(pipeline.batch,'collect',unexpected)
    config={'pipeline':{'raw_root':str(raw),'prepared_root':str(tmp_path/'prepared/batch')}}
    with pytest.raises(ValueError,match='signature changed'):
        pipeline.full(config,tmp_path/'config',tmp_path,True)


def test_virtual_display_is_cleaned_when_carla_spawn_fails(tmp_path,monkeypatch):
    import batch
    stopped=[]
    class OwnedDisplay:
        def poll(self): return None
        def terminate(self): stopped.append('display')
        def wait(self,timeout): return 0
    server=batch.CarlaServer({'runtime':dict(host='127.0.0.1',port=15184,
        carla_server='/missing',quality_level='Epic',graphics_api='vulkan')},tmp_path)
    def display(runtime):
        server.display_process=OwnedDisplay()
        return {}
    monkeypatch.setattr(server,'display_environment',display)
    monkeypatch.setattr(batch,'port_open',lambda *args:False)
    def fail(*args,**kwargs):raise OSError('spawn failed')
    monkeypatch.setattr(batch.subprocess,'Popen',fail)
    with pytest.raises(OSError,match='spawn failed'):
        server.__enter__()
    assert stopped==['display']


def test_carla_refuses_existing_server_without_spawning_or_stopping(tmp_path,monkeypatch):
    import batch
    monkeypatch.setattr(batch,'port_open',lambda *args:True)
    def forbidden(*args,**kwargs):raise AssertionError('Must not touch existing server')
    monkeypatch.setattr(batch.subprocess,'Popen',forbidden)
    with pytest.raises(RuntimeError,match='occupied'):
        batch.CarlaServer({'runtime':dict(host='127.0.0.1',port=15184)},tmp_path).__enter__()


def test_sequential_stages_share_only_an_idle_gpu(monkeypatch):
    pipeline=pipeline_module()
    config={'runtime':dict(carla_device=0,radar_devices=[0],host='127.0.0.1',port=15184,tm_port=25184)}
    monkeypatch.setattr(pipeline.batch,'port_open',lambda *args:False)
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *args,**kwargs:'0,184,0\n')
    assert pipeline.resources(config)['ports_available']
    monkeypatch.setattr(pipeline.subprocess,'check_output',lambda *args,**kwargs:'0,12000,80\n')
    with pytest.raises(RuntimeError,match='busy'):
        pipeline.resources(config)


def test_signature_identical_through_source_symlink(tmp_path,monkeypatch):
    pipeline=pipeline_module()
    monkeypatch.setattr(pipeline.batch,'signature',lambda _: 'same-config-and-collector')
    manifest=tmp_path/'config'
    direct=pipeline.signature(manifest)
    alias=tmp_path/'alias_pipeline.py'
    alias.symlink_to(Path(pipeline.__file__).resolve())
    monkeypatch.setattr(pipeline,'__file__',str(alias))
    assert pipeline.signature(manifest)==direct
