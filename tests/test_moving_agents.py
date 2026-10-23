"""Motion-only membership, full target sets and background penalties."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from radar_vla.moving_agents import exclusion_reason, rebuild_agents, attach_supervision, POLICY


@pytest.mark.parametrize('speed,kind,reason', [
    (0., 'vehicle.test', 'stationary'), (.099999, 'vehicle.test', 'stationary'),
    (.1, 'vehicle.test', None), (-.1, 'vehicle.test', None),
    (2., 'walker.pedestrian.0001', 'pedestrian'),
    (2., 'vehicle.bh.crossbike', 'cyclist'),
    (2., 'vehicle.kawasaki.ninja', None)])
def test_current_world_speed_threshold(speed, kind, reason):
    assert exclusion_reason(dict(type_id=kind, velocity_world_mps=[speed, 0., 0.])) == reason


def actor(identifier, speed=1., kind='vehicle.test', x=10.):
    return dict(actor_id=identifier, type_id=kind, actor_type='pedestrian' if kind.startswith('walker') else 'vehicle',
        role_name='', velocity_world_mps=[speed, 0., 0.],
        pose=dict(position_m=[x, 0., 0.], rotation_deg=[0., 0., 0.]),
        bbox=dict(center_actor_m=[0., 0., 0.], rotation_actor_deg=[0., 0., 0.], extent_m=[2., 1., 1.]))


def inputs():
    from test_local_risk import record
    actors = [actor(i, x=8.+i) for i in range(12)]
    actors += [actor(90, 0.), actor(91, 1., 'walker.pedestrian.0001'), actor(92, 1., 'vehicle.bh.crossbike')]
    current = dict(episode_id='ep', carla_frame=100, actors=actors,
        ego=dict(pose=dict(position_m=[0., 0., 0.], rotation_deg=[0., 0., 0.])))
    meta = dict(episode_id='ep', agent_label_range_m=50,
                sensors=dict(semantic_lidar=dict(x=0., y=0., z=0., roll=0., pitch=0., yaw=0.)))
    futures = {str(a['actor_id']):dict(valid=[True]*30,
        world_xyzyaw=[[a['pose']['position_m'][0] + a['velocity_world_mps'][0]*i/10, 0., 0., 0.] for i in range(1, 31)]) for a in actors}
    anchor = dict(carla_frame=100, futures_by_actor_id=futures)
    old = record([])
    old.update(episode_id='ep', carla_frame=100, road_segments=[], road_supervision_available=True)
    return old, current, meta, anchor


def test_full_set_restoration_no_distance_count_cap():
    old, current, meta, anchor = inputs()
    before = copy.deepcopy(current)
    new = attach_supervision(rebuild_agents(old, current, meta, anchor))
    assert len(new['agents']) == len(new['local_agent_risk_labels']) == len(new['risk_agents']) == 12
    assert len(new['planner_agent_ids']) == 8
    assert new['ignored_agents'] == [] and new['agent_ignore_policy'] == 'none'
    assert {a['reason'] for a in new['background_agents']} == {'stationary', 'pedestrian', 'cyclist'}
    assert current == before
    from radar_vla.task_supervision import validate_tasks
    validate_tasks(new)


def test_current_membership_keeps_future_stop_and_missing_masks():
    old, current, meta, anchor = inputs()
    anchor['futures_by_actor_id']['0']['world_xyzyaw'] = [[8., 0., 0., 0.]]*30
    anchor['futures_by_actor_id']['1']['valid'][5] = False
    new = rebuild_agents(old, current, meta, anchor)
    assert new['agents'][0]['future_xy'] == [[8., 0.]]*6
    assert not new['agents'][1]['future30_valid'][5]
    current['actors'][0]['velocity_world_mps'] = [0., 0., 0.]
    assert '0' not in {a['id'] for a in rebuild_agents(old, current, meta, anchor)['agents']}
    current['actors'][0]['velocity_world_mps'] = [.1, 0., 0.]
    assert '0' in {a['id'] for a in rebuild_agents(old, current, meta, anchor)['agents']}


def test_unknown_annotation_is_not_empty_negative():
    old, current, meta, anchor = inputs()
    old['agent_supervision_available'] = False
    current['actors'] = []
    new = attach_supervision(rebuild_agents(old, current, meta, anchor))
    assert not new['task_masks']['detection'] and new['mode_label'] == 'UNKNOWN'


def test_background_detections_are_false_positives():
    from radar_vla.detection_metrics import DetectionMetrics
    batch = dict(agent_mask=torch.zeros(1, 0, dtype=torch.bool), agent_state=torch.zeros(1, 0, 5),
        agent_future=torch.zeros(1, 0, 6, 2), agent_future_mask=torch.zeros(1, 0, 6, dtype=torch.bool),
        agent_supervision_mask=torch.tensor([True]), detection_ignore_boxes=torch.zeros(1, 0, 5),
        detection_ignore_mask=torch.zeros(1, 0, dtype=torch.bool))
    detections = [dict(position_m=[x, 0.], size_m=[4., 2.], heading_rad=0., category='agent') for x in (10., 20., 30.)]
    metric = DetectionMetrics(class_schema='agent_v1')
    metric.update([dict(detections=detections, detection_status='available', detection_truncated=False)], batch)
    assert metric.fp == 3 and metric.ignored_removed_agents == 0


def test_background_objectness_has_positive_logit_gradient():
    from radar_vla.losses import loss_grounding
    # Canonical model/loss: no positives and no ignored boxes means every
    # in-domain objectness prediction must be penalized as background.
    from radar_vla.model import RadarVLAGrounder, ModelConfig
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset, collate_batch
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        manifest = write_synthetic_dataset(Path(temp), scenes_per_split=1, frames_per_scene=1, frames=4, range_bins=16, azimuth_bins=17)
        r = json.loads(manifest.read_text().splitlines()[0]); r['agents'] = []
        p = Path(temp)/'empty.jsonl'; p.write_text(json.dumps(r)+'\n')
        b = collate_batch([RadarDataset(p, r['split'])[0]])
        model = RadarVLAGrounder(ModelConfig(hidden_dim=16, num_heads=2))
        output = model(b)
        key = 'detection_logits'
        assert key in output
        output[key].retain_grad()
        loss_grounding(output, b)['detection_objectness'].backward()
        assert (output[key].grad > 0).any()


def test_config_rejects_caps_and_policy_mismatch():
    from radar_vla.pipeline import resolve_config, validate_input_shapes
    project = Path(__file__).resolve().parents[1]
    cfg = json.loads((project/'configs/l40s_qwen25_3b_taskwise_moving_v1.json').read_text())
    resolve_config(cfg)
    capped = copy.deepcopy(cfg); capped['model']['detection_max_objects'] = 8
    with pytest.raises(ValueError, match='without count caps'):
        resolve_config(capped)
    class Dataset:
        records = [dict(agent_selection_policy=POLICY)]
        supervision_protocol='taskwise_local_v1'
        risk_schema='local_agent_collision_v1'
        class_schema='agent_v1'
    old = copy.deepcopy(cfg); old['data'].pop('agent_selection_policy')
    with pytest.raises(ValueError, match='selection policy'):
        validate_input_shapes(Dataset(), old)


def test_layout_preserves_old_paths_and_manifest(tmp_path):
    from radar_vla.moving_migration import organize_existing
    old = tmp_path/'analysis/taskwise_local_v1'; old.mkdir(parents=True)
    r = dict(route_archive='/raw/route.json', sensors=dict(radar=dict(power=['/raw/p.npy'], unfolded_doppler=['/raw/d.npy'], doppler_valid='/raw/m.npy')))
    content=json.dumps(r)+'\n'; (old/'manifest.jsonl').write_text(content)
    (old/'REPORT.md').write_text('report'); (old/'radar').mkdir()
    manifest=organize_existing(tmp_path)
    assert manifest.read_text() == (old/'manifest.jsonl').read_text() == content
    assert (old/'manifest.jsonl').is_symlink() and (old/'REPORT.md').is_file()
    assert (manifest.parent/'REPORT.md').is_symlink()
    assert organize_existing(tmp_path) == manifest


def test_migration_resumes_committed_rows_and_rejects_changed_source(tmp_path, monkeypatch):
    from radar_vla import moving_migration as migration
    import radar_vla.data as data
    source = tmp_path/'source.jsonl'
    baseline = [dict(sample_id=str(i), episode_id='ep', split='train', agents=[], background_agents=[],
                     mode_label='LONG', sft_available=True) for i in range(2)]
    source.write_text(''.join(json.dumps(r)+'\n' for r in baseline))
    class Evidence:
        def __init__(self, source): self.hashes={}
        def rebuild(self, r): return copy.deepcopy(r)
    monkeypatch.setattr(migration, 'Evidence', Evidence)
    monkeypatch.setattr(data, '_validate_record', lambda *args: None)
    def interrupt(r):
        if r['sample_id']=='1': raise RuntimeError('interrupted')
        return r
    monkeypatch.setattr(migration, 'attach_supervision', interrupt)
    config = Path(__file__).resolve().parents[1]/'configs/l40s_qwen25_3b_taskwise_moving_v1.json'
    out, reports = tmp_path/'prepared', tmp_path/'reports'
    with pytest.raises(RuntimeError, match='interrupted'):
        migration.migrate(source, out, reports, config)
    # Simulate an uncommitted write left by process death.
    with (out/'manifest.partial.jsonl').open('a') as stream: stream.write('partial garbage')
    monkeypatch.setattr(migration, 'attach_supervision', lambda r:r)
    migration.migrate(source, out, reports, config)
    assert [r['sample_id'] for r in migration.rows(out/'manifest.jsonl')] == ['0','1']
    source.write_text(source.read_text()+'\n')
    with pytest.raises(ValueError, match='Source/code/config changed'):
        migration.migrate(source, out, reports, config)
