import math

from radar_vla.collection_adapter import (_to_ego, _future, _bbox_world_xy,
                                          _coverage_from_snapshots, _could_affect_risk,
                                          _selected_anchors, _nearest_planner_ids,
                                          _product_frames)
from radar_vla.road_segments import road_segments


def _corridor():
    points = [[float(i), 0.] for i in range(16)]
    ids = [{'road_id': 6, 'section_id': 0}] * 8 + [{'road_id': 283, 'section_id': 0}] * 8
    return {'map': {'centerline_ego_m': points, 'width_m': [4.]*8+[8.]*8,
                    'valid': [True]*16, 'break_before': [False]*8+[True]+[False]*7,
                    'source_ids': ids}}


def test_route_segments_do_not_average_or_bridge_road_change():
    segments = road_segments(_corridor())
    assert [(s['ordinal'], s['road_id'], s['width_m']) for s in segments] == [(0, 6, 4.), (1, 283, 8.)]
    assert segments[0]['source_point_indices'] == list(range(8))
    assert segments[1]['source_point_indices'] == list(range(8, 16))


def test_invalid_gap_and_width_span_remain_explicit():
    source = _corridor()
    source['map']['valid'][3] = False
    source['map']['width_m'][12] = 9.
    segments = road_segments(source)
    assert [s['source_point_indices'] for s in segments] == [[0, 1, 2], [4, 5, 6, 7], list(range(8, 16))]
    assert segments[-1]['width_m'] is None and not segments[-1]['width_valid']


def test_carla_right_to_model_left_and_real_future_mask():
    assert _to_ego([1., 2.], [0., 0.], 0.) == [1., -2.]
    assert all(abs(a-b) < 1e-8 for a,b in zip(_to_ego([0., 1.], [0., 0.], 90.), [1., 0.]))
    entry = {'world_xyzyaw': [[float(i), 2., 0., 0.] for i in range(1,31)],
             'valid': [True]*30}
    entry['valid'][9] = False
    xy, yaw, valid = _future(entry, [0.,0.], 0., [4,9,29])
    assert xy == [[5.,-2.],[0.,0.],[30.,-2.]]
    assert valid == [True,False,True]
    assert all(math.isfinite(v) for v in yaw)
    assert all(abs(a-b) < 1e-8 for a,b in zip(
        _bbox_world_xy([10., 20.], 90., [2., 1.]), [9., 22.]))


def test_segmented_teacher_decode_and_gradient(tmp_path):
    import torch
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset, collate_batch
    from radar_vla.planner import (PlannerConfig, RiskConditionedPlanner,
                                   create_supervision, planner_physics_loss)

    manifest = write_synthetic_dataset(tmp_path/'synthetic', scenes_per_split=1, frames_per_scene=1)
    batch = collate_batch([RadarDataset(manifest, 'train')[0]])
    batch['road_segments'] = [[{'ordinal': 0, 'centerline_ego_m': [[0.,0.],[10.,0.],[20.,0.]],
                                'width_m': 4., 'width_valid': True},
                               {'ordinal': 1, 'centerline_ego_m': [[23.,1.],[30.,2.],[40.,4.]],
                                'width_m': 8., 'width_valid': True}]]
    config = PlannerConfig(road_format='segments', reasoning_policy='always_long')
    model = RiskConditionedPlanner(config)
    supervision = create_supervision(batch, model.tokenizer, config)
    ids = [i for i in supervision['target_ids'][0].tolist() if i != model.tokenizer.pad_id]
    assert len(supervision['rows'][0]['road']['segments']) == 2
    decoded = model.decode(ids)['road']['segments']
    assert len(decoded) == 2 and all(len(segment['centerline_points']) == 16 for segment in decoded)
    for i, value in enumerate(ids):
        assert value in model._allowed_next(ids[:i])
    logits = torch.randn(1, len(ids), model.tokenizer.vocab_size, requires_grad=True)
    loss = planner_physics_loss(logits, supervision, batch, model.tokenizer, config)['road']
    loss.backward()
    assert torch.isfinite(loss) and logits.grad.abs().sum() > 0


def test_world_actor_inventory_masks_new_entrant(tmp_path):
    import json
    current = {'ego': {'actor_id': 1}, 'actors': [{'actor_id': 2}]}
    anchor = {'futures_by_actor_id': {'1': {'valid': [True]*30},
                                     '2': {'valid': [True]*30}}}
    for step in range(1,31):
        folder = tmp_path/'frames'/f'{step:06d}'
        folder.mkdir(parents=True)
        actors = [{'actor_id': 2}] + ([{'actor_id': 3}] if step == 11 else [])
        (folder/'frame.json').write_text(json.dumps({'ego': {'actor_id': 1}, 'actors': actors}))
    assert _coverage_from_snapshots(tmp_path, 0, current, anchor) == [True, True, False, False, False, False]


def test_risk_distance_bound_keeps_near_and_drops_provably_far():
    actor = {'pose': {'position_m': [300.,0.,0.]},
             'bbox': {'extent_m': [2.,1.,1.]}}
    future = {'valid':[True]*30, 'world_xyzyaw':[[300.,0.,0.,0.]]*30}
    assert not _could_affect_risk(actor, future, [0.,0.], 10., [2.,1.,1.])
    actor['pose']['position_m'][0] = 50.
    assert _could_affect_risk(actor, future, [0.,0.], 10., [2.,1.,1.])


def test_preparation_journal_resumes_without_repeating_committed_rows(tmp_path, monkeypatch):
    import json
    import pytest
    import radar_vla.collection_adapter as adapter
    full = tmp_path/'full'
    full.mkdir()
    rows = [dict(source_index=i,carla_frame=100+i,job_id='j',episode_id='e',
                 split='train',category='normal',weather='clear_day') for i in (3,4,5)]
    (full/'accepted_anchors.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    complete = full/'jobs/j/radar/a/attempt_001/complete.json'
    complete.parent.mkdir(parents=True)
    complete.write_text('{}')
    monkeypatch.setattr(adapter, '_product_frames', lambda _: {3: object()})
    monkeypatch.setattr(adapter, '_anchor_at', lambda *_: {'carla_frame': 103})
    monkeypatch.setattr(adapter, '_attach_supervision', lambda record: record)
    seen = []
    def interrupted(row, *_):
        seen.append(row['carla_frame'])
        if len(seen) == 2:
            raise KeyboardInterrupt
        return dict(sample_id=str(row['carla_frame']),scene_id='j',split='train',
                    category='normal',weather='clear_day',tracking_coverage_valid=[True]*6)
    monkeypatch.setattr(adapter, '_build_one', interrupted)
    with pytest.raises(KeyboardInterrupt):
        adapter.prepare(tmp_path,sample=2)
    monkeypatch.setattr(adapter, '_build_one', lambda row,*_: dict(
        sample_id=str(row['carla_frame']),scene_id='j',split='train',
        category='normal',weather='clear_day',tracking_coverage_valid=[True]*6))
    report = adapter.prepare(tmp_path,sample=2)
    assert report['prepared_rows'] == 2 and report['excluded_rows'] == 0
    assert [json.loads(line)['sample_id'] for line in open(report['manifest'])] == ['103','104']


def test_full_anchor_reader_keeps_only_selected_frames(tmp_path):
    import json
    path = tmp_path/'anchors.jsonl'
    path.write_text(''.join(json.dumps({'carla_frame': i, 'large_future': 'x'*1000})+'\n'
                            for i in range(100,110)))
    assert sorted(_selected_anchors(path, {102,108})) == [102,108]
    assert _selected_anchors(path, {999}) == {}


def test_nearest_eight_are_planner_only_and_ids_break_distance_ties():
    agents = [{'id': str(i), 'position': [float(15-i), 0.]} for i in range(1, 11)]
    selected = _nearest_planner_ids(agents)
    assert selected == ['10', '9', '8', '7', '6', '5', '4', '3']
    assert len(agents) == 10  # Detection and scene Risk keep the full list.
    tied = [{'id': '12', 'position': [1., 0.]}, {'id': '2', 'position': [-1., 0.]}]
    assert _nearest_planner_ids(tied, 1) == ['2']


def test_latest_complete_radar_attempt_matches_final_index(tmp_path):
    import json
    base = tmp_path/'radar/000010_000010'
    for attempt in ('attempt_001', 'attempt_002'):
        path = base/attempt
        frame = path/'frames/000000'
        frame.mkdir(parents=True)
        (path/'complete.json').write_text(json.dumps({'core_frames': [100]}))
        (path/'radar_meta.json').write_text(json.dumps({
            'source_start_index': 10, 'shape_ra': [256,107],
            'shape_dra': [64,256,107], 'window_mode': 'hann_3d',
            'frames': [{'source_index': 10, 'carla_frame': 100}]}))
        (frame/'folded_ra.npz').touch()
        (frame/'unfolded_ra.npz').touch()
    assert _product_frames(tmp_path)[10][2].name == 'attempt_002'


def test_planner_uses_eight_slots_while_grounder_keeps_ten(tmp_path):
    import torch
    from radar_vla.synthetic import write_synthetic_dataset
    from radar_vla.data import RadarDataset, collate_batch
    from radar_vla.planner import PlannerConfig, RiskConditionedPlanner, create_supervision
    manifest = write_synthetic_dataset(tmp_path/'synthetic', scenes_per_split=1, frames_per_scene=1)
    batch = collate_batch([RadarDataset(manifest, 'train')[0]])
    for key in ('agent_mask', 'agent_state', 'agent_state_mask',
                'agent_future', 'agent_future_mask'):
        batch[key] = batch[key].repeat(1, 10, *([1] * (batch[key].ndim-2)))
    batch['critical_agent_order'] = torch.arange(10).reshape(1, 10)
    batch['agent_planner_mask'] = torch.tensor([[True]*8+[False]*2])
    batch['road_segments'] = [[{'ordinal': 0, 'centerline_ego_m': [[0.,0.],[10.,0.]],
                                'width_m': 4., 'width_valid': True}]]
    planner = RiskConditionedPlanner(PlannerConfig(road_format='segments', reasoning_policy='always_long'))
    supervision = create_supervision(batch, planner.tokenizer, planner.config)
    assert len(supervision['rows'][0]['agents']) == 8
    assert int(batch['agent_mask'].sum()) == 10


def test_completed_manifest_reuse_indexes_only_successes(tmp_path):
    import json
    from radar_vla.collection_adapter import _reuse_offsets

    previous = tmp_path / 'previous'
    previous.mkdir()
    selected = [dict(episode_id='abc123', carla_frame=100),
                dict(episode_id='abc123', carla_frame=101)]
    (previous/'report.json').write_text(json.dumps({
        'status':'PREPARED', 'source_rows':2, 'prepared_rows':1}))
    (previous/'manifest.jsonl').write_text(json.dumps({
        'sample_id':'abc123_100', 'episode_id':'abc123', 'carla_frame':100})+'\n')
    manifest, offsets = _reuse_offsets(previous, selected)
    assert manifest == previous/'manifest.jsonl'
    assert set(offsets) == {('abc123', 100)}
    assert ('abc123', 101) not in offsets  # The old exclusion is rebuilt.


def test_reused_record_copies_assets_and_selects_planner_ids(tmp_path, monkeypatch):
    import json
    import pytest
    import radar_vla.collection_adapter as adapter

    previous, output = tmp_path/'previous', tmp_path/'new'
    (previous/'radar').mkdir(parents=True)
    (previous/'radar_masks').mkdir()
    (previous/'radar/a.npy').write_bytes(b'old-power')
    (previous/'radar_masks/m.npy').write_bytes(b'old-mask')
    row = dict(episode_id='abc123', carla_frame=100, job_id='job',
               split='train', category='normal', weather='fog')
    record = dict(schema_version='radar_vla_v3', episode_id='abc123',
                  carla_frame=100, scene_id='job', split='train',
                  category='normal', weather='fog', agents=[
                      {'id':'1', 'position':[2.,0.]},
                      {'id':'2', 'position':[1.,0.]}],
                  risk_label={}, agent_risk_labels={'1':{}, '2':{}},
                  teacher_mode='long', sensors={'radar':{
                      'power':['radar/a.npy'],
                      'unfolded_doppler':['radar/a.npy'],
                      'doppler_valid':'radar_masks/m.npy'}})
    monkeypatch.setattr(adapter, '_mode_from_risk', lambda *_: 'long')
    reused = adapter._reuse_record(record, row, previous, output)
    assert reused['planner_agent_ids'] == ['2', '1']
    assert (output/'radar/a.npy').read_bytes() == b'old-power'
    (output/'radar/a.npy').write_bytes(b'new-power')
    assert (previous/'radar/a.npy').read_bytes() == b'old-power'
    with pytest.raises(ValueError, match='does not match'):
        adapter._reuse_record(json.loads(json.dumps(record)),
                              dict(row, weather='rain'), previous, tmp_path/'other')
