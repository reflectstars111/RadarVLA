import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def test_xy_to_ra_and_matching():
    from radar_vla.visual_diagnosis import match_detections, xy_to_ra
    azimuth, radius = xy_to_ra([[10., 0.], [0., 10.]])
    assert np.allclose(azimuth, [0., 90.])
    assert np.allclose(radius, [10., 10.])
    record = {'agents': [{'position': [10., 0.]}, {'position': [20., 0.]}]}
    predictions = [{'position_m': [10.5, 0.]}, {'position_m': [25., 0.]}]
    result = match_detections(record, predictions)
    assert (result['tp'], result['fp'], result['fn']) == (1, 1, 1)
    assert result['ignored_predictions'] == []


def test_matching_ignores_unmatched_predictions_outside_explicit_domain():
    from radar_vla.visual_diagnosis import match_detections
    record = {'agents': [], 'detection_annotation_domain': {
        'type': 'radar_bev_sector', 'range_m': 50.,
        'azimuth_min_rad': -.925, 'azimuth_max_rad': .925}}
    predictions = [{'position_m': [10., 0.], 'size_m': [2., 2.], 'heading_rad': 0.},
                   {'position_m': [80., 0.], 'size_m': [2., 2.], 'heading_rad': 0.}]
    result = match_detections(record, predictions, radar_pose=np.eye(4))
    assert result['fp'] == 1 and result['ignored_predictions'] == [1]


def test_manifest_seek_index_excludes_test(tmp_path):
    from radar_vla.visual_diagnosis import manifest_index, read_record
    path = tmp_path / 'manifest.jsonl'
    rows = []
    for split in ('val', 'test'):
        rows.append({'sample_id': split, 'episode_id': f'e-{split}', 'scene_id': f's-{split}',
                     'split': split, 'agents': [], 'risk_label': {'pcol': [0, 0, 0],
                     'pcol_mask': [True, True, True]}, 'language': {'instruction': 'follow lane'}})
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    index = manifest_index(path)
    assert [row['sample_id'] for row in index] == ['val']
    assert read_record(path, index[0]['offset'])['split'] == 'val'


def test_unfolded_doppler_render_path_does_not_apply_validity_mask(monkeypatch):
    """The visualizer reads channel 1 directly; no validity field reaches render."""
    from radar_vla import visual_diagnosis as visual
    captured = []
    original = matplotlib.axes.Axes.imshow
    def spy(self, value, *args, **kwargs):
        captured.append(np.asarray(value).copy())
        return original(self, value, *args, **kwargs)
    monkeypatch.setattr(matplotlib.axes.Axes, 'imshow', spy)
    monkeypatch.setattr(visual, '_decoded', lambda result, threshold: {'detections': []})
    record = {'sample_id': 'x', 'episode_id': 'e', 'scene_id': 's', 'agents': [],
              'road_segments': [], 'language': {'instruction': 'follow lane'},
              'risk_label': {'pcol': [0, 0, 0]}}
    radar = np.zeros((1, 2, 2, 2), np.float32)
    radar[0, 0] = [[1., 2.], [3., 4.]]
    radar[0, 1] = [[-7., 8.], [9., -10.]]
    import torch
    sample = {'radar': torch.from_numpy(radar), 'range_m': torch.tensor([1., 2.]),
              'azimuth_rad': torch.tensor([-.1, .1]),
              'radar_pose': torch.eye(4)[None]}
    output = {'risk': torch.zeros((1, 5)), 'detection_future': torch.zeros((1, 1, 1, 2))}
    times = torch.tensor([.5])
    grounding = {'output': output, 'epoch': 1, 'future_times_s': times}
    sft = {'output': output, 'epoch': 1, 'future_times_s': times,
           'plan': {'mode': 'short', 'valid': True, 'errors': []}}
    fig = visual.render(record, sample, grounding, sft)
    assert np.array_equal(captured[1], radar[0, 1])
    plt.close(fig)


def test_ego_to_radar_projection_uses_mount_translation_and_yaw():
    from radar_vla.visual_diagnosis import ego_to_radar_xy, _plot_ra_line, _draw_domain_bev
    from radar_vla.coordinates import transform_points
    angle = .35
    pose = np.eye(4)
    pose[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
    pose[:2,3] = [2., .4]
    local = np.array([[10.,0.],[20.,5.]])
    world = transform_points(local, pose)[:,:2]
    assert np.allclose(ego_to_radar_xy(world, pose), local)
    fig, (ra, bev) = plt.subplots(1,2)
    _plot_ra_line(ra, world, radar_pose=pose)
    assert np.allclose(ra.lines[0].get_xdata(), np.degrees(np.arctan2(local[:,1],local[:,0])))
    assert np.allclose(ra.lines[0].get_ydata(), np.linalg.norm(local,axis=1))
    _draw_domain_bev(bev,radar_pose=pose)
    assert np.allclose([bev.lines[1].get_ydata()[0],bev.lines[1].get_xdata()[0]],pose[:2,3])
    plt.close(fig)
