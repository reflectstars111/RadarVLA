"""Reward-only future annotation stitching never changes actor observations."""
import copy
import numpy as np
import pytest


def _row(stamp=0., pose_x=0., scene='scene', split='train'):
    pose = np.eye(4); pose[0, 3] = pose_x
    return dict(sample_id=f'{scene}_{stamp}', scene_id=scene, split=split, timestamp_s=stamp,
        coordinate_frame='current_ego', future_times_s=[1., 2., 3.],
        ego=dict(pose=pose.tolist(), velocity=[2., 0.], box_size=[1., .5],
                 future_xy=[[2., 0.], [4., 0.], [6., 0.]], future_valid=[True]*3),
        agents=[dict(id='obstacle', position=[10.-pose_x, 0.], velocity=[0., 0.],
                     heading=0., size=[1., .5], future_xy=[[10.-pose_x, 0.]]*3,
                     future_yaw=[0.]*3, future_valid=[True]*3)],
        agent_supervision_available=True, tracking_coverage_valid=[True]*3)


def test_reward_annotations_extend_world_consistent_gt_without_touching_actor():
    from radar_vla.response_annotations import reward_record
    first, later = _row(), _row(.5, 1.)
    original = copy.deepcopy(first)
    reward = reward_record(first, [first, later])
    assert first == original
    assert reward['future_times_s'] == [1., 2., 3., 3.5]
    np.testing.assert_allclose(reward['risk_agents'][0]['future_xy'][-1], [10., 0.])
    np.testing.assert_allclose(reward['ego']['future_xy'][-1], [7., 0.])
    assert reward['tracking_coverage_valid'] == [True]*4
    assert reward['response_annotation_provenance']['actor_horizon_s'] == 3.


def test_reward_annotations_never_bridge_unobserved_time_gaps():
    from radar_vla.response_annotations import reward_record
    from radar_vla.evaluation import _interpolate
    first, later = _row(), _row(5., 10.)
    reward = reward_record(first, [first, later])
    times = np.r_[0., reward['future_times_s']]
    valid = np.r_[True, reward['tracking_coverage_valid']]
    _, covered = _interpolate(times, np.zeros(len(times)), valid, [3.5, 4., 4.5])
    assert not covered.any()


def test_annotation_stitching_preserves_false_masks_and_unknown_entrants():
    from radar_vla.response_annotations import reward_record
    first, later = _row(), _row(.5, 1.)
    later['tracking_coverage_valid'][-1] = False
    entrant = copy.deepcopy(later['agents'][0]); entrant['id'] = 'entrant'
    later['agents'].append(entrant)
    reward = reward_record(first, [first, later])
    assert not reward['tracking_coverage_valid'][-1]
    entrant = next(a for a in reward['risk_agents'] if a['id'] == 'entrant')
    assert entrant['present_valid'] is False
    assert entrant['future_valid'][:3] == [True]*3  # observed by the later row
    assert entrant['future_valid'][-1]


def test_no_pose_no_stitch_and_other_scenes_or_splits_are_excluded():
    from radar_vla.response_annotations import reward_record
    first = _row()
    assert reward_record(first, [_row(.5, 1., scene='other'), _row(.5, 1., split='test')])['future_times_s'] == [1., 2., 3.]
    first['ego'].pop('pose')
    assert reward_record(first, [_row(.5, 1.)])['future_times_s'] == [1., 2., 3.]


def test_explicit_longer_reward_annotations_work_without_pose():
    from radar_vla.response_annotations import reward_record
    first = _row(); first['ego'].pop('pose')
    payload = copy.deepcopy(first)
    payload['future_times_s'].append(4.)
    payload['tracking_coverage_valid'].append(True)
    payload['ego']['future_xy'].append([8., 0.]); payload['ego']['future_valid'].append(True)
    payload['agents'][0]['future_xy'].append([10., 0.])
    payload['agents'][0]['future_yaw'].append(0.)
    payload['agents'][0]['future_valid'].append(True)
    first['response_evaluation'] = payload
    result = reward_record(first, [])
    assert result['future_times_s'][-1] == 4.
    assert len(first['future_times_s']) == 3
    assert result['response_annotation_provenance']['source'] == 'explicit_response_evaluation'
    first['response_evaluation']['agents'][0]['future_valid'].pop()
    with pytest.raises(ValueError, match='shape|align'):
        reward_record(first, [])


def test_explicit_reward_payload_cannot_rewrite_existing_gt_or_omit_a_track():
    from radar_vla.response_annotations import reward_record
    first = _row()
    payload = copy.deepcopy(first)
    payload['agents'][0]['future_xy'][0] = [99., 0.]
    first['response_evaluation'] = payload
    with pytest.raises(ValueError, match='preserve'):
        reward_record(first, [])
    payload['agents'] = []
    with pytest.raises(ValueError, match='track'):
        reward_record(first, [])
