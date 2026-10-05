"""Offline plan evaluation must not turn unknown coverage into safety."""
import pytest


def scene():
    return dict(future_times_s=[.5, 1., 2., 3.],
                ego=dict(velocity=[2., 0.], yaw_rate=0., box_size=[2., 1.],
                         future_xy=[[1., 0.], [2., 0.], [4., 0.], [6., 0.]]),
                agents=[dict(id='crossing', position=[4., 4.], velocity=[0., -2.],
                             heading=1.57079632679, size=[2., 1.],
                             future_xy=[[4., 3.], [4., 2.], [4., 0.], [4., -2.]],
                             future_yaw=[1.57079632679] * 4, future_valid=[True] * 4)])


def test_crossing_collision_uses_boxes_and_future_motion():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    plan = dict(valid=True, ego_trajectory=record['ego']['future_xy'], ego_times_s=record['future_times_s'])
    result = evaluate_plan(record, plan)
    assert result['collision'] == 1
    assert result['first_safety_violation_s'] <= 2
    assert result['minimum_ttc_s'] is not None
    assert result['ego_ade_m'] == 0
    assert result['ego_final_horizon_fde_m'] == 0
    assert result['route_completion'] is None


def test_missing_agent_futures_cannot_be_scored_as_safe():
    from radar_vla.evaluation import evaluate_plan, aggregate_plan_metrics
    record = scene()
    record['agents'][0]['future_valid'] = [False] * 4
    plan = dict(valid=True, ego_trajectory=[[0., 0.]] * 4, ego_times_s=record['future_times_s'])
    result = evaluate_plan(record, plan)
    assert result['collision'] is None
    assert result['safety_coverage'] < 1
    aggregate = aggregate_plan_metrics([result])
    assert aggregate['collision_rate'] is None
    assert aggregate['collision_rate_count'] == 0


def test_short_response_has_no_full_horizon_fde_or_route_success():
    from radar_vla.evaluation import evaluate_plan
    result = evaluate_plan(scene(), dict(valid=True, ego_trajectory=[[1., 0.]], ego_times_s=[.5]))
    assert result['ego_final_horizon_fde_m'] is None
    assert result['ego_point_coverage'] == .25
    assert result['planned_horizon_s'] == .5


def test_nonuniform_sampling_constant_velocity_has_zero_jerk():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['agents'] = []
    times = [.1, .4, 1., 3.]
    result = evaluate_plan(record, dict(valid=True, ego_trajectory=[[2 * t, 0.] for t in times], ego_times_s=times))
    assert result['jerk_rms_mps3'] == pytest.approx(0, abs=1e-10)
    assert result['collision'] == 0
    assert result['brake_onset_s'] is None
    assert result['hard_brake_fraction'] == 0


def test_ttc_respects_box_extent_and_receding_motion():
    from radar_vla.evaluation import translational_box_ttc
    assert translational_box_ttc([0, 0], 0, [2, 1], [2, 0], [10, 0], 0, [2, 1], [0, 0]) == pytest.approx(4.)
    assert translational_box_ttc([0, 0], 0, [2, 1], [0, 0], [10, 0], 0, [2, 1], [2, 0]) is None


def test_route_progress_only_uses_explicit_route_and_declared_goal():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['map'] = {'route': {'centerline': [[0, 0], [10, 0]], 'goal_s_m': 10.}}
    result = evaluate_plan(record, dict(valid=True, ego_trajectory=record['ego']['future_xy'], ego_times_s=record['future_times_s']))
    assert result['route_completion'] == pytest.approx(.6)


def test_unknown_tracking_coverage_never_reports_an_empty_scene_as_safe():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['agents'] = []
    record['agent_supervision_available'] = False
    result = evaluate_plan(record, dict(valid=True, ego_trajectory=[[0., 0.]] * 4, ego_times_s=record['future_times_s']))
    assert result['collision'] is None
    assert result['safety_coverage'] == 0


def test_hard_braking_is_measured_in_time_not_waypoint_index():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['agents'] = []
    record['ego']['velocity'] = [10., 0.]
    times = [.1, .25, .5, 1.]
    # Constant physical deceleration -4m/s² has nonuniform waypoint distances.
    xy = [[10 * t - 2 * t * t, 0.] for t in times]
    result = evaluate_plan(record, dict(valid=True, ego_trajectory=xy, ego_times_s=times))
    assert result['hard_brake_fraction'] == pytest.approx(1.)
    assert result['brake_onset_s'] == 0.
    assert result['jerk_rms_mps3'] == pytest.approx(0., abs=1e-8)


def test_schema_3d_route_is_projected_to_ground_plane_and_null_map_is_allowed():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    plan = dict(valid=True, ego_trajectory=record['ego']['future_xy'], ego_times_s=record['future_times_s'])
    record['map'] = {'route': {'centerline': [[0., 0., 3.], [10., 0., 4.]], 'goal_s_m': 10.}}
    assert evaluate_plan(record, plan)['route_completion'] == pytest.approx(.6)
    record['map'] = None
    assert evaluate_plan(record, plan)['route_completion'] is None


def test_future_tracking_coverage_is_required_for_negative_collision_labels():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['agents'] = []
    record['tracking_coverage_valid'] = [True, False, False, False]
    result = evaluate_plan(record, dict(valid=True, ego_trajectory=[[0., 0.]] * 4, ego_times_s=record['future_times_s']))
    assert result['collision'] is None
    assert 0 < result['safety_coverage'] < 1
    record.pop('tracking_coverage_valid')
    record['schema_version'] = 'radar_vla_v2'
    assert evaluate_plan(record, dict(valid=True, ego_trajectory=[[0., 0.]] * 4, ego_times_s=record['future_times_s']))['collision'] is None


def test_future_entrant_is_scored_without_inventing_a_present_box():
    from radar_vla.evaluation import evaluate_plan
    record = scene()
    record['agents'] = []
    entrant = dict(id='future_entrant', position=[0., 0.], velocity=[0., 0.],
                   heading=0., size=[2., 1.], present_valid=False,
                   future_xy=[[20., 0.], [15., 0.], [4., 0.], [10., 0.]],
                   future_yaw=[0.] * 4, future_valid=[False, True, True, True])
    record['risk_agents'] = [entrant]
    record['tracking_coverage_valid'] = [True] * 4
    plan = dict(valid=True, ego_trajectory=record['ego']['future_xy'], ego_times_s=record['future_times_s'])
    result = evaluate_plan(record, plan)
    assert result['collision'] == 1
    assert 1 < result['first_safety_violation_s'] <= 2


def test_collision_evaluation_samples_the_actual_spline_between_output_waypoints():
    from radar_vla.evaluation import evaluate_plan
    record = dict(future_times_s=[1., 2., 3.],
                  ego=dict(velocity=[2., 0.], box_size=[.2, .2], yaw_rate=0.),
                  agents=[dict(id='bend', position=[2., 4.], velocity=[0., 0.], heading=0., size=[.2, .2],
                               future_xy=[[2., 4.]] * 3, future_yaw=[0.] * 3, future_valid=[True] * 3)])
    # Sparse output only includes the final point; shared spline bends through
    # the obstacle at t=1. A straight interpolation would miss it completely.
    plan = dict(valid=True, ego_trajectory=[[6., 0.]], ego_times_s=[3.],
                ego_control_points=[[0., 0.], [2., 4.], [4., 4.], [6., 0.]],
                control_times_s=[0., 1., 2., 3.])
    result = evaluate_plan(record, plan)
    assert result['collision'] == 1
    assert result['first_safety_violation_s'] <= 1.
    assert result['trajectory_sampling'].startswith('shared natural cubic spline')
