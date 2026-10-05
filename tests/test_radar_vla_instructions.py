import pytest


def test_instruction_constraints_use_physical_times_and_do_not_guess_text():
    from radar_vla.instruction_metrics import evaluate_instruction
    plan = {'valid': True, 'ego_trajectory': [[1., 0.], [2., 0.]], 'ego_times_s': [.5, 1.]}
    assert evaluate_instruction({'language': {'instruction': 'stop'}}, plan)['success'] is None
    record = {'language': {'constraints': [
        {'type': 'speed_limit', 'max_mps': 1.},
        {'type': 'goal_region', 'at_s': 1., 'min_xy': [1.9, -.1], 'max_xy': [2.1, .1]}]}}
    result = evaluate_instruction(record, plan)
    assert result['success'] is False
    assert [c['passed'] for c in result['checks']] == [False, True]


def test_short_plan_cannot_claim_instruction_completed_after_its_horizon():
    from radar_vla.instruction_metrics import evaluate_instruction
    record = {'language': {'constraints': [{'type': 'stop_by', 'time_s': 3., 'max_mps': .1}]}}
    result = evaluate_instruction(record, {'valid': True, 'ego_trajectory': [[0., 0.]], 'ego_times_s': [1.]})
    assert result['success'] is None
    assert result['coverage'] == 0


def test_instruction_evaluation_uses_spline_velocity_and_rejects_unknown_constraints():
    from radar_vla.instruction_metrics import evaluate_instruction
    plan = {'valid': True, 'ego_trajectory': [[1., 0.], [2., 0.]], 'ego_times_s': [.5, 1.],
            'ego_control_points': [[0., 0.], [1., 0.], [2., 0.]], 'control_times_s': [0., .5, 1.]}
    record = {'language': {'constraints': [{'type': 'stop_by', 'time_s': .5, 'max_mps': .1}]}}
    assert evaluate_instruction(record, plan)['success'] is False
    record['language']['constraints'][0]['type'] = 'guess_language'
    with pytest.raises(ValueError, match='Unsupported'):
        evaluate_instruction(record, plan)


def test_default_constraint_window_is_fixed_by_record_not_generated_plan_length():
    from radar_vla.instruction_metrics import evaluate_instruction
    record = {'future_times_s': [.5, 1., 2., 3.],
              'language': {'constraints': [{'type': 'speed_limit', 'max_mps': 3.}]}}
    short = {'valid': True, 'ego_trajectory': [[1., 0.]], 'ego_times_s': [1.]}
    result = evaluate_instruction(record, short)
    assert result['success'] is None
    assert result['coverage'] == 0


def test_explicit_vehicle_heading_is_distinct_from_reverse_motion_direction():
    from radar_vla.instruction_metrics import evaluate_instruction
    record = {'language': {'constraints': [{'type': 'heading_at', 'at_s': 1., 'yaw_rad': 0., 'tolerance_rad': .1}]}}
    plan = {'valid': True, 'ego_trajectory': [[-1., 0.]], 'ego_times_s': [1.], 'ego_yaw': [0.]}
    assert evaluate_instruction(record, plan)['success'] is True
    del plan['ego_yaw']
    assert evaluate_instruction(record, plan)['success'] is None
