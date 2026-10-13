"""Regression cases for persistent, annotation-only response episodes."""
import copy
import json

import pytest


def frame(index, x=None, *, present=True, horizon=3., split='train', scene='scene'):
    times = [horizon / 2, horizon]
    agents = [] if x is None else [dict(
        id='target', position=[x, 0.], velocity=[0., 0.], heading=0., size=[1., .5],
        present_valid=present, future_xy=[[x, 0.]] * 2,
        future_valid=[True] * 2, future_yaw=[0.] * 2)]
    return dict(sample_id=f'{scene}_{index}', scene_id=scene, split=split,
                timestamp_s=index * .1, agents=agents,
                ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
                future_times_s=times, tracking_coverage_valid=[True] * 2,
                agent_supervision_available=True)


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def prepare(tmp_path, records, **kwargs):
    from radar_vla.response_events import load_events, prepare_response_events
    manifest = write_rows(tmp_path / 'frames.jsonl', records)
    output = tmp_path / 'events.jsonl'
    prepare_response_events(manifest, output, **kwargs)
    return load_events(output, records)


def test_already_visible_safe_target_becomes_its_own_hazard_episode(tmp_path):
    # A first-appearance-only builder loses the newly hazardous third frame.
    rows = [frame(0, 14.4), frame(1, 14.), frame(2, 13.6), frame(3, 13.2)]
    events = prepare(tmp_path, rows)
    assert len(events) == 1
    assert events[0]['event_kind'] == 'hazard_transition'
    assert events[0]['trigger_sample_id'] == 'scene_2'
    assert events[0]['trigger_timestamp_s'] == pytest.approx(.2)
    assert events[0]['hazard'] is True


def test_harmless_appearance_does_not_swallow_later_hazard(tmp_path):
    rows = [frame(0), frame(1, 14.4), frame(2, 14.), frame(3, 13.6), frame(4, 13.2)]
    events = prepare(tmp_path, rows)
    assert [(e['event_kind'], e['trigger_sample_id'], e['hazard']) for e in events] == [
        ('appearance', 'scene_1', False), ('hazard_transition', 'scene_3', True)]
    assert events[0]['episode_index'] != events[1]['episode_index']


def test_multiple_safe_to_hazard_transitions_for_same_track(tmp_path):
    rows = [frame(0, 15.), frame(1, 13.), frame(2, 12.), frame(3, 15.), frame(4, 13.)]
    events = prepare(tmp_path, rows)
    assert [e['trigger_sample_id'] for e in events] == ['scene_1', 'scene_4']
    assert [e['event_kind'] for e in events] == ['hazard_transition', 'hazard_transition']


def test_reappearance_requires_observed_absence_and_filters_initial_invalid_track(tmp_path):
    rows = [frame(0, 20., present=False), frame(1, 20.), frame(2), frame(3, 20.)]
    events = prepare(tmp_path, rows)
    assert [(e['event_kind'], e['trigger_sample_id']) for e in events] == [
        ('appearance', 'scene_1'), ('reappearance', 'scene_3')]


def test_unknown_absence_cannot_trigger_reappearance(tmp_path):
    rows = [frame(0, 20.), frame(1), frame(2, 20.)]
    rows[1]['agent_supervision_available'] = False
    from radar_vla.response_events import prepare_response_events
    manifest = write_rows(tmp_path / 'frames.jsonl', rows)
    with pytest.raises(ValueError, match='No .*events'):
        prepare_response_events(manifest, tmp_path / 'events.jsonl')


def test_unavailable_current_annotations_cannot_create_an_appearance(tmp_path):
    rows = [frame(0), frame(1, 20.)]
    rows[1]['agent_supervision_available'] = False
    from radar_vla.response_events import prepare_response_events
    manifest = write_rows(tmp_path / 'frames.jsonl', rows)
    with pytest.raises(ValueError, match='No .*events'):
        prepare_response_events(manifest, tmp_path / 'events.jsonl')


def test_initially_visible_hazard_has_explicit_hazard_start(tmp_path):
    events = prepare(tmp_path, [frame(0, 10.), frame(1, 9.6)])
    assert events[0]['event_kind'] == 'hazard_start'
    assert events[0]['trigger_sample_id'] == 'scene_0'


def test_partial_future_can_confirm_hazard_without_claiming_full_coverage(tmp_path):
    events = prepare(tmp_path, [frame(0, 4., horizon=1.), frame(1, 3.6, horizon=1.)])
    assert events[0]['hazard'] is True
    assert events[0]['annotation_horizon_s'] == pytest.approx(1.)
    assert events[0]['requested_safety_horizon_s'] == pytest.approx(3.)
    assert events[0]['annotation_coverage_complete'] is False


def test_partial_labels_cannot_turn_appearance_into_a_safe_control(tmp_path):
    events = prepare(tmp_path, [frame(0), frame(1, 20., horizon=1.)])
    assert events[0]['hazard'] is None
    assert events[0]['annotation_coverage_complete'] is False


def test_explicit_reward_labels_can_annotate_hazard_beyond_actor_horizon(tmp_path):
    rows = [frame(0, 8., horizon=1.), frame(1, 7.6, horizon=1.)]
    for row in rows:
        payload = copy.deepcopy(row)
        times = [.5, 1., 1.5, 2., 2.5, 3.]
        payload['future_times_s'] = times
        payload['tracking_coverage_valid'] = [True] * 6
        agent = payload['agents'][0]
        agent.update(future_xy=[agent['position']] * 6, future_valid=[True] * 6,
                     future_yaw=[0.] * 6)
        row['response_evaluation'] = payload
    events = prepare(tmp_path, rows)
    assert events[0]['trigger_sample_id'] == 'scene_0'
    assert events[0]['hazard'] is True
    assert events[0]['annotation_horizon_s'] == 3.
    assert events[0]['annotation_coverage_complete'] is True
    assert rows[0]['future_times_s'] == [.5, 1.]


def test_stitched_reward_labels_can_annotate_initial_hazard_with_short_actor_horizon(tmp_path):
    rows = [frame(i, 8. - .4 * i, horizon=1.) for i in range(21)]
    for i, row in enumerate(rows):
        row['ego']['pose'] = [[1., 0., 0., .4*i], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]]
    events = prepare(tmp_path, rows)
    assert events[0]['trigger_sample_id'] == 'scene_0'
    assert events[0]['hazard'] is True
    assert events[0]['annotation_horizon_s'] == 3.
    assert rows[0]['future_times_s'] == [.5, 1.]


def test_unknown_future_does_not_reset_existing_hazard_episode(tmp_path):
    rows = [frame(0, 10.), frame(1, 10.), frame(2, 10.)]
    rows[1]['agents'][0]['future_valid'] = [False, False]
    events = prepare(tmp_path, rows)
    assert len(events) == 1


@pytest.mark.parametrize('position', [.5, 1.5])
def test_current_danger_with_no_future_annotations_still_starts_hazard(tmp_path, position):
    rows = [frame(0, position), frame(1, position)]
    for row in rows:
        row['agents'][0]['future_valid'] = [False, False]
        row['tracking_coverage_valid'] = [False, False]
    events = prepare(tmp_path, rows)
    assert events[0]['hazard'] is True
    assert events[0]['annotation_coverage_complete'] is False
    assert events[0]['annotation_horizon_s'] == 0.


def test_known_safe_transition_ends_hazard_episode_without_restarting_clock(tmp_path):
    rows = [frame(0, 10.), frame(1, 10.), frame(2, 20.), frame(3, 20.)]
    events = prepare(tmp_path, rows)
    assert len(events) == 1
    assert events[0]['sample_ids'] == ['scene_0', 'scene_1']
    assert events[0]['response_window_complete'] is False


def test_known_threat_prefix_is_retained_when_later_tracking_coverage_missing(tmp_path):
    rows = [frame(0, 4.), frame(1, 3.6)]
    for row in rows:
        row['tracking_coverage_valid'][-1] = False
    events = prepare(tmp_path, rows)
    assert events[0]['hazard'] is True
    assert events[0]['annotation_coverage_complete'] is False


def test_persistence_window_reaches_deadline_and_reports_short_scene(tmp_path):
    rows = [frame(0)] + [frame(i, 20.) for i in range(1, 15)]
    events = prepare(tmp_path, rows, window_s=.2)
    assert events[0]['sample_ids'][-1] == 'scene_11'
    assert events[0]['response_window_complete'] is True
    subdir = tmp_path / 'short'
    subdir.mkdir()
    short = prepare(subdir, rows[:4], window_s=.2)
    assert short[0]['response_window_complete'] is False


def test_v1_appearance_is_normalized_and_earlier_occurrence_allows_reappearance(tmp_path):
    from radar_vla.response_events import load_events
    rows = [frame(0, 20.), frame(1), frame(2, 20.), frame(3, 20.)]
    event = dict(event_id='legacy', target_agent_id='target', sample_ids=['scene_1', 'scene_2', 'scene_3'])
    loaded = load_events(write_rows(tmp_path / 'events.jsonl', [event]), rows)[0]
    assert loaded['trigger_sample_id'] == 'scene_2'
    assert loaded['event_kind'] == 'reappearance'
    assert loaded['scene_id'] == 'scene' and loaded['split'] == 'train'


def test_explicit_hazard_start_does_not_need_preceding_absence(tmp_path):
    from radar_vla.response_events import load_events
    rows = [frame(0, 20.), frame(1, 19.6), frame(2, 19.2)]
    event = dict(event_id='annotation', target_agent_id='target', event_kind='hazard_start',
                 trigger_sample_id='scene_1', trigger_timestamp_s=.1,
                 sample_ids=['scene_0', 'scene_1', 'scene_2'])
    loaded = load_events(write_rows(tmp_path / 'events.jsonl', [event]), rows)
    assert loaded[0]['trigger_sample_id'] == 'scene_1'


@pytest.mark.parametrize('change,match', [
    ({'sample_ids': ['scene_0', 'scene_2']}, 'consecutive'),
    ({'trigger_timestamp_s': 99.}, 'timestamp'),
    ({'scene_id': 'wrong'}, 'scene'),
    ({'split': 'test'}, 'split'),
    ({'event_kind': 'invented'}, 'event_kind'),
])
def test_v2_event_validation_rejects_inconsistent_identity_and_clock(tmp_path, change, match):
    from radar_vla.response_events import load_events
    rows = [frame(0), frame(1, 20.), frame(2, 20.)]
    event = dict(event_id='one', target_agent_id='target', event_kind='appearance',
                 trigger_sample_id='scene_1', sample_ids=['scene_0', 'scene_1', 'scene_2'])
    event.update(change)
    with pytest.raises(ValueError, match=match):
        load_events(write_rows(tmp_path / 'events.jsonl', [event]), rows)


def test_duplicate_overlapping_episode_cannot_restart_same_track_clock(tmp_path):
    from radar_vla.response_events import load_events
    rows = [frame(0, 10.), frame(1, 9.6), frame(2, 9.2)]
    first = dict(event_id='one', target_agent_id='target', event_kind='hazard_start',
                 trigger_sample_id='scene_0', sample_ids=['scene_0', 'scene_1', 'scene_2'])
    second = {**first, 'event_id': 'duplicate', 'trigger_sample_id': 'scene_1'}
    with pytest.raises(ValueError, match='overlap|Duplicate'):
        load_events(write_rows(tmp_path / 'events.jsonl', [first, second]), rows)


def test_event_annotation_does_not_mutate_causal_observation_records(tmp_path):
    rows = [frame(0), frame(1, 10.), frame(2, 9.6)]
    before = copy.deepcopy(rows)
    prepare(tmp_path, rows)
    assert rows == before


def test_scene_cannot_reuse_timestamps_across_splits(tmp_path):
    from radar_vla.response_events import prepare_response_events
    rows = [frame(0), frame(1, 20.), frame(2, 20., split='val')]
    manifest = write_rows(tmp_path / 'frames.jsonl', rows)
    with pytest.raises(ValueError, match='split'):
        prepare_response_events(manifest, tmp_path / 'events.jsonl')
