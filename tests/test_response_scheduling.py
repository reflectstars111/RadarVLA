"""Controlled service times exercise the real non-preemptive planner scheduler."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _context(times=(0., .1, .2, .3, .4, .5), ready=None, detected=None):
    record = dict(timestamp_s=10., future_times_s=[.5, 1., 2., 3.],
                  ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
                  agents=[dict(id='target', position=[30., 8.], velocity=[4., 0.],
                               size=[1., .5], heading=0.,
                               future_xy=[[30.+4*t, 8.] for t in [.5, 1., 2., 3.]],
                               future_yaw=[0.]*4, future_valid=[True]*4)],
                  agent_supervision_available=True, tracking_coverage_valid=[True]*4)
    ready = [t+.1 for t in times] if ready is None else ready
    frames = [dict(record={**record, 'timestamp_s':10.+t, 'sample_id':str(i)},
                   ready=10.+r, radar=torch.full((1, 2, 8), float(i)),
                   risk=torch.full((1, 5), float(i)), instructions=[str(i)],
                   future_times_s=[.5, 1., 2., 3.]) for i, (t, r) in enumerate(zip(times, ready))]
    detected = ready[0] if detected is None else detected
    report = dict(event_id='event', detected=True, eligible=True, hazard=False,
                  reason='eligible', detected_timestamp_s=10.+detected)
    return dict(record=record, event=dict(event_id='event', target_agent_id='target'),
                frames=frames, report=report, perception_s=detected)


def _timed_policy(monkeypatch, duration):
    import radar_vla.response_rl as rl
    timed, sampled = [], []
    def observe(radar, risk, instructions):
        i = int(radar[0, 0, 0])
        assert int(risk[0, 0]) == i and instructions == [str(i)]
        return i
    def sample(planner, radar, risk, instructions, *args):
        i = observe(radar, risk, instructions)
        sampled.append(i)
        return dict(ids=[i], token_log_probs=[0.], log_prob_sum=0.)
    def measure(planner, radar, risk, instructions, **kwargs):
        i = observe(radar, risk, instructions)
        timed.append(i)
        if kwargs['ids'] is not None:
            assert kwargs['ids'] == [i]
        plan = dict(valid=True, ego_complete=True, mode='long', control_times_s=[0., 1., 2., 3.],
                    ego_control_points=[[0., 0.], [4., 0.], [8., 0.], [12., 0.]])
        return dict(ids=[i], plan=plan, generation_s=duration)
    monkeypatch.setattr(rl, 'sample_plan', sample)
    monkeypatch.setattr(rl, 'measure_plan', measure)
    return timed, sampled


@pytest.mark.parametrize('greedy', [False, True])
def test_latest_ready_skips_backlog_without_preempting_or_resetting_clock(monkeypatch, greedy):
    from radar_vla.response_rl import ResponseRLConfig, _rollout, _aggregate
    context = _context()
    original_report = deepcopy(context['report'])
    timed, sampled = _timed_policy(monkeypatch, .35)
    row = _rollout(None, context, ResponseRLConfig(planner_observation_policy='latest_ready'),
                   'cpu', greedy=greedy)
    decisions = row['decisions']
    assert timed == [0, 3, 5]
    assert sampled == ([] if greedy else [0, 3, 5])
    assert [d['frame_index'] for d in decisions] == [0, 3, 5]
    assert [d['planner_start_s'] for d in decisions] == pytest.approx([.1, .45, .8])
    assert [d['ready_s'] for d in decisions] == pytest.approx([.45, .8, 1.15])
    assert [d['observation_timestamp_s'] for d in decisions] == pytest.approx([10., 10.3, 10.5])
    assert [d['observation_age_s'] for d in decisions] == pytest.approx([.1, .15, .3])
    assert [d['queue_s'] for d in decisions] == pytest.approx([0., .05, .2])
    assert [d['skipped_frame_count'] for d in decisions] == [0, 2, 1]
    assert context['report'] == original_report and context['record']['timestamp_s'] == 10.
    assert row['score']['perception_s'] == .1
    assert row['score']['terminal_latency_s'] == pytest.approx(1.05)
    assert row['score']['generation_s'] == .35
    assert row['score']['waiting_s'] == pytest.approx(0.)
    summary = _aggregate([row], [context['report']])
    assert summary['observation_age_p95_s'] == pytest.approx(np.percentile([.1, .15, .3], 95))
    assert summary['skipped_frame_count'] == 3
    assert summary['perception_observation_policy'] == 'fifo'


def test_latest_ready_matches_fifo_when_service_keeps_up(monkeypatch):
    from radar_vla.response_rl import ResponseRLConfig, _rollout
    context = _context()
    timed, sampled = _timed_policy(monkeypatch, .04)
    fifo = _rollout(None, context, ResponseRLConfig(), 'cpu', greedy=True)
    latest = _rollout(None, context, ResponseRLConfig(planner_observation_policy='latest_ready'),
                      'cpu', greedy=True)
    assert timed == list(range(6))*2 and not sampled
    assert latest['ids'] == fifo['ids']
    assert [d['ready_s'] for d in latest['decisions']] == pytest.approx([.14, .24, .34, .44, .54, .64])
    assert all(d['skipped_frame_count'] == 0 for d in latest['decisions'])
    assert [d['observation_age_s'] for d in latest['decisions']] == pytest.approx([.1]*6)


def test_latest_ready_waits_for_perception_and_never_uses_future_ready_frame(monkeypatch):
    from radar_vla.response_rl import ResponseRLConfig, _rollout
    context = _context(times=(0., .1, .2), ready=(.1, .2, .9))
    timed, _ = _timed_policy(monkeypatch, .35)
    row = _rollout(None, context, ResponseRLConfig(planner_observation_policy='latest_ready'),
                   'cpu', greedy=True)
    assert timed == [0, 1, 2]  # frame 2 already exists at .45, but is not perception-ready.
    assert [d['planner_start_s'] for d in row['decisions']] == pytest.approx([.1, .45, .9])
    assert [d['observation_age_s'] for d in row['decisions']] == pytest.approx([.1, .35, .7])
    assert all(d['planner_start_s'] >= d['observation_ready_s'] for d in row['decisions'])


def test_first_detected_clock_survives_initial_ready_burst(monkeypatch):
    from radar_vla.response_rl import ResponseRLConfig, _rollout
    context = _context(times=(0., .1, .2), ready=(.4, .4, .4))
    timed, _ = _timed_policy(monkeypatch, .35)
    row = _rollout(None, context, ResponseRLConfig(planner_observation_policy='latest_ready'),
                   'cpu', greedy=True)
    assert timed == [2]
    assert row['decisions'][0]['planner_start_s'] == pytest.approx(.4)
    assert row['decisions'][0]['skipped_frame_count'] == 2
    assert row['score']['perception_s'] == .4
    assert row['score']['terminal_latency_s'] == pytest.approx(.35)
    assert row['score']['waiting_s'] == pytest.approx(0.)


def test_default_fifo_retains_old_backlog_timing_and_config(monkeypatch):
    from radar_vla.response_rl import ResponseRLConfig, _rollout
    context = _context()
    timed, _ = _timed_policy(monkeypatch, .35)
    default = _rollout(None, context, ResponseRLConfig(), 'cpu', greedy=True)
    explicit = _rollout(None, context, ResponseRLConfig(planner_observation_policy='fifo'),
                        'cpu', greedy=True)
    assert timed == list(range(6))*2
    assert [d['ready_s'] for d in default['decisions']] == pytest.approx([.45, .8, 1.15, 1.5, 1.85, 2.2])
    assert explicit['decisions'] == default['decisions']
    assert all(d['skipped_frame_count'] == 0 for d in default['decisions'])
    with pytest.raises(ValueError, match='planner_observation_policy'):
        ResponseRLConfig(planner_observation_policy='preempt')


def test_rl_rescores_selected_frame_instead_of_zipping_the_unfiltered_context(tmp_path, monkeypatch):
    """Tiny policy boundary doubles isolate production update-loop bookkeeping."""
    import radar_vla.response_rl as rl
    context = _context()
    planner = torch.nn.Linear(1, 1)
    planner.config = SimpleNamespace(risk_source='predicted')
    grounder = torch.nn.Linear(1, 1)
    grounder.config = SimpleNamespace(use_detection=True)
    checkpoint, manifest, events = (tmp_path/name for name in ('initial.pt', 'manifest.jsonl', 'events.jsonl'))
    for path in (checkpoint, manifest, events):
        path.write_text('')
    monkeypatch.setattr(rl, 'load_models', lambda *a: (grounder, planner, {'protocol': {'stage':'sft', 'config':{}}}))
    monkeypatch.setattr(rl, 'load_manifest', lambda *a: [])
    monkeypatch.setattr(rl, 'load_events', lambda *a: [])
    monkeypatch.setattr(rl, '_contexts', lambda *a: ([context], [context['report']]))
    timed, sampled = _timed_policy(monkeypatch, .35)
    rescored = []
    def statistics(model, radar, risk, instructions, ids, *args):
        i = int(radar[0, 0, 0])
        assert ids == [i], 'Selected action was rescored against a skipped observation'
        assert int(risk[0, 0]) == i and instructions == [str(i)]
        rescored.append(i)
        zero = model.weight.sum()*0.
        return dict(log_prob_sum=zero, kl=zero, token_log_probs=zero[None])
    monkeypatch.setattr(rl, 'policy_statistics', statistics)
    status = rl.run_response_rl(checkpoint, manifest, events, tmp_path/'run', epochs=1,
        config=dict(group_size=2, require_controls=False, planner_observation_policy='latest_ready'))
    assert status['state'] == 'complete'
    assert sampled == rescored == [0, 3, 5]*2
    assert timed == [0, 3, 5]*5  # warmup, baseline, two sampled rollouts, validation.
