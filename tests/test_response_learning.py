"""Finite-candidate learning direction; not evidence of learned LM driving.

The physical reward and leave-one-out estimator are production code. Only the
policy is deliberately small, so a wrong reward ordering or reversed gradient
can be detected without claiming a random language model has learned to drive.
"""
import copy

import pytest
import torch

from radar_vla.response_reward import response_reward
from radar_vla.rl_policy import leave_one_out_advantages


def _record(lateral):
    times = [.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.]
    return dict(future_times_s=times,
        ego=dict(velocity=[4., 0.], yaw_rate=0., box_size=[1., .5]),
        agent_supervision_available=True, tracking_coverage_valid=[True] * len(times),
        agents=[dict(id='obstacle', position=[10., lateral], velocity=[0., 0.],
            heading=0., size=[1., .5], future_xy=[[10., lateral]] * len(times),
            future_valid=[True] * len(times), future_yaw=[0.] * len(times))],
        map=dict(lane_centerlines=[dict(id='lane', left_boundary_id='left', right_boundary_id='right')],
            lane_boundaries=[dict(id='left', points=[[-5., 10.], [100., 10.]]),
                             dict(id='right', points=[[-5., -10.], [100., -10.]])]))


def _candidates():
    brake = dict(valid=True, ego_complete=True, control_times_s=[0., 1., 2., 3.],
        ego_control_points=[[0., 0.], [3.6, 0.], [5.6, 0.], [6., 0.]],
        ego_trajectory=[[3.6, 0.], [5.6, 0.], [6., 0.]], ego_times_s=[1., 2., 3.])
    steady = copy.deepcopy(brake)
    steady['ego_control_points'] = [[0., 0.], [4., 0.], [8., 0.], [12., 0.]]
    steady['ego_trajectory'] = [[4., 0.], [8., 0.], [12., 0.]]
    # The same action class must solve the hazardous scene and avoid needless
    # braking in the harmless scene. Slower identical braking is a third choice.
    return [(brake, .05), (brake, .25), (steady, .05)]


@pytest.mark.parametrize('seed', [11, 29])
def test_actual_reward_reinforce_learns_fast_avoidance_and_harmless_steady_control(seed):
    scores = [[response_reward(record, plan, 'obstacle', latency)
               for plan, latency in _candidates()] for record in (_record(0.), _record(8.))]
    assert scores[0][0]['effective'] and scores[0][1]['effective']
    assert scores[0][2]['collision'] == 1
    assert scores[1][0]['reason'] == scores[1][1]['reason'] == 'unnecessary_response'
    assert scores[1][2]['reason'] == 'safe_control'
    rewards = torch.tensor([[score['reward'] for score in scene] for scene in scores])
    assert rewards[0, 0] > rewards[0, 1] > rewards[0, 2]
    assert rewards[1, 2] > rewards[1, 0] > rewards[1, 1]

    logits = torch.nn.Parameter(torch.zeros(2, 3))
    optimizer = torch.optim.Adam([logits], lr=.06)
    generator = torch.Generator().manual_seed(seed)
    before = logits.detach().softmax(-1)
    for _ in range(180):
        log_probs = logits.log_softmax(-1)
        losses = []
        for scene in range(2):
            actions = torch.multinomial(log_probs[scene].detach().exp(), 16,
                                        replacement=True, generator=generator)
            sampled_rewards = rewards[scene, actions].tolist()
            advantages = torch.tensor(leave_one_out_advantages(sampled_rewards))
            losses.append(-(log_probs[scene, actions] * advantages).mean())
        optimizer.zero_grad()
        torch.stack(losses).mean().backward()
        optimizer.step()
    after = logits.detach().softmax(-1)

    assert after[0, 0] > .8 and after[0, 0] > before[0, 0]
    assert after[0, 1] < before[0, 1]
    assert after[0, 2] < .02
    assert after[1, 2] > .95 and after[1, 2] > before[1, 2]
    assert after[1, :2].sum() < .05
    # Both contexts improve under their own true reward, preventing a global
    # 'always brake' solution from passing solely on the hazardous example.
    assert torch.all((after * rewards).sum(-1) > (before * rewards).sum(-1))


def test_validation_guard_keeps_safe_sft_over_unsafe_or_needlessly_braking_policy():
    from radar_vla.response_rl import _validation_key
    candidates = _candidates()
    reports = [dict(event_id='hazard', hazard=True, detected=True, eligible=True, reason='eligible'),
               dict(event_id='control', hazard=False, detected=True, eligible=True, reason='eligible')]

    def rollouts(hazard_action, control_action):
        result = []
        for report, record, action in zip(reports, (_record(0.), _record(8.)),
                                           (hazard_action, control_action)):
            plan, latency = candidates[action]
            result.append(dict(event_id=report['event_id'], ids=[action],
                score=response_reward(record, plan, 'obstacle', latency)))
        return result

    baseline = _validation_key(rollouts(1, 2), reports)
    collision = _validation_key(rollouts(2, 2), reports)
    unnecessary_brake = _validation_key(rollouts(0, 0), reports)
    safe_improvement = _validation_key(rollouts(0, 2), reports)
    assert baseline < collision
    assert baseline < unnecessary_brake
    assert safe_improvement < baseline
