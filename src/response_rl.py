"""Stage 3: single-device, on-policy response fine-tuning on offline events."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .data import RadarDataset, collate_batch, load_manifest
from .detection import decode_detections
from .pipeline import (load_models, to_device, validate_input_shapes, planner_state,
                       restore_planner, data_fingerprint, source_fingerprint, file_digest,
                       save_checkpoint, write_json, _rng_state, _restore_rng)
from .response_reward import ResponseRewardConfig, response_reward, event_baseline
from .response_events import load_events, prepare_response_events
from .execution import measure_plan
from .response_replay import compose_execution
from .rl_policy import sample_plan, policy_statistics, leave_one_out_advantages, validate_policy_rescore


@dataclass
class ResponseRLConfig:
    group_size: int = 4
    temperature: float = 1.
    kl_weight: float = .02
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.
    max_new_tokens: int = 1024
    require_controls: bool = True
    planner_observation_policy: str = 'fifo'
    reward: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError('group_size must be an integer >= 2')
        if type(self.max_new_tokens) is not int or self.max_new_tokens <= 0:
            raise ValueError('max_new_tokens must be positive')
        for name in ('temperature', 'learning_rate', 'max_grad_norm'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.kl_weight) or self.kl_weight < 0:
            raise ValueError('kl_weight must be finite and nonnegative')
        if type(self.require_controls) is not bool:
            raise ValueError('require_controls must be boolean')
        if self.planner_observation_policy not in ('fifo', 'latest_ready'):
            raise ValueError('planner_observation_policy must be fifo or latest_ready')
        self.reward = asdict(ResponseRewardConfig(**self.reward))


def match_event_target(detections, agents, target_agent_id, distance_m):
    """One-to-one center matching; per-frame local_id is never a tracking ID."""
    agents = [a for a in agents if a.get('present_valid', True)]
    if not detections or not agents:
        return None
    xy = np.asarray([d['position_m'] for d in detections])
    gt = np.asarray([a['position'] for a in agents])
    distance = np.linalg.norm(xy[:, None] - gt[None], axis=-1)
    cost = np.where(distance <= distance_m, distance, (len(agents) + len(detections) + 1) * (distance_m + 1))
    predicted, truth = linear_sum_assignment(cost)
    for p, g in zip(predicted, truth):
        if agents[g]['id'] == target_agent_id and distance[p, g] <= distance_m:
            return int(p)
    return None


def _sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def _observation(dataset, index, device):
    batch = collate_batch([dataset[index]])
    # Explicit observation whitelist: no GT future, map, agents, or GT risk.
    keys = ('radar', 'range_m', 'azimuth_rad', 'time_offsets_s', 'ego', 'ego_velocity', 'ego_acceleration', 'radar_pose',
            'radar_cartesian', 'radar_los', 'radar_sensor_velocity', 'radar_doppler_valid', 'future_times_s')
    # Version contracts describe model inputs, not supervision; legacy samples
    # may omit them, while explicit mismatches must reach model validation.
    keys += tuple(key for key in ('risk_horizon_schema_id', 'detection_class_schema_id') if key in batch)
    return to_device({key: batch[key] for key in keys}, device)


@torch.no_grad()
def _contexts(grounder, manifest, events, split, config, device, reward_config):
    dataset = RadarDataset(manifest, split, supervision=False)
    validate_input_shapes(dataset, config)
    by_id = {r['sample_id']: i for i, r in enumerate(dataset.records)}
    # Warm up once; benchmark cold-start separately, not as a learned action cost.
    grounder(_observation(dataset, 0, device)); _sync(device)
    selected_events = [e for e in events if e['split'] == split]
    scene_ends = {}
    needed = {sample for event in selected_events for sample in event['sample_ids']}
    for event in selected_events:
        last = dataset.records[by_id[event['sample_ids'][-1]]]
        scene_ends[event['scene_id']] = max(scene_ends.get(event['scene_id'], -float('inf')), last['timestamp_s'])
    # Replay each relevant scene ONCE from its beginning. An event window must
    # not erase existing FIFO backlog, or give shared frames different clocks.
    cached, busy = {}, {}
    ordered = sorted(enumerate(dataset.records), key=lambda pair: (pair[1]['scene_id'], pair[1]['timestamp_s']))
    for i, record in ordered:
        scene = record['scene_id']
        if scene not in scene_ends or record['timestamp_s'] > scene_ends[scene]:
            continue
        batch = _observation(dataset, i, device)
        _sync(device); start = time.perf_counter()
        prediction = grounder(batch)
        detections = decode_detections(prediction, grounder.config, batch['future_times_s'])[0]
        _sync(device); compute = time.perf_counter() - start
        ready = max(record['timestamp_s'], busy.get(scene, -float('inf'))) + compute
        busy[scene] = ready
        if record['sample_id'] in needed:
            risk = prediction['risk'] if config['planner']['risk_source'] == 'predicted' else torch.zeros_like(prediction['risk'])
            cached[record['sample_id']] = dict(detections=detections, ready=ready,
                radar=prediction['radar_tokens'].detach().cpu(), risk=risk.detach().cpu())
    from .response_annotations import reward_record
    scenes = defaultdict(list)
    for record in dataset.records:
        scenes[record['scene_id']].append(record)
    contexts, reports = [], []
    for event in selected_events:
        trigger = event.get('trigger_sample_id') or event['sample_ids'][1]
        ids = event['sample_ids'][event['sample_ids'].index(trigger):]
        trigger_record = dataset.records[by_id[trigger]]
        known = trigger_record.get('agent_supervision_available',
            trigger_record.get('agent_supervision', 'agents' in trigger_record))
        if not known or sum(a['id'] == event['target_agent_id'] and a.get('present_valid', True)
                            for a in trigger_record.get('agents', [])) != 1:
            raise ValueError('Event trigger requires one present annotated target_agent_id')
        # Physical motion begins at the trigger, even when the detector first
        # reports the target in a later observation. GT remains reward-only.
        anchor = reward_record(trigger_record, scenes[trigger_record['scene_id']])
        report = dict(event_id=event['event_id'], target_agent_id=event['target_agent_id'],
                      detected=False, eligible=False, reason='missed_detection',
                      hazard=event.get('hazard'), event_kind=event.get('event_kind', 'appearance'),
                      trigger_sample_id=trigger, trigger_timestamp_s=trigger_record['timestamp_s'],
                      response_window_complete=event.get('response_window_complete', False),
                      response_window_censored=not event.get('response_window_complete', False))
        if report['event_kind'].startswith('hazard_'):
            report['hazard'] = True
        frames = []
        for sample_id in ids:
            i = by_id[sample_id]
            record = dataset.records[i]
            frame = cached[sample_id]
            detections = frame['detections']
            evaluated = reward_record(record, scenes[record['scene_id']])
            match = match_event_target(detections['detections'], record.get('agents', []),
                                       event['target_agent_id'], grounder.config.detection_match_distance_m)
            present = any(a['id'] == event['target_agent_id'] and a.get('present_valid', True)
                          for a in record.get('agents', []))
            if present:
                base = event_baseline(evaluated, event['target_agent_id'], reward_config)
                if base['hazard']:
                    report['hazard'] = True
                elif report['hazard'] is None and base['coverage_complete']:
                    report['hazard'] = False
            if not report['detected']:
                if match is None:
                    continue
                perception = frame['ready'] - anchor['timestamp_s']
                report.update(detected=True, sample_id=sample_id,
                              detected_timestamp_s=frame['ready'], perception_s=perception,
                              matched_observation_timestamp_s=record['timestamp_s'],
                              matched_frame_perception_s=frame['ready'] - record['timestamp_s'],
                              detection_delay_from_trigger_s=perception,
                              detection=detections['detections'][match])
                if detections['detection_truncated']:
                    report['reason'] = 'truncated_detection'
                    break
                # Unknown suffixes stay in replay: known collisions still count.
                report.update(eligible=True, reason='eligible')
            frames.append(dict(record=evaluated, future_times_s=record['future_times_s'],
                               ready=frame['ready'], radar=frame['radar'], risk=frame['risk'],
                               instructions=[record['language']['instruction']]))
        if frames:
            contexts.append(dict(record=anchor, event=event, report=report, frames=frames,
                                 perception_s=report['perception_s']))
        reports.append(report)
    if not reports:
        raise ValueError(f'No {split} events')
    return contexts, reports


def _rollout(planner, context, cfg, device, greedy=False):
    """Stateful action replay over recorded, exogenous observations.

    First-detection clock is immutable. Actions do not generate new radar
    observations; this is not closed-loop sensor or traffic simulation.
    The optional latest-ready planner drops only work not yet started. The
    perception replay remains FIFO; its ready timestamps are never changed.
    """
    anchor = context['record']['timestamp_s']
    detected = context['report']['detected_timestamp_s']
    busy, decisions = detected, []
    frames, next_frame = context['frames'], 0
    while next_frame < len(frames):
        start = max(frames[next_frame]['ready'], busy)
        selected = next_frame
        if cfg.planner_observation_policy == 'latest_ready':
            # _contexts supplies chronological observations with FIFO perception
            # completion times. Frames arriving during this generation cannot
            # replace the in-flight action, only pending work at the next start.
            while selected + 1 < len(frames) and frames[selected + 1]['ready'] <= start:
                selected += 1
        frame = frames[selected]
        radar, risk = frame['radar'].to(device), frame['risk'].to(device)
        sampled = {} if greedy else sample_plan(planner, radar, risk, frame['instructions'],
                                                 cfg.max_new_tokens, cfg.temperature)
        measured = measure_plan(planner, radar, risk, frame['instructions'],
                                ids=sampled.get('ids'), max_new_tokens=cfg.max_new_tokens,
                                future_times_s=frame['future_times_s'])
        busy = start + measured['generation_s']
        decisions.append(dict(**sampled, plan=measured['plan'], ready_s=busy-anchor,
                              queue_s=start-frame['ready'], generation_s=measured['generation_s'],
                              frame_index=selected, observation_sample_id=frame['record'].get('sample_id'),
                              observation_timestamp_s=frame['record']['timestamp_s'],
                              observation_ready_s=frame['ready']-anchor, planner_start_s=start-anchor,
                              observation_age_s=start-frame['record']['timestamp_s'],
                              skipped_frame_count=selected-next_frame))
        decisions[-1]['ids'] = measured['ids']
        next_frame = selected + 1
    reward_cfg = ResponseRewardConfig(**cfg.reward)
    composite, first_ready = compose_execution(context['record'], decisions, reward_cfg)
    first_generation = decisions[0]['generation_s']
    waiting = max(0., anchor + first_ready - detected - first_generation)
    score = response_reward(context['record'], composite, context['event']['target_agent_id'],
                            first_generation, waiting_s=waiting,
                            perception_s=context['perception_s'], config=reward_cfg)
    score['terminal_latency_s'] = max(0., busy - detected)
    if ('reward' in score and not score.get('effective')
            and score.get('reason') not in ('safe_control', 'unnecessary_response',
                                             'unknown_coverage', 'unknown_road')):
        # The physical reward charges the first adoption clock. Charge the
        # rest of a failed episode uniformly as well: a slower later failure
        # must not tie or outrank its faster counterpart.
        extra_wait = max(0., decisions[-1]['ready_s'] - first_ready)
        score['reward'] -= reward_cfg.latency_weight * extra_wait / reward_cfg.response_deadline_s
        if 'quality_adjustment' in score:
            score['quality_adjustment'] = score['reward'] - score['response_time_reward']
    score['episode_generation_s'] = sum(d['generation_s'] for d in decisions)
    score['planner_queue_max_s'] = max(d['queue_s'] for d in decisions)
    return dict(ids=[token for d in decisions for token in d['ids']],
                decisions=decisions, score=score, event_id=context['event']['event_id'],
                planner_observation_policy=cfg.planner_observation_policy)


def _aggregate(rollouts, reports):
    scores = [r['score'] for r in rollouts]
    ages = [d['observation_age_s'] for r in rollouts for d in r.get('decisions', [])
            if 'observation_age_s' in d]
    skipped = [sum(d.get('skipped_frame_count', 0) for d in r.get('decisions', [])) for r in rollouts]
    grouped = defaultdict(list)
    for row in rollouts:
        grouped[row['event_id']].append(row['score'])
    reactions = [s['reaction_s'] for s in scores
                 if s.get('response_detected', s['effective']) and s['reaction_s'] is not None]
    effective_reactions = [s.get('effective_reaction_s', s['reaction_s']) for s in scores if s['effective']]
    hazards = [r for r in reports if r.get('hazard') is True]
    control_ids = {r['event_id'] for r in reports if r.get('hazard') is False}
    controls = [r['score'] for r in rollouts if r['event_id'] in control_ids]
    def rate(values):
        return sum(values) / len(values) if values else None
    outcomes = Counter(s['reason'] for s in scores)
    outcomes.update(r['reason'] for r in reports if r['event_id'] not in grouped)
    success = [rate([s['effective'] for s in grouped[r['event_id']]]) or 0. for r in hazards]
    return dict(events=len(reports), detected_events=sum(r['detected'] for r in reports),
                eligible_events=sum(r['eligible'] for r in reports), sampled_plans=len(scores),
                generated_plans=sum(len(r.get('decisions', [r])) for r in rollouts),
                detected_event_rate=rate([r['detected'] for r in reports]),
                eligible_event_rate=rate([r['eligible'] for r in reports]),
                exclusion_reasons=dict(Counter(r['reason'] for r in reports if not r['eligible'])),
                reward_mean=rate([s['reward'] for s in scores]),
                effective_response_rate=rate(success), hazard_events=len(hazards),
                unknown_hazard_events=sum(r.get('hazard') is None for r in reports),
                collision_rate=rate([s['collision'] for s in scores if s['collision'] is not None]),
                collision_rate_population='scored_detected_episode_rollouts_only',
                collision_scored_plans=sum(s['collision'] is not None for s in scores),
                censored_window_events=sum(not r.get('response_window_complete', False) for r in reports),
                unnecessary_response_rate=rate([s.get('response_detected',
                    s['reason'] == 'unnecessary_response') for s in controls]),
                hazard_plans=sum(len(grouped[r['event_id']]) for r in hazards), control_plans=len(controls),
                response_p50_s=float(np.percentile(reactions, 50)) if reactions else None,
                response_p95_s=float(np.percentile(reactions, 95)) if reactions else None,
                response_timed_events=len(reactions),
                response_timed_successes=len(effective_reactions),
                response_p50_population='detected_trajectory_changes',
                effective_response_p50_s=float(np.percentile(effective_reactions, 50)) if effective_reactions else None,
                effective_response_p95_s=float(np.percentile(effective_reactions, 95)) if effective_reactions else None,
                safety_unknown_plans=sum(s.get('safety_status') == 'unknown' for s in scores),
                generation_mean_s=rate([d['generation_s'] for r in rollouts
                                       for d in r.get('decisions', [r['score']])]),
                first_generation_mean_s=rate([s['generation_s'] for s in scores]),
                episode_generation_mean_s=rate([s.get('episode_generation_s', s['generation_s']) for s in scores]),
                observation_age_p50_s=float(np.percentile(ages, 50)) if ages else None,
                observation_age_p95_s=float(np.percentile(ages, 95)) if ages else None,
                observation_age_population='generated_plans_at_planner_start',
                skipped_frame_count=sum(skipped), skipped_frame_count_mean=rate(skipped),
                skipped_frame_count_p95=float(np.percentile(skipped, 95)) if skipped else None,
                planner_observation_policies=dict(Counter(r.get('planner_observation_policy', 'fifo') for r in rollouts)),
                perception_observation_policy='fifo',
                generated_tokens_mean=rate([len(r['ids']) for r in rollouts]),
                outcomes=dict(outcomes))


def _validate_models(grounder, planner):
    if planner is None:
        raise ValueError('Response RL requires a Stage 2 SFT planner checkpoint')
    if getattr(planner.config, 'action_representation', 'token_spline_v1') != 'token_spline_v1':
        raise ValueError('continuous action Response RL is not implemented; token RL cannot score continuous actions')
    if not grounder.config.use_detection:
        raise ValueError('Response RL requires a trained dense detector')
    if planner.config.risk_source == 'oracle':
        raise ValueError('Response RL cannot use oracle risk at inference')
    grounder.eval().requires_grad_(False)
    planner.eval()


def _validation_key(rollouts, reports):
    summary = _aggregate(rollouts, reports)
    failures = sum(r['score']['reason'] in ('collision', 'unsafe_clearance', 'invalid_dynamics',
        'invalid_plan', 'short_horizon', 'road_departure')
        for r in rollouts)
    return [failures / max(len(rollouts), 1),
            -(summary['effective_response_rate'] or 0.),
            summary['unnecessary_response_rate'] or 0.,
            -(summary['reward_mean'] if summary['reward_mean'] is not None else -1e6)]


def run_response_rl(checkpoint, manifest, events, output, epochs=1, config=None,
                    device='cpu', seed=42, resume=False):
    """One on-policy update per episode, with leave-one-out REINFORCE + SFT KL.

    Single device only. Wall-clock reward is intentionally hardware-dependent;
    RNG/optimizer restoration does not make timing bitwise deterministic.
    """
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Response RL supports one device; do not launch with torchrun')
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive integer')
    cfg = ResponseRLConfig(**(config or {}))
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError('RL output is nonempty; use --resume or a new directory')
    torch.manual_seed(seed); random.seed(seed)
    grounder, planner, initial = load_models(checkpoint, device)
    if initial['protocol'].get('stage') != 'sft':
        raise ValueError('Initialize RL from SFT; --resume restores an existing RL run')
    _validate_models(grounder, planner)
    model_config = initial['protocol']['config']
    protocol = dict(schema='radar_vla_pipeline_v2', stage='rl', config=model_config,
                    language_model_files=initial['protocol'].get('language_model_files'),
                    precision='float32', rl_config=asdict(cfg), seed=seed, device=str(device),
                    initial_checkpoint_sha256=file_digest(checkpoint), data=data_fingerprint(manifest),
                    events_sha256=file_digest(events), source=source_fingerprint(),
                    algorithm='leave-one-out REINFORCE with conditional SFT KL', world_size=1)
    event_rows = load_events(events, load_manifest(Path(manifest)))
    reference = copy.deepcopy(planner).eval().requires_grad_(False)
    parameters = [p for p in planner.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg.learning_rate, weight_decay=0.)
    history, best, start_epoch = [], None, 1
    if resume:
        restored = torch.load(output / 'last.pt', map_location=device, weights_only=True)
        if restored['protocol'] != protocol:
            raise ValueError('RL resume provenance differs: model, data, events, source, device or configuration')
        restore_planner(planner, restored['planner'])
        optimizer.load_state_dict(restored['optimizer'])
        _restore_rng(restored['rng'], device)
        history, best, start_epoch = restored['history'], restored['best_key'], restored['epoch'] + 1
        if epochs < restored['epoch']:
            raise ValueError('epochs cannot be smaller than the completed checkpoint epoch')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'status.json', dict(state='running', stage='rl'))
    try:
        reward_cfg = ResponseRewardConfig(**cfg.reward)
        train, train_reports = _contexts(grounder, manifest, event_rows, 'train', model_config, device, reward_cfg)
        val, val_reports = _contexts(grounder, manifest, event_rows, 'val', model_config, device, reward_cfg)
        write_json(output / 'event_coverage.json', dict(train=train_reports, val=val_reports))
        if not train or not val:
            raise ValueError('No eligible detected events in train/val; inspect event_coverage.json, detection and future coverage')
        if cfg.require_controls and any(not {True, False}.issubset(
                {c['report']['hazard'] for c in rows}) for rows in (train, val)):
            raise ValueError('Train and val each require both hazard and harmless control events')
        # Warm generation without consuming the policy RNG used for training.
        rng = _rng_state(device)
        _rollout(planner, train[0], cfg, device, greedy=True)
        _restore_rng(rng, device)
        if not history:
            # Select against the unchanged SFT model before any RL update.
            baseline = [_rollout(planner, c, cfg, device, greedy=True) for c in val]
            best = _validation_key(baseline, val_reports)
            history.append(dict(epoch=0, source='initial_sft', validation=_aggregate(baseline, val_reports)))
            state = dict(protocol=protocol, epoch=0, grounder=grounder.state_dict(),
                         planner=planner_state(planner), optimizer=optimizer.state_dict(),
                         rng=_rng_state(device), history=history, best_key=best)
            save_checkpoint(output / 'best.pt', state)
            save_checkpoint(output / 'last.pt', state)
            write_json(output / 'validation_epoch_0.json', baseline)
        for epoch in range(start_epoch, epochs + 1):
            order = list(train)
            random.shuffle(order)
            sampled, losses, advantages, kls = [], [], [], []
            for context in order:
                group = [_rollout(planner, context, cfg, device) for _ in range(cfg.group_size)]
                advantage = leave_one_out_advantages([r['score']['reward'] for r in group])
                optimizer.zero_grad(set_to_none=True)
                for row, value in zip(group, advantage):
                    for action in row['decisions']:
                        frame = context['frames'][action['frame_index']]
                        stats = policy_statistics(planner, frame['radar'].to(device), frame['risk'].to(device),
                                                  frame['instructions'], action['ids'], cfg.temperature, reference)
                        validate_policy_rescore(planner, action, stats)
                        # Episode probability is the product over decisions.
                        loss = (-value * stats['log_prob_sum'] + cfg.kl_weight * stats['kl'] /
                                len(row['decisions'])) / cfg.group_size
                        if not torch.isfinite(loss):
                            raise ValueError('Nonfinite RL loss')
                        loss.backward()
                        losses.append(float(loss.detach()) * cfg.group_size)
                        kls.append(float(stats['kl'].detach()))
                torch.nn.utils.clip_grad_norm_(parameters, cfg.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                sampled.extend(group); advantages.extend(advantage)
            validation = [_rollout(planner, c, cfg, device, greedy=True) for c in val]
            summary = _aggregate(validation, val_reports)
            key = _validation_key(validation, val_reports)
            improved = best is None or key < best
            if improved:
                best = key
            history.append(dict(epoch=epoch, loss_mean=float(np.mean(losses)), kl_mean=float(np.mean(kls)),
                                nonzero_advantages=sum(abs(a) > 1e-12 for a in advantages),
                                train=_aggregate(sampled, train_reports), validation=summary))
            state = dict(protocol=protocol, epoch=epoch, grounder=grounder.state_dict(),
                         planner=planner_state(planner), optimizer=optimizer.state_dict(),
                         rng=_rng_state(device), history=history, best_key=best)
            save_checkpoint(output / 'last.pt', state)
            if improved:
                save_checkpoint(output / 'best.pt', state)
            write_json(output / f'validation_epoch_{epoch}.json', validation)
            write_json(output / 'history.json', history)
            print(json.dumps(dict(stage='rl', epoch=epoch, validation=summary)), flush=True)
        status = dict(state='complete', stage='rl', epoch=history[-1]['epoch'],
                      test_evaluated=False, output=str(output),
                      semantics='stateful episode replay over exogenous recorded observations')
        write_json(output / 'status.json', status)
        return status
    except BaseException as error:
        write_json(output / 'status.json', dict(state='failed', stage='rl', error=str(error)))
        raise


def evaluate_response_rl(checkpoint, manifest, events, output, split='test', device='cpu', config=None):
    """Greedy event evaluation for either the SFT baseline or the final RL model."""
    grounder, planner, state = load_models(checkpoint, device)
    _validate_models(grounder, planner)
    cfg = ResponseRLConfig(**(config if config is not None else state['protocol'].get('rl_config', {})))
    rows = load_events(events, load_manifest(Path(manifest)))
    contexts, reports = _contexts(grounder, manifest, rows, split, state['protocol']['config'],
                                  device, ResponseRewardConfig(**cfg.reward))
    if contexts:
        _rollout(planner, contexts[0], cfg, device, greedy=True)
    rollouts = [_rollout(planner, c, cfg, device, greedy=True) for c in contexts]
    result = dict(**_aggregate(rollouts, reports), split=split, rl_config=asdict(cfg),
                  event_coverage=reports, rollouts=rollouts, checkpoint_sha256=file_digest(checkpoint),
                  events_sha256=file_digest(events), data=data_fingerprint(manifest), device=str(device),
                  semantics='stateful recorded-observation replay; no closed-loop sensor or traffic simulation')
    write_json(output, result)
    return result
