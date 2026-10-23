"""Compare validation ego trajectories under codec, teacher forcing and generation."""
from __future__ import annotations

from radar_vla.local_risk import supervision_risk
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from radar_vla.curves import spline_basis
from radar_vla.data import RadarDataset, collate_batch
from radar_vla.pipeline import load_models, to_device, file_digest, autocast_context, validate_input_shapes
from radar_vla.planner import create_supervision


def selected_indices(records):
    episodes = defaultdict(list)
    for index, record in enumerate(records):
        episodes[record['episode_id']].append(index)
    selected = set()
    for indices in episodes.values():
        for fraction in (0., .25, .5, .75, 1.):
            selected.add(indices[round(fraction * (len(indices) - 1))])
        for index in indices:
            risk = supervision_risk(records[index])
            if risk['pcol_mask'][2] and risk['pcol'][2] == 1:
                selected.add(index)
    return sorted(selected)


def score(plan, record):
    if (not plan or not plan.get('valid') or not plan.get('ego_trajectory')
            or any(point is None for point in plan['ego_trajectory'])):
        return dict(valid=False, ade_m=None, fde_m=None, initial_velocity_error_mps=None,
                    mode=None if not plan else plan.get('mode'))
    target = np.asarray(record['ego']['future_xy'], dtype=float)
    truth_times = np.asarray(record['future_times_s'], dtype=float)
    valid = np.asarray(record['ego']['future_valid'], dtype=bool)
    points = np.asarray(plan['ego_trajectory'], dtype=float)
    times = np.asarray(plan['ego_times_s'], dtype=float)
    if points.shape != (len(times), 2):
        raise ValueError('decoded plan has inconsistent trajectory length')
    if not np.isfinite(points).all():
        return dict(valid=False, ade_m=None, fde_m=None,
                    initial_velocity_error_mps=None, mode=plan.get('mode'))
    errors = []
    scored_times = []
    final = None
    for point, moment in zip(points, times):
        match = np.flatnonzero(np.isclose(truth_times, moment, atol=1e-5, rtol=0))
        if len(match) == 1 and valid[match[0]]:
            delta = float(np.linalg.norm(point - target[match[0]]))
            errors.append(delta)
            scored_times.append(float(moment))
            if match[0] == len(truth_times) - 1:
                final = delta
    velocity_error = None
    if plan.get('ego_control_points') and plan.get('control_times_s'):
        controls = torch.tensor(plan['ego_control_points'], dtype=torch.float64)
        knots = torch.tensor(plan['control_times_s'], dtype=torch.float64)
        initial = (spline_basis(knots, knots[:1], derivative=1) @ controls)[0].numpy()
        velocity_error = float(np.linalg.norm(initial - np.asarray(record['ego']['velocity'])))
    return dict(valid=True, ade_m=float(np.mean(errors)) if errors else None,
                fde_m=final, initial_velocity_error_mps=velocity_error,
                mode=plan.get('mode'), point_errors_m=errors, scored_times_s=scored_times,
                point_coverage=len(errors)/int(valid.sum()) if valid.any() else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    grounder, planner, state = load_models(args.checkpoint, args.device)
    if planner is None:
        raise ValueError('SFT checkpoint required')
    dataset = RadarDataset(args.manifest, 'val')
    validate_input_shapes(dataset, state['protocol']['config'])
    indices = selected_indices(dataset.records)
    rows = []
    for number, index in enumerate(indices, 1):
        record = dataset.records[index]
        batch = to_device(collate_batch([dataset[index]]), args.device)
        with torch.no_grad(), autocast_context(args.device, state['protocol'].get('precision','float32')):
            prediction = grounder(batch)
            supervision = create_supervision(batch, planner.tokenizer, planner.config)
            targets = supervision['target_ids']
            logits = planner(prediction['radar_tokens'], prediction['risk'], batch['instruction'], targets)
            generated = planner.generate(prediction['radar_tokens'], prediction['risk'],
                                         batch['instruction'], max_new_tokens=512)[0]
        target_ids = targets[0].cpu().tolist()
        while target_ids and target_ids[-1] == planner.tokenizer.pad_id:
            target_ids.pop()
        forced_ids = target_ids[:]
        for position, token in enumerate(target_ids):
            for kind in ('position', 'velocity'):
                offset = (planner.tokenizer.position_offset if kind == 'position'
                          else planner.tokenizer.velocity_offset)
                if offset <= token < offset + planner.tokenizer.bins:
                    forced_ids[position] = int(offset + logits[0, position,
                        offset:offset + planner.tokenizer.bins].argmax().item())
                    break
        times = record['future_times_s']
        codec = planner.decode(target_ids, times)
        teacher = planner.decode(forced_ids, times)
        free = planner.decode(generated, times)
        velocity = np.asarray(record['ego']['velocity'])
        cv = {'valid': True, 'mode': 'constant_velocity', 'ego_times_s': times,
              'ego_trajectory': [(velocity * t).tolist() for t in times]}
        row = dict(sample_id=record['sample_id'], episode_id=record['episode_id'],
                   category=record['category'], weather=record['weather'],
                   collision_3s=supervision_risk(record)['pcol'][2],
                   teacher_mode=supervision['rows'][0]['mode'],
                   scores={name: score(plan, record) for name, plan in
                           [('constant_velocity', cv), ('codec', codec),
                            ('teacher_forced', teacher), ('free_generation', free)]},
                   free_tokens=len(generated))
        row['plans'] = dict(constant_velocity=cv, codec=codec, teacher_forced=teacher,
                            free_generation=free)
        row['speed_mps'] = float(np.linalg.norm(velocity))
        row['yaw_rate_radps'] = float(record['ego']['yaw_rate'])
        rows.append(row)
        print(f'{number}/{len(indices)} {row["sample_id"]} '
              f'CV={row["scores"]["constant_velocity"]["ade_m"]:.3f} '
              f'free={row["scores"]["free_generation"]["ade_m"]}', flush=True)
    summary = {}
    for method in rows[0]['scores']:
        values = [row['scores'][method] for row in rows]
        summary[method] = dict(samples=len(values), valid=sum(value['valid'] for value in values),
            **{key: (float(np.mean(found)) if found else None) for key in
               ('ade_m', 'fde_m', 'initial_velocity_error_mps')
               for found in [[value[key] for value in values if value[key] is not None]]})
    result = dict(checkpoint=str(args.checkpoint.resolve()), manifest=str(args.manifest.resolve()),
                  checkpoint_sha256=file_digest(args.checkpoint),
                  manifest_sha256=file_digest(args.manifest),
                  script_sha256=file_digest(Path(__file__)),
                  selection='Five spaced frames plus every 3s collision-positive frame per validation episode',
                  summary=summary, rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
