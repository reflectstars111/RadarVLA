"""Deterministic toy scenes for exercising the pipeline, not scientific evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import build_risk_labels, nominal_rollout
from .data import SCHEMA_VERSION


def write_synthetic_dataset(output: Path, scenes_per_split=2, frames_per_scene=2,
                            seed=42, frames=3, range_bins=32, azimuth_bins=24,
                            horizon_steps=6):
    """Generate RA power/relative-Doppler arrays and scene-disjoint train/val/test.

    Radar arrays are a noisy analytic return model, not a calibrated sensor
    simulator. Absolute agent velocities and futures use the current ego frame.
    """
    if min(scenes_per_split, frames_per_scene, frames, range_bins, azimuth_bins, horizon_steps) < 1:
        raise ValueError('synthetic dimensions and counts must all be positive')
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to replace nonempty synthetic output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    arrays_dir = output / 'radar'
    arrays_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    ranges = np.linspace(.5, 70., range_bins)
    angles = np.linspace(-np.pi / 3, np.pi / 3, azimuth_bins)
    offsets = np.linspace(-.1 * (frames - 1), 0., frames)
    times = np.linspace(3. / horizon_steps, 3., horizon_steps)
    rr, aa = np.meshgrid(ranges, angles, indexing='ij')
    records = []
    for split_index, split in enumerate(('train', 'val', 'test')):
        for scene_index in range(scenes_per_split):
            scene_id = f'{split}_scene_{scene_index:04d}'
            scenario = ('stopped_leader', 'crossing', 'clear')[(scene_index + split_index) % 3]
            speed = float(rng.uniform(7., 12.))
            initial_x = float(rng.uniform(17., 24.))
            for frame_index in range(frames_per_scene):
                sample_id = f'{scene_id}_frame_{frame_index:04d}'
                now = frame_index * .1
                ego_velocity = np.array([speed, 0.])
                if scenario == 'stopped_leader':
                    position, velocity, heading = np.array([initial_x - speed * now, 0.]), np.zeros(2), 0.
                    size = [4.5, 1.9]
                    braking = 4.
                elif scenario == 'crossing':
                    velocity = np.array([0., -2.5])
                    position = np.array([initial_x - speed * now, 5. + velocity[1] * now])
                    heading, size, braking = -np.pi / 2, [1., .7], 4.
                else:
                    position, velocity, heading = np.array([initial_x, 7.]), ego_velocity.copy(), 0.
                    size, braking = [4.5, 1.9], 0.
                future_xy = position[None] + times[:, None] * velocity[None]
                ego_future, _ = nominal_rollout(ego_velocity, 0., times, braking)
                power, doppler = [], []
                for dt in offsets:
                    history_xy = position + (velocity - ego_velocity) * dt
                    target_range = np.linalg.norm(history_xy)
                    target_angle = np.arctan2(history_xy[1], history_xy[0])
                    radial_velocity = np.dot(velocity - ego_velocity, history_xy / max(target_range, 1e-8))
                    blob = np.exp(-.5 * ((rr - target_range) / 1.8) ** 2
                                  - .5 * ((aa - target_angle) / .08) ** 2)
                    noise = rng.exponential(.025, rr.shape)
                    power.append((noise + 5. * blob).astype(np.float32))
                    velocity_map = rng.normal(0., .15, rr.shape)
                    velocity_map = velocity_map * (1 - blob) + radial_velocity * blob
                    doppler.append(velocity_map.astype(np.float32))
                power_path = Path('radar') / f'{sample_id}_power.npy'
                doppler_path = Path('radar') / f'{sample_id}_doppler.npy'
                np.save(output / power_path, np.stack(power))
                np.save(output / doppler_path, np.stack(doppler))
                record = {
                    'schema_version': SCHEMA_VERSION, 'sample_id': sample_id, 'scene_id': scene_id,
                    'split': split, 'timestamp_s': now, 'source': 'synthetic_smoke_only', 'scenario': scenario,
                    'sensors': {'radar': {'power': str(power_path), 'unfolded_doppler': str(doppler_path),
                                           'range_m': ranges.tolist(), 'azimuth_rad': angles.tolist(),
                                           'time_offsets_s': offsets.tolist()}},
                    'ego': {'velocity': ego_velocity.tolist(), 'acceleration': [0., 0.], 'yaw_rate': 0.,
                            'box_size': [4.5, 1.9], 'future_xy': ego_future.tolist(),
                            'future_valid': [True] * horizon_steps},
                    'agents': [{'id': 'target_0', 'position': position.tolist(), 'velocity': velocity.tolist(),
                                'heading': float(heading), 'size': size, 'future_xy': future_xy.tolist(),
                                'future_yaw': [float(heading)] * horizon_steps,
                                'future_valid': [True] * horizon_steps}],
                    'future_times_s': times.tolist(),
                    'language': {'instruction': 'Continue ahead while maintaining a safe distance.'},
                }
                record['risk_label'] = build_risk_labels(record)
                records.append(record)
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    return manifest
