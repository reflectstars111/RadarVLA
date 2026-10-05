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
                    'coordinate_frame': 'current_ego',
                    'split': split, 'timestamp_s': now, 'source': 'synthetic_smoke_only', 'scenario': scenario,
                    'sensors': {'radar': {'power': str(power_path), 'unfolded_doppler': str(doppler_path),
                                           'range_m': ranges.tolist(), 'azimuth_rad': angles.tolist(),
                                           'time_offsets_s': offsets.tolist(),
                                           'poses_current_ego': [(np.eye(4) + np.array([[0, 0, 0, speed*dt], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])).tolist() for dt in offsets],
                                           'sensor_velocity_current_ego': [ego_velocity.tolist()] * frames}},
                    'ego': {'velocity': ego_velocity.tolist(), 'acceleration': [0., 0.], 'yaw_rate': 0.,
                            'box_size': [4.5, 1.9], 'future_xy': ego_future.tolist(),
                            'future_valid': [True] * horizon_steps},
                    'agents': [{'id': 'target_0', 'position': position.tolist(), 'velocity': velocity.tolist(),
                                'heading': float(heading), 'size': size, 'future_xy': future_xy.tolist(),
                                'future_yaw': [float(heading)] * horizon_steps,
                                'future_valid': [True] * horizon_steps}],
                    'tracking_coverage_valid': [True] * len(times),
                    'future_times_s': times.tolist(),
                    'map': {'lane_centerlines': [{'id': 'route_lane', 'points': [[-10., 0., 0.], [70., 0., 0.]],
                              'left_boundary_id': 'left', 'right_boundary_id': 'right'}],
                            'lane_boundaries': [{'id': 'left', 'points': [[-10., 2., 0.], [70., 2., 0.]]},
                                                {'id': 'right', 'points': [[-10., -2., 0.], [70., -2., 0.]]}],
                            'traffic_elements': [], 'route_lane_id': 'route_lane',
                            'route': {'centerline': [[0., 0., 0.], [70., 0., 0.]], 'goal_s_m': 70.}},
                    'language': {'instruction': 'Continue ahead while maintaining a safe distance.'},
                }
                record['risk_label'] = build_risk_labels(record)
                records.append(record)
    manifest = output / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, allow_nan=False) + '\n' for r in records), encoding='utf-8')
    return manifest


def write_synthetic_frames(output: Path, *, scenes_per_split=1, frames_per_scene=6,
                           seed=42, range_bins=256, azimuth_bins=107):
    """Write a fully populated raw frame_t fixture for the import/prepare CLI.

    Camera pixels and point clouds are deterministic geometry fixtures; radar
    returns use the same analytic approximation as write_synthetic_dataset.
    This is a format/integration fixture, never evidence of driving performance.
    """
    from PIL import Image
    from .coordinates import transform_points
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'refusing to overwrite nonempty raw fixture output: {output}')
    output.mkdir(parents=True, exist_ok=True)
    prepared = write_synthetic_dataset(output/'analytic_source', scenes_per_split=scenes_per_split,
                                      frames_per_scene=frames_per_scene, seed=seed, frames=1,
                                      range_bins=range_bins, azimuth_bins=azimuth_bins)
    (output/'sensors').mkdir()
    # Calibrated optical frame x-right,y-down,z-forward → ego x-forward,y-left,z-up.
    camera_pose = np.array([[0., 0., 1., 0.], [-1., 0., 0., 0.], [0., -1., 0., 1.5], [0., 0., 0., 1.]])
    yy, xx = np.mgrid[:48, :64]
    image = np.stack((xx*4, yy*5, np.zeros_like(xx)), -1).astype(np.uint8)
    Image.fromarray(image).save(output/'sensors/front.png')
    rows = []
    for line in prepared.read_text().splitlines():
        sample = json.loads(line); radar = sample['sensors']['radar']; stamp = sample['timestamp_s']
        pose = np.eye(4); pose[0, 3] = sample['ego']['velocity'][0]*stamp
        current_radar = dict(range_m=radar['range_m'], azimuth_rad=radar['azimuth_rad'],
                             timestamp_s=stamp, T_ego_sensor=np.eye(4).tolist())
        for key in ('power', 'unfolded_doppler'):
            filename = Path('sensors')/f'{sample["sample_id"]}_{key}.npy'
            np.save(output/filename, np.load(prepared.parent/radar[key], allow_pickle=False)[0])
            current_radar[key] = str(filename)
        folded_path = Path('sensors')/f'{sample["sample_id"]}_folded.npy'
        np.save(output/folded_path, (np.load(output/current_radar['unfolded_doppler'])+5.) % 10.-5.)
        current_radar['folded_doppler'] = str(folded_path)
        current_radar['max_unambiguous_velocity'] = 5.
        points = np.asarray([[*a['position'], .5, 1.] for a in sample['agents']], np.float32).reshape(-1, 4)
        lidar_path = Path('sensors')/f'{sample["sample_id"]}_lidar.npy'; np.save(output/lidar_path, points)
        actors = []
        for agent in sample['agents']:
            actors.append(dict(id=agent['id'], bbox3d=dict(center=transform_points(agent['position'], pose).tolist(),
                size=[*agent['size'], 1.5], yaw=agent['heading']),
                velocity=[*agent['velocity'], 0.], heading=agent['heading'],
                future_trajectory=[dict(timestamp_s=stamp+t, position=transform_points(p, pose).tolist(), yaw=y, valid=v)
                                   for t, p, y, v in zip(sample['future_times_s'], agent['future_xy'], agent['future_yaw'], agent['future_valid'])]))
        ego = dict(pose=pose.tolist(), velocity=[*sample['ego']['velocity'], 0.], acceleration=[0., 0., 0.],
                   yaw_rate=0., box_size=[*sample['ego']['box_size'], 1.5],
                   future_trajectory=[dict(timestamp_s=stamp+t, position=transform_points(p, pose).tolist(), yaw=0., valid=True)
                                      for t, p in zip(sample['future_times_s'], sample['ego']['future_xy'])])
        map_record = copy_map = json.loads(json.dumps(sample['map']))
        for key in ('lane_centerlines', 'lane_boundaries'):
            for lane in copy_map[key]:
                lane['points'] = transform_points(lane['points'], pose).tolist()
        copy_map['route']['centerline'] = transform_points(copy_map['route']['centerline'], pose).tolist()
        copy_map['traffic_elements'] = [dict(id='synthetic_sign', type='speed_limit',
                                            position=transform_points([35., 3., 2.], pose).tolist(), value_mps=12.)]
        rows.append(dict(schema_version='radar_frame_v2', sample_id=sample['sample_id'], scene_id=sample['scene_id'],
            split=sample['split'], timestamp_s=stamp, coordinate_frame='world',
            source='synthetic_format_fixture_only', tracking={'coverage': 'complete_relevant_agents'},
            sensors=dict(radar=current_radar,
                camera={'front_rgb': dict(path='sensors/front.png', timestamp_s=stamp,
                        intrinsics=[[50., 0., 32.], [0., 50., 24.], [0., 0., 1.]], T_ego_sensor=camera_pose.tolist())},
                lidar=dict(pointcloud=str(lidar_path), timestamp_s=stamp, T_ego_sensor=np.eye(4).tolist())),
            ego=ego, agents=actors, map=map_record, language=sample['language']))
    manifest = output/'frames.jsonl'
    manifest.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in rows), encoding='utf-8')
    return manifest
