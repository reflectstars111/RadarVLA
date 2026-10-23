"""Read-only v5 episode/C-Shenron to versioned training-manifest preparation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time
import numpy as np


from .road_segments import road_segments
from .geometry import build_risk_labels
from .annotation_domain import detection_domain_values

PREPARATION_CONTRACT = 'collection_v3_agent50_ignore_pedcyclist_mount_v1'


def _read_json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _anchor_at(path, carla_frame):
    with Path(path).open() as stream:
        for line in stream:
            # The frame ID is at the start of every compact anchor row. This
            # avoids decoding large future dictionaries for unrelated frames.
            prefix = line[:80]
            if f'"carla_frame": {carla_frame},' in prefix:
                item = json.loads(line)
                if item['carla_frame'] == carla_frame:
                    return item
    return None


def _selected_anchors(path, frame_ids):
    """Read only selected anchors from an episode, never its entire future archive."""
    wanted = set(frame_ids)
    found = {}
    with Path(path).open() as stream:
        for line in stream:
            match = re.search(r'"carla_frame"\s*:\s*(\d+)', line[:100])
            if match and int(match.group(1)) in wanted:
                item = json.loads(line)
                found[item['carla_frame']] = item
                if len(found) == len(wanted):
                    break
    return found


def _to_ego(world_xy, origin, yaw_deg):
    dx, dy = float(world_xy[0]) - origin[0], float(world_xy[1]) - origin[1]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [dx*c + dy*s, dx*s - dy*c]


def _vector_to_ego(world_xy, yaw_deg):
    return _to_ego(world_xy, (0., 0.), yaw_deg)


def _frame_path(episode, index):
    return episode / 'frames' / f'{index:06d}'


_ACTOR_ID = re.compile(r'"actor_id"\s*:\s*(\d+)')


@lru_cache(maxsize=2048)
def _snapshot_ids(path: str):
    """All vehicle/pedestrian IDs plus ego in one recorded world snapshot."""
    return frozenset(int(value) for value in _ACTOR_ID.findall(Path(path).read_text()))


def _coverage_from_snapshots(episode, index, current, anchor):
    ids = {int(current['ego']['actor_id'])} | {int(actor['actor_id']) for actor in current['actors']}
    if ids != {int(key) for key in anchor['futures_by_actor_id']}:
        raise ValueError('anchor future IDs differ from current world snapshot')
    complete = True
    coverage = []
    for step in range(1, 31):
        path = _frame_path(episode, index+step) / 'frame.json'
        if not path.is_file():
            complete = False
        else:
            observed = _snapshot_ids(str(path))
            if observed != ids:
                complete = False
        if any(not entry['valid'][step-1] for entry in anchor['futures_by_actor_id'].values()):
            complete = False
        if step % 5 == 0:
            coverage.append(complete)
    return coverage


def _could_affect_risk(actor, future, origin, ego_speed, ego_extent):
    """Conservative geometric bound for the formula's 80 m distance cap."""
    if future is None or not all(future['valid']):
        return True
    points = [actor['pose']['position_m'][:2]] + [point[:2] for point in future['world_xyzyaw']]
    nearest = min(math.dist(point, origin) for point in points)
    max_step = max(math.dist(a,b) for a,b in zip(points,points[1:]))
    box = actor['bbox']['extent_m']
    radii = math.hypot(ego_extent[0],ego_extent[1]) + math.hypot(box[0],box[1])
    return nearest <= 80. + 3.*ego_speed + max_step + radii + 1.


def _load_frame(episode, index):
    root = _frame_path(episode, index)
    frame = _read_json(root / 'frame.json')
    if frame['index'] != index or frame['camera_frame'] != frame['carla_frame'] or frame['lidar_frame'] != frame['carla_frame']:
        raise ValueError('camera/LiDAR/frame ID mismatch')
    if not (root / frame['sensors']['camera']).is_file() or not (root / frame['sensors']['semantic_lidar']).is_file():
        raise ValueError('missing camera or semantic LiDAR')
    return frame



def _load_taskwise_frame(episode,index):
    frame=_read_json(_frame_path(episode,index)/'frame.json')
    if frame['index']!=index:raise ValueError('Raw source frame index mismatch')
    ego=frame['ego']
    required=[*ego['velocity_world_mps'],*ego['acceleration_world_mps2'],*ego['angular_velocity_world_degps'],
              *ego['pose']['position_m'],*ego['pose']['rotation_deg'],*ego['bbox']['extent_m']]
    if not np.isfinite(np.asarray(required,float)).all():raise ValueError('Missing/nonfinite required ego input')
    return frame

def _product_frames(job_dir):
    """Map source index to a completed core/halo product; core takes precedence."""
    products = {}
    for complete_path in sorted((job_dir / 'radar').glob('*/attempt_*/complete.json')):
        product = complete_path.parent
        complete = _read_json(complete_path)
        meta = _read_json(product / 'radar_meta.json')
        start = int(meta['source_start_index'])
        if meta['shape_ra'] != [256, 107] or meta['shape_dra'] != [64, 256, 107] or meta['window_mode'] != 'hann_3d':
            raise ValueError(f'unexpected radar shape/window in {product}')
        for offset, item in enumerate(meta['frames']):
            index = int(item['source_index'])
            if index != start + offset:
                raise ValueError(f'nonconsecutive radar metadata in {product}')
            folder = product / 'frames' / f'{offset:06d}'
            if not (folder / 'folded_ra.npz').is_file() or not (folder / 'unfolded_ra.npz').is_file():
                continue
            priority = int(item['carla_frame'] in complete['core_frames'])
            # A repeated completed attempt for the same core can be referenced
            # by the final dataset index. Prefer the later attempt on a tie.
            if index not in products or priority >= products[index][0]:
                products[index] = (priority, folder, product, item)
    return products


def _future(entry, origin, yaw, indices):
    positions, yaws, valid = [], [], []
    if entry is None:
        return [[0., 0.] for _ in indices], [0. for _ in indices], [False for _ in indices]
    source_points, source_valid = entry['world_xyzyaw'], entry['valid']
    if len(source_points) != 30 or len(source_valid) != 30:
        raise ValueError('future must have exactly 30 observed slots')
    for i in indices:
        ok = bool(source_valid[i])
        point = source_points[i]
        positions.append(_to_ego(point[:2], origin, yaw) if ok else [0., 0.])
        yaws.append(-math.radians(point[3] - yaw) if ok else 0.)
        valid.append(ok)
    return positions, yaws, valid


def _radar_pose(frame, origin, yaw, sensor_x):
    pose = frame['ego']['pose']
    old_yaw = float(pose['rotation_deg'][2])
    forward = (math.cos(math.radians(old_yaw)), math.sin(math.radians(old_yaw)))
    sensor_world = [pose['position_m'][0] + sensor_x * forward[0],
                    pose['position_m'][1] + sensor_x * forward[1]]
    x, y = _to_ego(sensor_world, origin, yaw)
    angle = math.radians(yaw - old_yaw)
    c, s = math.cos(angle), math.sin(angle)
    return [[c, -s, 0., x], [s, c, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]]


@lru_cache(maxsize=1)
@lru_cache(maxsize=1)
def collector_geometry():
    """Use the original collection geometry without importing CARLA."""
    import importlib.util
    path = Path(__file__).resolve().parents[1] / 'collection/collector/road_map.py'
    spec = importlib.util.spec_from_file_location('radarvla_collector_geometry', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _radar_sensor_velocity(frame, current_yaw, sensor_x):
    """World rigid-body velocity, resolved in the current ego ground frame.

    A numeric sensor_x retains the historical planar helper interface. A mount
    dictionary uses the complete offset and observed world angular velocity.
    """
    ego = frame['ego']
    if isinstance(sensor_x, dict):
        rotation = collector_geometry().carla_rotation_matrix(ego['pose']['rotation_deg'])
        offset = rotation @ np.array([sensor_x[k] for k in ('x', 'y', 'z')], float)
        omega = np.radians(ego['angular_velocity_world_degps'])
        velocity = np.asarray(ego['velocity_world_mps']) + np.cross(omega, offset)
        return _vector_to_ego(velocity[:2], current_yaw)
    heading = math.radians(float(ego['pose']['rotation_deg'][2]))
    omega = math.radians(float(ego['angular_velocity_world_degps'][2]))
    vx, vy = ego['velocity_world_mps'][:2]
    return _vector_to_ego([vx - omega*sensor_x*math.sin(heading),
                           vy + omega*sensor_x*math.cos(heading)], current_yaw)


def radar_bev_pose(frame, origin, current_yaw, mount):
    """Planar projection of the actual sensor mount; arrays stay untouched."""
    position, rotation = collector_geometry().radar_pose(frame['ego']['pose'], mount)
    x, y = _to_ego(position[:2], origin, current_yaw)
    forward = _vector_to_ego(rotation[:2, 0], current_yaw)
    if math.hypot(*forward) < 1e-8:
        raise ValueError('Radar heading has no BEV projection')
    angle = math.atan2(forward[1], forward[0]); c, sn = math.cos(angle), math.sin(angle)
    return [[c, -sn, 0., x], [sn, c, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]]


def agent50_label(agent, raw_actor=None):
    """Relabel an already admitted instance; never decide actor membership."""
    result = dict(agent)
    result['source_category'] = agent.get('source_category', agent.get('category', 'unknown'))
    result['category'] = 'agent'
    if raw_actor is not None:
        for key in ('type_id', 'actor_type', 'role_name'):
            if key in raw_actor:
                result[key] = raw_actor[key]
    return result


def _bbox_world_xy(pose_xy, actor_yaw_deg, center_actor):
    c, s = math.cos(math.radians(actor_yaw_deg)), math.sin(math.radians(actor_yaw_deg))
    return [pose_xy[0] + c*center_actor[0] - s*center_actor[1],
            pose_xy[1] + s*center_actor[0] + c*center_actor[1]]


def _agent(actor, anchor, origin, yaw, future, indices):
    pose = actor['pose']
    bbox = actor['bbox']
    center_actor = bbox['center_actor_m']
    actor_yaw = float(pose['rotation_deg'][2])
    position = _to_ego(_bbox_world_xy(pose['position_m'], actor_yaw, center_actor), origin, yaw)
    velocity = _vector_to_ego(actor['velocity_world_mps'], yaw)
    fx, fyaw, mask = _future(future, origin, yaw, indices)
    fx30, fyaw30, mask30 = _future(future, origin, yaw, list(range(30)))
    heading_offset = math.radians(float(bbox['rotation_actor_deg'][2]))
    fyaw = [angle - heading_offset if valid else 0. for angle, valid in zip(fyaw, mask)]
    fyaw30 = [angle - heading_offset if valid else 0. for angle, valid in zip(fyaw30, mask30)]
    if future is not None:
        for output, valid, selections in ((fx, mask, indices), (fx30, mask30, range(30))):
            for j, step in enumerate(selections):
                if valid[j]:
                    point = future['world_xyzyaw'][step]
                    output[j] = _to_ego(_bbox_world_xy(point[:2], point[3], center_actor), origin, yaw)
    extent = bbox['extent_m']
    return {'id': str(actor['actor_id']), 'track_key': f"{anchor['episode_id']}:{actor['actor_id']}",
            'category': actor['actor_type'],
            **{key: actor[key] for key in ('type_id', 'actor_type', 'role_name') if key in actor},
            'position': position, 'velocity': velocity,
            'size': [2*extent[0], 2*extent[1]],
            'heading': -math.radians(actor_yaw - yaw) - heading_offset,
            'future_xy': fx, 'future_yaw': fyaw, 'future_valid': mask,
            'future30_xy': fx30, 'future30_yaw': fyaw30, 'future30_valid': mask30}


def _nearest_planner_ids(agents, limit=8):
    return [agent['id'] for agent in sorted(
        agents, key=lambda agent: (math.hypot(*agent['position']), int(agent['id'])))[:limit]]


def _build_one(row, job_dir, product_frames, anchors, output, cached_radar, *, taskwise=False):
    episode = job_dir / 'episode'
    index = int(row['source_index'])
    load_frame = _load_taskwise_frame if taskwise else _load_frame
    current = load_frame(episode, index)
    if current['carla_frame'] != row['carla_frame'] or current['episode_id'] != row['episode_id']:
        raise ValueError('selected anchor/frame mismatch')
    anchor = anchors.get(current['carla_frame'])
    if taskwise and (anchor is None or anchor.get('future_steps')!=30 or abs(anchor.get('future_dt_s',0)-.1)>1e-6 or abs(anchor.get('timestamp_s',float('inf'))-current['timestamp_s'])>1e-5):
        anchor=dict(timestamp_s=current['timestamp_s'],futures_by_actor_id={},future_steps=30,future_dt_s=.1)
    if anchor is None or anchor['future_steps'] != 30 or abs(anchor['future_dt_s'] - .1) > 1e-6:
        raise ValueError('missing real 30-frame anchor')
    if abs(anchor['timestamp_s'] - current['timestamp_s']) > 1e-5:
        raise ValueError('anchor timestamp mismatch')
    history_indices = list(range(index-3, index+1))
    if history_indices[0] < 0 or any(i not in product_frames for i in history_indices):
        raise ValueError('missing four causal radar products')
    history = [load_frame(episode, i) for i in history_indices]
    if any(history[j+1]['carla_frame'] != history[j]['carla_frame'] + 1 for j in range(3)):
        raise ValueError('nonconsecutive CARLA history frames')
    if any(product_frames[i][3]['carla_frame'] != frame['carla_frame'] for i, frame in zip(history_indices, history)):
        raise ValueError('radar/episode CARLA frame mismatch')
    if 'radar_product' in row:
        _, indexed_folder, indexed_product, _ = product_frames[index]
        if (Path(row['radar_product']).resolve() != indexed_product.resolve() or
                int(row['radar_frame_index']) != int(indexed_folder.name)):
            raise ValueError('final dataset index points to a different radar product/frame')
    ego = current['ego']
    origin = ego['pose']['position_m'][:2]
    yaw = float(ego['pose']['rotation_deg'][2])
    indices = [4, 9, 14, 19, 24, 29]
    future = anchor['futures_by_actor_id']
    ego_future, _, ego_valid = _future(future.get(str(ego['actor_id'])), origin, yaw, indices)
    ego_future30, _, ego_valid30 = _future(future.get(str(ego['actor_id'])), origin, yaw, list(range(30)))
    if not all(ego_valid30) and not taskwise:
        raise ValueError('incomplete ego 30-frame future')
    agent_path=_frame_path(episode,index)/'agents.json';map_path=_frame_path(episode,index)/'map.json'
    agent_record = _read_json(agent_path) if agent_path.is_file() or not taskwise else dict(carla_frame=None,agents=[])
    map_record = _read_json(map_path) if map_path.is_file() or not taskwise else dict(carla_frame=None)
    agent_known=agent_record['carla_frame']==current['carla_frame']
    road_known=map_record['carla_frame']==current['carla_frame']
    if not taskwise and (not agent_known or not road_known):raise ValueError('agent/map CARLA frame mismatch')
    segments = road_segments(map_record) if road_known else []
    if not taskwise and (not segments or any(not segment['width_valid'] for segment in segments)):
        raise ValueError('missing or invalid segmented road width')
    official_route = _read_json(episode / 'route.json')
    if official_route.get('source') != 'Leaderboard.set_global_plan full dense world route':
        raise ValueError('route archive is not the official dense global plan')
    route_ids = [(waypoint['road_id'], waypoint['section_id']) for waypoint in official_route['waypoints']]
    progress = int(current['route_progress_index'])
    if not 0 <= progress < len(route_ids):
        raise ValueError('route progress index outside official route')
    # Navigation announces an upcoming maneuver, while route_progress_index
    # points at the current waypoint. Keep the lookahead semantics intact.
    if current['navigation']['road_option'] not in {
            waypoint['road_option'] for waypoint in official_route['waypoints'][progress:]}:
        raise ValueError('navigation RoadOption absent from remaining official route')
    position = progress
    for segment in segments:
        identity = (segment['road_id'], segment['section_id'])
        while position < len(route_ids) and route_ids[position] != identity:
            position += 1
        if position >= len(route_ids):
            raise ValueError('road segment is absent from official route order')
    for segment in segments:
        segment['centerline_ego_m'] = [[point[0], -point[1]] for point in segment['centerline_ego_m']]
    if len(segments) > 8 and not taskwise:
        raise ValueError('more than eight road segments')
    actors = agent_record.get('observed_agents', agent_record['agents']) if agent_known else []
    duplicate_ids=[]
    if len({actor['actor_id'] for actor in actors}) != len(actors):
        if not taskwise:raise ValueError('duplicate actor ID')
        deduplicated={};conflict=False
        for actor in actors:
            identifier=actor['actor_id']
            if identifier in deduplicated:
                duplicate_ids.append(str(identifier));conflict |= deduplicated[identifier]!=actor
            deduplicated[identifier]=actor
        actors=list(deduplicated.values()) if not conflict else []
        agent_known=agent_known and not conflict
    agents = [agent50_label(_agent(actor, current, origin, yaw, future.get(str(actor['actor_id'])), indices), actor) for actor in actors]
    planner_agent_ids = _nearest_planner_ids(agents)
    spatial_removed=[]
    if taskwise:
        metadata=_read_json(episode/'metadata.json')
        if metadata['episode_id']!=current['episode_id'] or metadata['agent_label_range_m']!=50:raise ValueError('Original Agent50 metadata mismatch')
        geom=collector_geometry();sensor_pos,sensor_rot=geom.radar_pose(current['ego']['pose'],metadata['sensors']['semantic_lidar'])
        spatial_removed=[str(actor['actor_id']) for actor in actors if not geom.actor_intersects_fov(actor,sensor_pos,sensor_rot,50.)]
        agents=[agent for agent in agents if str(agent['id']) not in spatial_removed]
        planner_agent_ids=[identifier for identifier in planner_agent_ids if str(identifier) not in spatial_removed]
    try:coverage = _coverage_from_snapshots(episode, index, current, anchor)
    except ValueError:
        if not taskwise:raise
        coverage=[False]*6
    ego_speed = math.hypot(*ego['velocity_world_mps'][:2])
    ego_extent = ego['bbox']['extent_m']
    risk_actors = [actor for actor in current['actors'] if _could_affect_risk(
        actor, future.get(str(actor['actor_id'])), origin, ego_speed, ego_extent)]
    risk_agents = [_agent(actor, current, origin, yaw,
                          future.get(str(actor['actor_id'])), indices) for actor in risk_actors]
    power_paths, doppler_paths, mask_planes = [], [], []
    for source_index in history_indices:
        _, folder, product, _ = product_frames[source_index]
        key = (row['job_id'], source_index)
        if key not in cached_radar:
            folded = np.load(folder / 'folded_ra.npz')
            unfolded = np.load(folder / 'unfolded_ra.npz')
            axes = np.load(product / 'radar_axes.npz')
            if folded['power_peak_ra'].shape != (256, 107) or unfolded['doppler_unfolded_dense_ra_mps'].shape != (256, 107):
                raise ValueError('radar RA shape mismatch')
            # C-Shenron azimuth is positive to the right; model y is positive left.
            power = np.asarray(folded['power_peak_ra'][:, ::-1], np.float32)
            doppler = np.asarray(unfolded['doppler_unfolded_dense_ra_mps'][:, ::-1], np.float32)
            valid = np.asarray(unfolded['reliable_mask'][:, ::-1], bool)
            if not np.isfinite(power).all() or not np.isfinite(doppler).all() or (power < 0).any():
                raise ValueError('invalid radar product values')
            if not np.allclose(axes['azimuth_deg'], np.arange(-53, 54)):
                raise ValueError('unexpected azimuth axis')
            destination = output / 'radar' / row['job_id']
            destination.mkdir(parents=True, exist_ok=True)
            base = destination / f'{source_index:06d}'
            np.save(base.with_name(base.name + '_power.npy'), power)
            np.save(base.with_name(base.name + '_doppler.npy'), doppler)
            np.save(base.with_name(base.name + '_valid.npy'), valid)
            cached_radar[key] = (base, np.asarray(axes['range_m'], float).tolist(),
                                 np.deg2rad(-np.asarray(axes['azimuth_deg'])[::-1]).tolist())
        base, ranges, azimuth = cached_radar[key]
        power_paths.append(os.path.relpath(base.with_name(base.name + '_power.npy'), output))
        doppler_paths.append(os.path.relpath(base.with_name(base.name + '_doppler.npy'), output))
        mask_planes.append(np.load(base.with_name(base.name + '_valid.npy'), allow_pickle=False))
    sample_id = f"{row['episode_id']}_{current['carla_frame']}"
    mask_path = output / 'radar_masks' / f'{sample_id}.npy'
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, np.stack(mask_planes))
    episode_metadata = _read_json(episode / 'metadata.json')
    mount = episode_metadata['sensors']['semantic_lidar']
    annotation_domain = dict(type='radar_bev_sector',
                             range_m=float(episode_metadata['agent_label_range_m']),
                             azimuth_min_rad=-math.radians(53.), azimuth_max_rad=math.radians(53.),
                             source='episode metadata agent_label_range_m and collector bbox/FOV contract')
    detection_domain_values(annotation_domain)
    poses = [radar_bev_pose(frame, origin, yaw, mount) for frame in history]
    times = [float(frame['timestamp_s'] - current['timestamp_s']) for frame in history]
    if abs(times[-1]) > 1e-6 or any(times[j+1] <= times[j] for j in range(3)):
        raise ValueError('invalid causal timestamps')
    sensor_velocities = [_radar_sensor_velocity(frame, yaw, mount) for frame in history]
    return {'schema_version': 'radar_vla_v3', 'grounding_label_schema': 'agent50_v1', 'sample_id': sample_id,
            'preparation_contract': PREPARATION_CONTRACT,
            'detection_annotation_domain': annotation_domain,
            'scene_id': row['job_id'], 'split': row['split'], 'timestamp_s': current['timestamp_s'],
            'coordinate_frame': 'current_ego', 'episode_id': row['episode_id'],
            'carla_frame': current['carla_frame'], 'category': row['category'],
            'weather': row['weather'], 'route_id': _read_json(episode / 'metadata.json')['route_id'],
            'route_sha256': _read_json(episode / 'metadata.json')['route_sha256'],
            'route_archive': os.path.relpath(episode/'route.json', output),
            'future_times_s': [.5, 1., 1.5, 2., 2.5, 3.],
            'sensors': {'radar': {'power': power_paths, 'unfolded_doppler': doppler_paths,
                                  'doppler_valid': os.path.relpath(mask_path, output),
                                  'range_m': ranges, 'azimuth_rad': azimuth,
                                  'time_offsets_s': times, 'poses_current_ego': poses,
                                  'sensor_velocity_current_ego': sensor_velocities},
                        'camera': {'front': {'path': os.path.relpath(_frame_path(episode,index)/current['sensors']['camera'], output)}},
                        'lidar': {'pointcloud': os.path.relpath(_frame_path(episode,index)/current['sensors']['semantic_lidar'], output)}},
            'ego': {'velocity': _vector_to_ego(ego['velocity_world_mps'], yaw),
                    'acceleration': _vector_to_ego(ego['acceleration_world_mps2'], yaw),
                    'yaw_rate': -math.radians(ego['angular_velocity_world_degps'][2]),
                    'box_size': [2*ego['bbox']['extent_m'][0], 2*ego['bbox']['extent_m'][1]],
                    'future_xy': ego_future, 'future_valid': ego_valid,
                    'future30_xy': ego_future30, 'future30_valid': ego_valid30},
            'agents': agents, 'planner_agent_ids': planner_agent_ids,
            'risk_agents': risk_agents,
            'agent_supervision_available': agent_known,
            **({'road_supervision_available':road_known,'collection_duplicate_actor_ids':duplicate_ids,'collection_spatial_removed_ids':spatial_removed} if taskwise else {}),
            'tracking_coverage_valid': coverage,
            'tracking_coverage_source': 'recorded full world snapshots and all current actor futures; invalid after entrant/disappearance',
            'road_segments': segments,
            'language': {'instruction': current['navigation']['instruction'],
                         'road_option': current['navigation']['road_option'],
                         'source': current['navigation']['source']}}


def _mode_from_risk(risk, individual, has_agents, *, supervision_protocol=None, set_known=True):
    if supervision_protocol == 'taskwise_local_v1':
        from .task_supervision import local_mode
        mode,_=local_mode(risk,set_known=set_known,agent_count=int(has_agents))
        return None if mode=='UNKNOWN' else mode.lower()
    hazard = (any(risk['pcol_mask'][i] and risk['pcol'][i] >= .5 for i in (0,1)) or
              risk['dmin_mask'] and risk['dmin'] <= .5 or
              risk['areq_mask'] and risk['areq'] >= 3.)
    if hazard:
        visible_hazard = any(
            any(label['pcol_mask'][i] and label['pcol'][i] >= .5 for i in (0,1)) or
            label['dmin_mask'] and label['dmin'] <= .5 or
            label['areq_mask'] and label['areq'] >= 3.
            for label in individual.values())
        return 'short' if visible_hazard else None
    if all(risk['pcol_mask'][:2]) and (not has_agents or risk['dmin_mask'] and risk['areq_mask']):
        return 'long'
    return None


def _attach_supervision(record, *, taskwise=False):
    if taskwise:
        from .task_supervision import attach_local_tasks
        return attach_local_tasks(record)
    risk = build_risk_labels(record)
    individual = {agent['id']: build_risk_labels(dict(
        record, agents=[agent], risk_agents=[agent])) for agent in record['agents']}
    selected = {key: individual[key] for key in record['planner_agent_ids']}
    mode = _mode_from_risk(risk, selected, bool(selected))
    if mode is None:
        hazard = (any(risk['pcol_mask'][i] and risk['pcol'][i] >= .5 for i in (0, 1)) or
                  risk['dmin_mask'] and risk['dmin'] <= .5 or
                  risk['areq_mask'] and risk['areq'] >= 3.)
        if hazard:
            raise ValueError('Risk hazard cannot be attributed to a planner-selected visible agent')
        if not all(risk['pcol_mask'][:2]):
            raise ValueError('Risk safe teacher lacks complete collision coverage')
        raise ValueError('Risk safe teacher lacks valid distance or braking evidence')
    if mode == 'long' and any(not all(agent['future_valid']) for agent in record['agents']
                              if agent['id'] in selected):
        raise ValueError('LONG agent future has invalid query frames')
    record['risk_label'] = risk
    record['agent_risk_labels'] = individual
    record['teacher_mode'] = mode
    if record.get('grounding_label_schema') == 'agent50_v1':
        from .agent50_filter import filter_record
        record, _ = filter_record(record)
    return record


_SAMPLE_ID = re.compile(rb'"sample_id"\s*:\s*"([^"_]+)_(\d+)"')


def _reuse_offsets(previous: Path, selected):
    """Index a completed manifest by source key without retaining its records."""
    report_path, manifest = previous/'report.json', previous/'manifest.jsonl'
    if not report_path.is_file() or not manifest.is_file():
        raise RuntimeError(f'reuse requires a completed prepared manifest: {previous}')
    report = _read_json(report_path)
    if report.get('status') != 'PREPARED' or report.get('source_rows') != len(selected):
        raise RuntimeError('prior prepared report does not cover this candidate count')
    allowed = {(row['episode_id'], int(row['carla_frame'])) for row in selected}
    offsets = {}
    with manifest.open('rb') as stream:
        while line := stream.readline():
            match = _SAMPLE_ID.search(line[:250])
            if not match:
                raise RuntimeError('prior manifest row lacks a parseable source sample_id')
            key = (match.group(1).decode(), int(match.group(2)))
            if key not in allowed or key in offsets:
                raise RuntimeError('prior prepared manifest has an unknown or duplicate source frame')
            offsets[key] = stream.tell()-len(line)
    if len(offsets) != report.get('prepared_rows'):
        raise RuntimeError('prior prepared report and manifest row counts disagree')
    return manifest, offsets


def _reuse_record(record, row, previous: Path, output: Path):
    if record.get("supervision_protocol")=="taskwise_local_v1":raise ValueError("Taskwise records cannot enter legacy preparation reuse")
    """Copy cached arrays; a fresh writer must never mutate an old hardlink."""
    if (record.get('preparation_contract') != PREPARATION_CONTRACT or
            record.get('detection_annotation_domain') is None or
            record.get('grounding_label_schema') != 'agent50_v1' or
            record.get('agent_ignore_policy') != 'pedestrian_cyclist_regions_v1'):
        raise ValueError('prior preparation contract lacks agent50 schema, annotation domain or corrected sensor velocity; rebuild from raw')
    detection_domain_values(record['detection_annotation_domain'])
    if (record.get('schema_version') != 'radar_vla_v3' or
            record.get('episode_id') != row['episode_id'] or
            record.get('carla_frame') != row['carla_frame'] or
            record.get('scene_id') != row['job_id'] or
            any(record.get(key) != row[key] for key in ('split','category','weather'))):
        raise ValueError('prior prepared record does not match the current index row')
    agents = record.get('agents', [])
    if len({agent['id'] for agent in agents}) != len(agents):
        raise ValueError('prior prepared record repeats an agent ID')
    prior_chosen = record.get('planner_agent_ids')
    if record.get('agent_ignore_policy') == 'pedestrian_cyclist_regions_v1':
        if prior_chosen is None or not set(map(str,prior_chosen)) <= {str(a['id']) for a in agents}:
            raise ValueError('Invalid filtered planner selection')
        chosen = prior_chosen
    else:
        chosen = _nearest_planner_ids(agents)
    if prior_chosen is not None and prior_chosen != chosen:
        raise ValueError('prior planner agent selection is incompatible')
    if len(agents) > 8 and prior_chosen is None:
        raise ValueError('prior record omitted visible agents without a planner selection')
    risk = record.get('risk_label')
    individual = record.get('agent_risk_labels', {})
    if risk is None or any(key not in individual for key in chosen):
        raise ValueError('prior prepared Risk cache is incomplete')
    mode = _mode_from_risk(risk, {key:individual[key] for key in chosen}, bool(chosen))
    if record.get('planner_supervision_available',True) and (mode is None or mode != record.get('teacher_mode')):
        raise ValueError('prior Risk teacher is incompatible with the current planner selection')
    paths = [*record['sensors']['radar']['power'],
             *record['sensors']['radar']['unfolded_doppler'],
             record['sensors']['radar']['doppler_valid']]
    for value in paths:
        relative = Path(value)
        if (relative.is_absolute() or '..' in relative.parts or
                relative.parts[0] not in ('radar', 'radar_masks')):
            raise ValueError('prior cached radar path is outside its prepared directory')
        source, destination = previous/relative, output/relative
        if not source.is_file():
            raise ValueError(f'prior cached radar file is missing: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copyfile(source, destination)
    record['planner_agent_ids'] = chosen
    record['prepared_reused_from'] = str(previous)
    return record


def prepare(data_root: Path, *, sample: int | None = None,
            reuse_prepared: Path | None = None, moving_agents: bool = False):
    data_root = Path(data_root).resolve()
    if moving_agents and reuse_prepared is not None:
        raise ValueError('Motion preparation must rebuild target sets; legacy reuse is not supported')
    if reuse_prepared is not None:
        reuse_prepared = Path(reuse_prepared).resolve()
        if sample is not None:
            raise ValueError('prepared reuse is supported only for --full')
    full = data_root / 'full'
    accepted_path = full / 'accepted_anchors.jsonl'
    if not accepted_path.is_file():
        raise FileNotFoundError(accepted_path)
    if sample is None:
        status_path, index_path = full / 'status.json', full / 'dataset_index.jsonl'
        if not status_path.is_file() or not index_path.is_file():
            raise RuntimeError(f'full preparation requires {status_path} and {index_path}')
        if _read_json(status_path).get('status') != 'RAW_RADAR_COMPLETE':
            raise RuntimeError('full raw/radar status is not RAW_RADAR_COMPLETE')
        selected = list(_rows(index_path))
        if len(selected) != 10000:
            raise RuntimeError(f'full dataset index has {len(selected)} rows, expected 10000')
        keys = [(row['episode_id'], row['carla_frame']) for row in selected]
        if len(set(keys)) != len(keys):
            raise RuntimeError('full dataset index repeats an episode/CARLA frame')
    else:
        if sample < 1:
            raise ValueError('--sample must be positive')
        groups = defaultdict(deque)
        available_jobs = {}
        for row in _rows(accepted_path):
            if int(row['source_index']) < 3:
                continue
            job = row['job_id']
            if job not in available_jobs:
                available_jobs[job] = any((full/'jobs'/job/'radar').glob('*/attempt_*/complete.json'))
            if not available_jobs[job]:
                continue
            groups[(row['split'], row['category'], row['weather'])].append(row)
        selected = []
        keys = sorted(groups)
        while keys:
            next_keys = []
            for key in keys:
                selected.append(groups[key].popleft())
                if groups[key]:
                    next_keys.append(key)
            keys = next_keys
    motion_code = b''
    if moving_agents:
        from . import moving_agents as motion
        motion_code = b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py'))) + Path(motion._path).read_bytes()
    source_hash = hashlib.sha256(motion_code + Path(__file__).read_bytes() + Path(road_segments.__code__.co_filename).read_bytes()
                                 + Path(detection_domain_values.__code__.co_filename).read_bytes()).hexdigest()
    input_hash = hashlib.sha256((index_path if sample is None else accepted_path).read_bytes()).hexdigest()
    selected_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
    version = hashlib.sha256(json.dumps({'schema':'radar_vla_v3','source':str(accepted_path),
                                         'sample':sample,'sampling':'stratified_v2','code':source_hash,
                                         'input':input_hash,'selected':selected_hash,
                                         'reuse_prepared':str(reuse_prepared) if reuse_prepared else None},
                                         sort_keys=True).encode()).hexdigest()[:12]
    output = data_root / 'prepared' / f"{'sample' if sample else 'full'}_v3_{version}"
    if reuse_prepared == output:
        raise ValueError('cannot reuse the output directory itself')
    reuse_manifest, reuse_offsets = (_reuse_offsets(reuse_prepared, selected)
                                     if reuse_prepared is not None else (None, {}))
    manifest_partial = output/'manifest.partial.jsonl'
    exclusions_partial = output/'exclusions.partial.jsonl'
    progress_path = output/'progress.json'
    if output.exists():
        report_path = output/'report.json'
        if report_path.is_file() and (output/'manifest.jsonl').is_file():
            return _read_json(report_path)
        if not all(path.is_file() for path in (manifest_partial, exclusions_partial, progress_path)):
            raise RuntimeError(f'prior prepared attempt has no resumable journal: {output}')
        progress = _read_json(progress_path)
        if progress.get('version') != version or not 0 <= progress['next_index'] <= len(selected):
            raise RuntimeError('prepared journal input/code version does not match')
        with manifest_partial.open('r+b') as stream:
            stream.truncate(progress['manifest_bytes'])
        with exclusions_partial.open('r+b') as stream:
            stream.truncate(progress['exclusions_bytes'])
        record_count = exclusion_count = reused_count = 0
        split_counts, category_counts, weather_counts = Counter(), Counter(), Counter()
        full_coverage = any_coverage = 0
        scenes = {}
        for record in _rows(manifest_partial):
            record_count += 1
            reused_count += 'prepared_reused_from' in record
            split_counts[record['split']] += 1
            category_counts[record['category']] += 1
            weather_counts[record['weather']] += 1
            full_coverage += all(record['tracking_coverage_valid'])
            any_coverage += any(record['tracking_coverage_valid'])
            scene = record['scene_id']
            if scene in scenes and scenes[scene] != record['split']:
                raise ValueError(f'cross-split scene leakage: {scene}')
            scenes[scene] = record['split']
        for _ in _rows(exclusions_partial):
            exclusion_count += 1
        start_index = progress['next_index']
        print(f'resuming prepared output at candidate {start_index}; {record_count} records', flush=True)
    else:
        output.mkdir(parents=True)
        manifest_partial.write_bytes(b'')
        exclusions_partial.write_bytes(b'')
        start_index = record_count = exclusion_count = reused_count = full_coverage = any_coverage = 0
        split_counts, category_counts, weather_counts = Counter(), Counter(), Counter()
        scenes = {}
        progress_path.write_text(json.dumps({'version':version,'next_index':0,
                                             'manifest_bytes':0,'exclusions_bytes':0}))
    job_cache, radar_cache = {}, {}
    full_targets = defaultdict(set)
    if sample is None:
        for row in selected[start_index:]:
            full_targets[row['job_id']].add(row['carla_frame'])
    started = time.monotonic()
    with manifest_partial.open('ab') as manifest_stream, exclusions_partial.open('ab') as exclusion_stream, \
            (reuse_manifest.open('rb') if reuse_manifest else open(os.devnull, 'rb')) as reuse_stream:
        for candidate_index in range(start_index, len(selected)):
            if sample is not None and record_count >= sample:
                break
            row = selected[candidate_index]
            job = row['job_id']
            try:
                key = (row['episode_id'], int(row['carla_frame']))
                record = None
                if key in reuse_offsets:
                    reuse_stream.seek(reuse_offsets[key])
                    prior = json.loads(reuse_stream.readline())
                    try:
                        record = _reuse_record(prior, row, reuse_prepared, output)
                    except (ValueError, KeyError, OSError, IndexError):
                        # An incompatible or missing prior asset is processed from
                        # the raw episode rather than silently copied as success.
                        record = None
                if record is None:
                    if job not in job_cache:
                        directory = full / 'jobs' / job
                        products = _product_frames(directory)
                        anchors = (_selected_anchors(directory/'episode'/'anchors.jsonl', full_targets[job])
                                   if products and sample is None else {})
                        job_cache[job] = (directory, anchors, products)
                        if len(job_cache) > 8:
                            job_cache.pop(next(iter(job_cache)))
                    directory, anchors, products = job_cache[job]
                    if sample is not None and row['carla_frame'] not in anchors and products:
                        item = _anchor_at(directory/'episode'/'anchors.jsonl', row['carla_frame'])
                        if item is not None:
                            anchors[row['carla_frame']] = item
                    if moving_agents:
                        record = _build_one(row, directory, products, anchors, output, radar_cache, taskwise=True)
                        index = row['source_index']
                        record = motion.rebuild_agents(record, _load_frame(directory/'episode', index),
                            _read_json(directory/'episode/metadata.json'), anchors[row['carla_frame']])
                        record = motion.attach_supervision(record)
                    else:
                        record = _attach_supervision(_build_one(row, directory, products, anchors, output, radar_cache))
                scene = record['scene_id']
                if scene in scenes and scenes[scene] != record['split']:
                    raise RuntimeError(f'cross-split scene leakage: {scene}')
                manifest_stream.write((json.dumps(record, allow_nan=False)+'\n').encode())
                manifest_stream.flush()
                record_count += 1
                reused_count += 'prepared_reused_from' in record
                split_counts[record['split']] += 1
                category_counts[record['category']] += 1
                weather_counts[record['weather']] += 1
                full_coverage += all(record['tracking_coverage_valid'])
                any_coverage += any(record['tracking_coverage_valid'])
                scenes[scene] = record['split']
                if record_count % 10 == 0:
                    elapsed = max(time.monotonic()-started, 1e-3)
                    rate = (record_count+exclusion_count-start_index)/elapsed
                    eta = max((sample or len(selected))-record_count,0)/max(rate,1e-6)
                    print(f'prepared {record_count} / {sample or len(selected)}; reused {reused_count}; excluded {exclusion_count}; ETA {eta/60:.1f} min', flush=True)
            except (ValueError, KeyError, OSError, IndexError) as exc:
                exclusion = {'job_id': job, 'carla_frame': row['carla_frame'],
                             'reason': f'{type(exc).__name__}: {exc}'}
                exclusion_stream.write((json.dumps(exclusion)+'\n').encode())
                exclusion_stream.flush()
                exclusion_count += 1
            staged = progress_path.with_suffix('.json.inprogress')
            staged.write_text(json.dumps({'version':version, 'next_index':candidate_index+1,
                                          'manifest_bytes':manifest_stream.tell(),
                                          'exclusions_bytes':exclusion_stream.tell()}))
            staged.replace(progress_path)
    if not record_count:
        shutil.copyfile(exclusions_partial, output/'exclusions.jsonl')
        raise RuntimeError(f'no training records; {exclusion_count} exclusions, see {output}')
    for source, target in ((manifest_partial, output/'manifest.jsonl'),
                           (exclusions_partial, output/'exclusions.jsonl')):
        staged = target.with_suffix(target.suffix+'.inprogress')
        shutil.copyfile(source, staged)
        staged.replace(target)
    report = {'status':'PREPARED', 'source_rows':len(selected), 'prepared_rows':record_count,
              'excluded_rows':exclusion_count, 'reused_rows':reused_count,
              'fresh_rows':record_count-reused_count,
              'reuse_source':str(reuse_prepared) if reuse_prepared else None,
              'manifest':str(output/'manifest.jsonl'),
              'split_counts':{s:split_counts[s] for s in ('train','val','test')},
              'category_counts':dict(category_counts),
              'weather_counts':dict(weather_counts),
              'full_risk_coverage':full_coverage,
              'any_risk_coverage':any_coverage,
              'output':str(output)}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',required=True,type=Path)
    parser.add_argument('--sample',type=int)
    parser.add_argument('--reuse-prepared',type=Path)
    parser.add_argument('--moving-agents', action='store_true')
    args=parser.parse_args(argv)
    try:
        print(json.dumps(prepare(args.data_root,sample=args.sample,
                                 reuse_prepared=args.reuse_prepared, moving_agents=args.moving_agents),indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')


if __name__ == '__main__':
    main()
