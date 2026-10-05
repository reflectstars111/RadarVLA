"""Complete frame_t sensor archive ingestion and causal RadarVLA preparation.

All transforms are right-handed SI units. T_ego_sensor maps sensor→ego;
 ego.pose maps ego→world. Raw timestamps are seconds in one common clock.
Camera/LiDAR are loaded for offline annotation/teacher use, never silently added
as backbone inputs. Radar cubes are read as calibrated opaque arrays; no FFT or
Doppler de-aliasing is invented without acquisition metadata and a causal prior.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path

import numpy as np

from .coordinates import (validate_transform, transform_points, transform_vectors,
                          transform_yaw, sensor_velocity, xyz)
from .radar_processing import load_numeric, unfold_doppler

GROUND_MOTION_TOL = 1e-4
RAW_SCHEMA_VERSION = 'radar_frame_v2'
PREPARED_SCHEMA_VERSION = 'radar_vla_v2'


def _ground_pose(matrix, name):
    matrix = validate_transform(matrix, name)
    if not np.allclose(matrix[:3, 2], [0., 0., 1.], atol=GROUND_MOTION_TOL, rtol=0.):
        raise ValueError(f'{name} must preserve the vertical axis for planar BEV planning; arbitrary camera/LiDAR extrinsics remain allowed')
    return matrix


def _ground_vector(vector, name):
    vector = xyz(vector, name)
    if vector.shape != (3,) or abs(vector[2]) > GROUND_MOTION_TOL:
        raise ValueError(f'{name} must describe planar ground motion (|z| <= {GROUND_MOTION_TOL})')
    return vector


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _file(root, path):
    if not isinstance(path, str) or not path:
        raise ValueError('sensor file path must be a nonempty string')
    result = (Path(root) / path).resolve()
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def _sensor_metadata(sensor, timestamp, max_skew_s, name, frame_pose):
    measured = _finite_number(sensor.get('timestamp_s'), f'{name}.timestamp_s')
    if abs(measured - timestamp) > max_skew_s:
        raise ValueError(f'{name} timestamp exceeds maximum synchronization skew {max_skew_s}s')
    calibration = validate_transform(sensor.get('T_ego_sensor'), f'{name}.T_ego_sensor')
    if abs(measured-timestamp) > 1e-6:
        if 'ego_pose_at_timestamp' not in sensor:
            raise ValueError(f'{name} asynchronous measurement requires ego_pose_at_timestamp; no silent pose approximation')
        pose_at_measurement = _ground_pose(sensor['ego_pose_at_timestamp'], f'{name}.ego_pose_at_timestamp')
        return np.linalg.inv(frame_pose) @ pose_at_measurement @ calibration
    return calibration


def _frame_transform(frame_name, ego_to_world):
    if frame_name not in ('world', 'ego'):
        raise ValueError('coordinate_frame must be world or ego')
    return np.linalg.inv(ego_to_world) if frame_name == 'world' else np.eye(4)


def _future_labels(trajectory, timestamp, current_position, current_yaw, transform, future_times,
                   max_gap_s=1.1, current_valid=True):
    horizon = len(future_times)
    out_xy, out_yaw, out_valid = np.zeros((horizon, 2)), np.zeros(horizon), np.zeros(horizon, bool)
    if trajectory is None:
        return out_xy.tolist(), out_yaw.tolist(), out_valid.tolist()
    if not isinstance(trajectory, list):
        raise ValueError('future_trajectory must be a timestamped list')
    knots = [(timestamp, np.asarray(current_position, float), float(current_yaw), bool(current_valid))]
    for item in trajectory:
        time = _finite_number(item.get('timestamp_s'), 'future timestamp_s')
        valid = item.get('valid', True)
        if type(valid) is not bool or time <= timestamp:
            raise ValueError('future trajectory requires later timestamps and boolean valid flags')
        position = transform_points(item['position'], transform)
        yaw = float(transform_yaw(_finite_number(item.get('yaw'), 'future yaw'), transform))
        knots.append((time, position, yaw, valid))
    knots.sort(key=lambda row: row[0])
    timestamps = np.asarray([k[0] for k in knots])
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError('duplicate future trajectory timestamps')
    positions = np.asarray([k[1] for k in knots]); yaws = np.unwrap([k[2] for k in knots])
    valid = np.asarray([k[3] for k in knots])
    for i, relative in enumerate(future_times):
        target = timestamp + relative
        right = int(np.searchsorted(timestamps, target))
        if right < len(knots) and abs(timestamps[right] - target) < 1e-7:
            if valid[right]:
                out_xy[i], out_yaw[i], out_valid[i] = positions[right, :2], yaws[right], True
        elif 0 < right < len(knots) and valid[right-1:right+1].all() and timestamps[right]-timestamps[right-1] <= max_gap_s:
            ratio = (target-timestamps[right-1])/(timestamps[right]-timestamps[right-1])
            out_xy[i] = positions[right-1, :2]*(1-ratio) + positions[right, :2]*ratio
            out_yaw[i] = yaws[right-1]*(1-ratio) + yaws[right]*ratio
            out_valid[i] = True
    return out_xy.tolist(), out_yaw.tolist(), out_valid.tolist()


def normalize_map(source, transform):
    if source is None:
        return None
    if not isinstance(source, dict):
        raise ValueError('map must be an object')
    result = copy.deepcopy(source)
    result['coordinate_frame'] = 'current_ego'
    for key in ('lane_centerlines', 'lane_boundaries'):
        lines = source.get(key, [])
        if not isinstance(lines, list):
            raise ValueError(f'map.{key} must be a list')
        identifiers = set()
        for line in result.setdefault(key, []):
            identifier = line.get('id')
            if not isinstance(identifier, (str, int)) or str(identifier) in identifiers:
                raise ValueError(f'map.{key} requires unique ids')
            identifiers.add(str(identifier))
            points = xyz(line.get('points'), f'map.{key}.points')
            if points.ndim != 2 or len(points) < 2 or np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 1e-8):
                raise ValueError('map polyline requires at least two distinct consecutive points')
            line['points'] = transform_points(points, transform).tolist()
    elements = result.setdefault('traffic_elements', [])
    if not isinstance(elements, list):
        raise ValueError('map.traffic_elements must be a list')
    identifiers = set()
    for element in elements:
        if (not isinstance(element.get('id'), (str, int)) or str(element['id']) in identifiers
                or not isinstance(element.get('type'), str) or not element['type']):
            raise ValueError('traffic elements require unique id and type')
        identifiers.add(str(element['id']))
        element['position'] = transform_points(element['position'], transform).tolist()
        if 'heading' in element:
            element['heading'] = float(transform_yaw(element['heading'], transform))
    if 'route' in result:
        route = result['route']
        points = xyz(route.get('centerline'), 'map.route.centerline')
        if points.ndim != 2 or len(points) < 2:
            raise ValueError('route centerline requires >=2 points')
        route['centerline'] = transform_points(points, transform).tolist()
        if 'goal_s_m' in route and _finite_number(route['goal_s_m'], 'route.goal_s_m') <= 0:
            raise ValueError('route goal_s_m must be positive')
    return result


def _polyline_projection(points, query):
    starts, edges = points[:-1], np.diff(points, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    parameter = np.clip(((query-starts)*edges).sum(-1)/(lengths**2), 0, 1)
    projected = starts+parameter[:, None]*edges
    distances = np.linalg.norm(projected-query, axis=1)
    idx = int(distances.argmin())
    arc = np.r_[0., np.cumsum(lengths)]
    return distances[idx], arc[idx]+parameter[idx]*lengths[idx], edges[idx]/lengths[idx]


def road_targets(map_record, *, samples=16, max_length_m=60.):
    """Forward route corridor from linked map boundaries; unknown width stays masked."""
    points = np.zeros((samples, 2), np.float32)
    point_mask = np.zeros(samples, bool)
    result = dict(road_centerline=points, road_centerline_mask=point_mask,
                  road_width=np.zeros(1, np.float32), road_mask=np.zeros(1, bool))
    if not map_record or not map_record.get('lane_centerlines'):
        return result
    choices = []
    route = map_record.get('route_lane_id')
    for line in map_record['lane_centerlines']:
        poly = np.asarray(line['points'], float)[:, :2]
        distance, start_arc, direction = _polyline_projection(poly, np.zeros(2))
        if direction[0] <= 0 or (route is not None and str(line['id']) != str(route)):
            continue
        choices.append((distance, str(line['id']), start_arc, line, poly))
    if not choices:
        return result
    _, _, start, line, poly = min(choices, key=lambda c: (c[0], c[1]))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
    end = min(arc[-1], start+max_length_m)
    if end-start < 1.:
        return result
    sample_arc = np.linspace(start, end, samples)
    points[:] = np.stack([np.interp(sample_arc, arc, poly[:, i]) for i in (0, 1)], -1)
    point_mask[:] = True
    boundaries = {str(b['id']): np.asarray(b['points'], float)[:, :2]
                  for b in map_record.get('lane_boundaries', [])}
    left, right = boundaries.get(str(line.get('left_boundary_id'))), boundaries.get(str(line.get('right_boundary_id')))
    if left is not None and right is not None:
        # Intersect the local centerline normal with each linked boundary. Mere
        # nearest-distance sums can invent a corridor when both lines lie on
        # the same side, or overstate width beyond a boundary endpoint.
        def normal_intersection(boundary, point, normal, side):
            starts, edges = boundary[:-1], np.diff(boundary, axis=0)
            determinant = normal[0]*edges[:, 1]-normal[1]*edges[:, 0]
            relative = starts-point
            usable = np.abs(determinant) > 1e-9
            parameter = np.divide(relative[:, 0]*normal[1]-relative[:, 1]*normal[0], determinant,
                                  out=np.full(len(edges), np.inf), where=usable)
            distance = np.divide(relative[:, 0]*edges[:, 1]-relative[:, 1]*edges[:, 0], determinant,
                                 out=np.full(len(edges), np.inf), where=usable)
            usable &= (parameter >= -1e-6) & (parameter <= 1+1e-6) & (side*distance > 1e-6)
            candidates = distance[usable]
            return float(candidates[np.argmin(np.abs(candidates))]) if len(candidates) else None
        widths = []
        for point in points:
            _, _, tangent = _polyline_projection(poly, point)
            normal = np.array([-tangent[1], tangent[0]])
            ldist, rdist = normal_intersection(left, point, normal, 1.), normal_intersection(right, point, normal, -1.)
            if ldist is None or rdist is None:
                break
            widths.append(ldist-rdist)
        if len(widths) == len(points):
            result['road_width'][0] = float(np.mean(widths))
            result['road_mask'][0] = True
    return result


def load_frame(frame, root, *, future_times_s=(.5, 1., 1.5, 2., 2.5, 3.),
               max_sensor_skew_s=.05, max_future_gap_s=1.1, supervision=True,
               load_auxiliary=True):
    """Validate/read a radar_frame_v2 archive record and normalize to its current ego.

    Returns canonical metadata plus ``sensor_payload`` containing actual loaded
    arrays. Raw camera pixels stay in optical sensor coordinates; LiDAR points
    are transformed to ego coordinates and retain intensity/extra attributes.
    """
    if not isinstance(frame, dict) or frame.get('schema_version') != RAW_SCHEMA_VERSION:
        raise ValueError(f'raw schema_version must be {RAW_SCHEMA_VERSION}')
    for key in ('sample_id', 'scene_id'):
        if not isinstance(frame.get(key), str) or not frame[key]:
            raise ValueError(f'{key} must be nonempty')
    if frame.get('split') not in ('train', 'val', 'test'):
        raise ValueError('split must be train, val or test')
    timestamp = _finite_number(frame.get('timestamp_s'), 'timestamp_s')
    times = np.asarray(future_times_s, float)
    if times.ndim != 1 or len(times) == 0 or np.any(times <= 0) or np.any(np.diff(times) <= 0) or not np.isfinite(times).all():
        raise ValueError('future_times_s must be increasing finite positive times')
    ego = frame.get('ego', {}); pose = _ground_pose(ego.get('pose'), 'ego.pose')
    transform = _frame_transform(frame.get('coordinate_frame'), pose)
    sensors = frame.get('sensors', {}); radar = copy.deepcopy(sensors.get('radar', {}))
    calibration = _sensor_metadata(radar, timestamp, max_sensor_skew_s, 'radar', pose)
    if radar['timestamp_s'] > timestamp+1e-6:
        raise ValueError('radar timestamp is in the future relative to frame; causal inputs required')
    radar['T_frame_ego_sensor'] = calibration.tolist()
    axes = []
    for key in ('range_m', 'azimuth_rad'):
        values = np.asarray(radar.get(key), float)
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
            raise ValueError(f'radar {key} must be a strictly increasing finite vector')
        axes.append(values)
    if np.any(axes[0] < 0):
        raise ValueError('radar range_m must be nonnegative')
    shape = tuple(len(a) for a in axes)
    payload = {}
    radar['power'] = str(_file(root, radar.get('power')))
    power = load_numeric(radar['power'], shape=shape, name='radar.power')
    if np.any(power < 0):
        raise ValueError('radar power must be linear nonnegative power')
    payload['power'] = power.astype(np.float32)
    for key in ('folded_doppler', 'unfolded_doppler', 'doppler_prior', 'doppler_valid'):
        if key in radar:
            radar[key] = str(_file(root, radar[key]))
            if key == 'doppler_valid':
                payload[key] = np.load(radar[key], allow_pickle=False)
                if payload[key].shape != shape:
                    raise ValueError('doppler_valid must have radar shape')
            else:
                payload[key] = load_numeric(radar[key], shape=shape, name=f'radar.{key}')
    if 'unfolded_doppler' not in payload:
        if 'folded_doppler' not in payload or 'doppler_prior' not in payload:
            raise ValueError('radar needs unfolded_doppler or folded_doppler with a causal doppler_prior')
        if radar.get('doppler_prior_source') not in ('causal_tracker', 'multi_prf', 'sensor_firmware'):
            raise ValueError('doppler_prior_source must certify a causal source, never GT/future velocity')
        payload['unfolded_doppler'], payload['doppler_valid'] = unfold_doppler(
            payload['folded_doppler'], payload['doppler_prior'], radar.get('max_unambiguous_velocity'),
            max_prior_error=radar.get('max_prior_error'))
    if 'doppler_valid' not in payload:
        payload['doppler_valid'] = np.ones(shape, bool)
    if np.any((payload['doppler_valid'] != 0) & (payload['doppler_valid'] != 1)):
        raise ValueError('doppler_valid must be binary')
    payload['doppler_valid'] = payload['doppler_valid'].astype(bool)
    # Invalid de-aliasing cells carry zero Doppler and an explicit observation mask.
    payload['unfolded_doppler'] = np.where(payload['doppler_valid'], payload['unfolded_doppler'], 0.).astype(np.float32)
    for key in ('raw_points', 'raw_cube'):
        if key in radar:
            radar[key] = str(_file(root, radar[key]))
            array = load_numeric(radar[key], complex_allowed=key == 'raw_cube', name=key)
            if (key == 'raw_points' and (array.ndim != 2 or array.shape[1] < 3)) or (key == 'raw_cube' and array.ndim < 3):
                raise ValueError(f'{key} has invalid dimensions')
            payload[key] = array
    metadata_sensors = {'radar': radar}
    cameras, camera_arrays = copy.deepcopy(sensors.get('camera', {})), {}
    if not isinstance(cameras, dict):
        raise ValueError('camera must be a dictionary keyed by camera name')
    for name, camera in cameras.items():
        camera_calibration = _sensor_metadata(camera, timestamp, max_sensor_skew_s, f'camera.{name}', pose)
        camera['T_frame_ego_sensor'] = camera_calibration.tolist()
        intrinsics = np.asarray(camera.get('intrinsics'), float)
        if (intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all() or intrinsics[0, 0] <= 0
                or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0., 0., 1.])):
            raise ValueError(f'camera.{name} requires calibrated 3x3 intrinsics')
        camera['path'] = str(_file(root, camera.get('path')))
        if load_auxiliary:
            from PIL import Image
            with Image.open(camera['path']) as image:
                camera_arrays[name] = np.asarray(image.convert('RGB')).copy()
    if cameras:
        metadata_sensors['camera'] = cameras
    payload['camera'] = camera_arrays
    if 'lidar' in sensors:
        lidar = copy.deepcopy(sensors['lidar'])
        lidar_pose = _sensor_metadata(lidar, timestamp, max_sensor_skew_s, 'lidar', pose)
        lidar['T_frame_ego_sensor'] = lidar_pose.tolist()
        lidar['pointcloud'] = str(_file(root, lidar.get('pointcloud')))
        if load_auxiliary:
            points = load_numeric(lidar['pointcloud'], name='lidar.pointcloud')
            if points.ndim != 2 or points.shape[1] < 3:
                raise ValueError('LiDAR pointcloud must be NxD with D>=3 (xyz plus optional attributes)')
            points = points.astype(np.float64, copy=True); points[:, :3] = transform_points(points[:, :3], lidar_pose)
            payload['lidar'] = points
        metadata_sensors['lidar'] = lidar
    vector_transform = _frame_transform(ego.get('vector_frame', frame['coordinate_frame']), pose)
    velocity = transform_vectors(ego.get('velocity'), vector_transform)
    acceleration = transform_vectors(ego.get('acceleration'), vector_transform)
    _ground_vector(velocity, 'ego.velocity'); _ground_vector(acceleration, 'ego.acceleration')
    if velocity.shape != (3,) or acceleration.shape != (3,):
        raise ValueError('ego velocity and acceleration must be 2D or 3D vectors')
    box_size = np.asarray(ego.get('box_size'), float)
    if box_size.shape not in ((2,), (3,)) or not np.isfinite(box_size).all() or np.any(box_size <= 0):
        raise ValueError('ego.box_size must contain positive length,width[,height]')
    normalized_ego = dict(pose=pose.tolist(), velocity=velocity[:2].tolist(), acceleration=acceleration[:2].tolist(),
                          velocity_xyz=velocity.tolist(), acceleration_xyz=acceleration.tolist(),
                          yaw_rate=_finite_number(ego.get('yaw_rate'), 'ego.yaw_rate'), box_size=box_size[:2].tolist())
    if abs(radar['timestamp_s']-timestamp) > 1e-6:
        sensor_ego_pose = validate_transform(radar['ego_pose_at_timestamp'])
        sensor_state = radar.get('ego_state_at_timestamp', {})
        sensor_ego_velocity = transform_vectors(sensor_state.get('velocity_world'), np.linalg.inv(sensor_ego_pose))
        _ground_vector(sensor_ego_velocity, 'radar.ego_state_at_timestamp.velocity_world')
        sensor_yaw_rate = _finite_number(sensor_state.get('yaw_rate'), 'radar.ego_state_at_timestamp.yaw_rate')
        sensor_v = sensor_velocity(sensor_ego_velocity, sensor_yaw_rate, radar['T_ego_sensor'])
        sensor_v = transform_vectors(sensor_v, np.linalg.inv(pose) @ sensor_ego_pose)
    else:
        sensor_v = sensor_velocity(velocity, normalized_ego['yaw_rate'], radar['T_ego_sensor'])
    radar['sensor_velocity_frame_ego'] = sensor_v[:2].tolist()
    future_transform = _frame_transform(ego.get('future_coordinate_frame', frame['coordinate_frame']), pose)
    normalized_ego['future_xy'], normalized_ego['future_yaw'], normalized_ego['future_valid'] = _future_labels(
        ego.get('future_trajectory'), timestamp, [0., 0., 0.], 0., future_transform, times, max_future_gap_s)
    source_agents = frame.get('agents')
    if source_agents is None and supervision:
        raise ValueError('supervised frames require agents tracking coverage; [] means observed empty')
    if source_agents is not None and not isinstance(source_agents, list):
        raise ValueError('agents must be a list')
    normalized_agents, identifiers = [], set()
    for agent in (source_agents or []) + frame.get('_future_entrants', []):
        identifier = agent.get('id')
        if not isinstance(identifier, (str, int)) or str(identifier) in identifiers:
            raise ValueError('agents require unique stable ids')
        identifiers.add(str(identifier))
        agent_transform = _frame_transform(agent.get('coordinate_frame', frame['coordinate_frame']), pose)
        box = agent.get('bbox3d', {})
        center = transform_points(box.get('center'), agent_transform)
        size = np.asarray(box.get('size'), float)
        if center.shape != (3,) or size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('bbox3d requires finite center[3] and positive size[3]')
        yaw = _finite_number(box.get('yaw'), 'bbox3d.yaw')
        if 'heading' in agent and not np.isclose(np.cos(agent['heading']-yaw), 1., atol=1e-6):
            raise ValueError('agent heading disagrees with bbox3d yaw')
        heading = float(transform_yaw(yaw, agent_transform))
        agent_velocity = transform_vectors(agent.get('velocity'), agent_transform)
        _ground_vector(agent_velocity, 'agent.velocity')
        if agent_velocity.shape != (3,):
            raise ValueError('agent.velocity must be a vector')
        normalized = dict(id=str(identifier), bbox3d=dict(center=center.tolist(), size=size.tolist(), yaw=heading),
                          position=center[:2].tolist(), size=size[:2].tolist(), heading=heading,
                          velocity=agent_velocity[:2].tolist(), velocity_xyz=agent_velocity.tolist())
        future_transform = _frame_transform(agent.get('future_coordinate_frame', agent.get('coordinate_frame', frame['coordinate_frame'])), pose)
        normalized['future_xy'], normalized['future_yaw'], normalized['future_valid'] = _future_labels(
            agent.get('future_trajectory'), timestamp, center, heading, future_transform, times, max_future_gap_s,
            current_valid=not agent.get('_future_entrant', False))
        if agent.get('_future_entrant'):
            normalized['present_valid'] = False
        normalized_agents.append(normalized)
    instruction = frame.get('language', {}).get('instruction')
    if not isinstance(instruction, str):
        raise ValueError('language.instruction must be a string')
    map_transform = _frame_transform((frame.get('map') or {}).get('coordinate_frame', frame['coordinate_frame']), pose)
    result = dict(schema_version=PREPARED_SCHEMA_VERSION, sample_id=frame['sample_id'], scene_id=frame['scene_id'],
                  split=frame['split'], timestamp_s=timestamp, coordinate_frame='current_ego', sensors=metadata_sensors,
                  ego=normalized_ego, agents=[a for a in normalized_agents if a.get('present_valid', True)],
                  risk_agents=normalized_agents, agent_supervision_available=source_agents is not None,
                  future_times_s=times.tolist(), language=copy.deepcopy(frame['language']),
                  map=normalize_map(frame.get('map'), map_transform), sensor_payload=payload)
    coverage = frame.get('tracking_coverage_valid')
    if coverage is None:
        complete = frame.get('tracking', {}).get('coverage') == 'complete_relevant_agents'
        coverage = [complete] * len(times)
    if not isinstance(coverage, list) or len(coverage) != len(times) or any(type(v) is not bool for v in coverage):
        raise ValueError('tracking_coverage_valid must be a boolean per future query')
    result['tracking_coverage_valid'] = coverage
    return result


def _read_frames(path):
    frames, identifiers, scenes, timestamps = [], set(), {}, set()
    for lineno, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        frame = json.loads(line)
        if not isinstance(frame, dict):
            raise ValueError(f'frame line {lineno} must be an object')
        sample, scene, split, timestamp = (frame.get(k) for k in ('sample_id', 'scene_id', 'split', 'timestamp_s'))
        if sample in identifiers:
            raise ValueError('duplicate frame sample_id')
        if scene in scenes and scenes[scene] != split:
            raise ValueError('scene occurs across splits')
        key = (scene, timestamp)
        if key in timestamps:
            raise ValueError('duplicate scene timestamp')
        identifiers.add(sample); scenes[scene] = split; timestamps.add(key); frames.append(frame)
    if not frames:
        raise ValueError('raw frame manifest is empty')
    return sorted(frames, key=lambda f: (f['scene_id'], f['timestamp_s']))


def _fill_future_from_tracks(frame, scene_frames, *, future_times_s, max_gap_s):
    """Only label fields use later frames; stable IDs are never inferred by proximity."""
    source = copy.deepcopy(frame)
    horizon_s = max(future_times_s)
    later = [f for f in scene_frames if 0 < f['timestamp_s']-frame['timestamp_s'] <= horizon_s+max_gap_s]
    if 'future_trajectory' not in source['ego']:
        source['ego']['future_coordinate_frame'] = 'world'
        source['ego']['future_trajectory'] = []
        for future in later:
            pose = validate_transform(future['ego']['pose'])
            source['ego']['future_trajectory'].append(dict(timestamp_s=future['timestamp_s'],
                position=pose[:3, 3].tolist(), yaw=float(np.arctan2(pose[1, 0], pose[0, 0])), valid=True))
    current_ids = {str(a['id']) for a in source.get('agents', [])}
    entrants = {}
    for future in later:
        for agent in future.get('agents', []):
            if str(agent['id']) not in current_ids and str(agent['id']) not in entrants:
                transformed = copy.deepcopy(agent)
                pose = validate_transform(future['ego']['pose'])
                transform = pose if agent.get('coordinate_frame', future['coordinate_frame']) == 'ego' else np.eye(4)
                transformed['bbox3d']['center'] = transform_points(agent['bbox3d']['center'], transform).tolist()
                transformed['bbox3d']['yaw'] = float(transform_yaw(agent['bbox3d']['yaw'], transform))
                transformed['velocity'] = transform_vectors(agent['velocity'], transform).tolist()
                transformed['coordinate_frame'] = 'world'; transformed['_future_entrant'] = True
                transformed.pop('heading', None); transformed.pop('future_trajectory', None)
                entrants[str(agent['id'])] = transformed
    source['_future_entrants'] = list(entrants.values())
    for agent in source.get('agents', []) + source['_future_entrants']:
        if 'future_trajectory' in agent:
            continue
        trajectory = []
        for future in later:
            matching = [a for a in future.get('agents', []) if str(a['id']) == str(agent['id'])]
            if matching:
                match = matching[0]
                pose = validate_transform(future['ego']['pose'])
                transform = pose if match.get('coordinate_frame', future['coordinate_frame']) == 'ego' else np.eye(4)
                position = transform_points(match['bbox3d']['center'], transform).tolist()
                yaw = float(transform_yaw(match['bbox3d']['yaw'], transform))
            else:
                position, yaw = [0., 0., 0.], 0.
            trajectory.append(dict(timestamp_s=future['timestamp_s'], position=position, yaw=yaw, valid=bool(matching)))
        agent['future_coordinate_frame'] = 'world'; agent['future_trajectory'] = trajectory
    if 'tracking_coverage_valid' not in source and source.get('tracking', {}).get('coverage') != 'complete_relevant_agents':
        observed_times = np.asarray([frame['timestamp_s']] + [f['timestamp_s'] for f in later])
        observed = np.asarray(['agents' in frame] + ['agents' in f for f in later])
        coverage = []
        for relative in future_times_s:
            target = frame['timestamp_s'] + relative
            right = np.searchsorted(observed_times, target)
            if right < len(observed_times) and abs(observed_times[right]-target) < 1e-7:
                coverage.append(bool(observed[right]))
            else:
                coverage.append(bool(0 < right < len(observed_times) and observed[right-1:right+1].all()
                                     and observed_times[right]-observed_times[right-1] <= max_gap_s))
        source['tracking_coverage_valid'] = coverage
    return source


def prepare_frames(input_jsonl, output_dir, *, history_frames=4,
                   future_times_s=(.5, 1., 1.5, 2., 2.5, 3.), max_gap_s=.5,
                   max_sensor_skew_s=.05, max_future_gap_s=1.1, supervision=True):
    """Build causal same-scene windows, persist native RA tensors and alignment poses.

    Incomplete history and timing gaps are explicitly listed in preparation_report.
    Every supplied camera, LiDAR and radar asset is decoded and validated during
    import, even though only radar enters online training. Existing outputs are
    never overwritten. Future annotations do not enter sensor feature tensors.
    """
    if history_frames < 1 or max_gap_s <= 0 or max_future_gap_s <= 0 or max_sensor_skew_s < 0:
        raise ValueError('invalid history/synchronization configuration')
    source, output = Path(input_jsonl).resolve(), Path(output_dir).resolve()
    frames = _read_frames(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'refusing to overwrite nonempty prepared output: {output}')
    output.mkdir(parents=True, exist_ok=True); (output/'radar').mkdir(exist_ok=True)
    scenes = {}
    for frame in frames:
        scenes.setdefault(frame['scene_id'], []).append(frame)
    records, dropped = [], []
    for scene_frames in scenes.values():
        # Retain only the N-frame sensor window in memory; native 256x107 cubes and
        # optional images/point clouds otherwise make long scenes unbounded in RAM.
        history = []
        for index, raw in enumerate(scene_frames):
            if supervision:
                enriched = _fill_future_from_tracks(raw, scene_frames[index+1:], future_times_s=future_times_s, max_gap_s=max_future_gap_s)
            else:
                enriched = copy.deepcopy(raw)
                enriched['ego'].pop('future_trajectory', None)
                for agent in enriched.get('agents', []):
                    agent.pop('future_trajectory', None)
                enriched['tracking_coverage_valid'] = [False] * len(future_times_s)
            current = load_frame(enriched, source.parent, future_times_s=future_times_s,
                                 max_sensor_skew_s=max_sensor_skew_s, max_future_gap_s=max_future_gap_s,
                                 supervision=supervision)
            history.append(current); history = history[-history_frames:]
            if len(history) < history_frames:
                dropped.append(dict(sample_id=current['sample_id'], reason='insufficient_history')); continue
            if np.any(np.diff([f['timestamp_s'] for f in history]) > max_gap_s):
                dropped.append(dict(sample_id=current['sample_id'], reason='history_time_gap')); continue
            axes = ('range_m', 'azimuth_rad')
            if any(any(h['sensors']['radar'][k] != current['sensors']['radar'][k] for k in axes) for h in history):
                raise ValueError('history radar coordinate axes changed; explicit resampling required')
            record = copy.deepcopy({k: v for k, v in current.items() if k != 'sensor_payload'})
            radar = record['sensors']['radar']
            window_id = f'{len(records):09d}'
            for key in ('power', 'unfolded_doppler', 'doppler_valid'):
                array = np.stack([f['sensor_payload'][key] for f in history])
                filename = Path('radar')/f'{window_id}_{key}.npy'
                np.save(output/filename, array); radar[key] = str(filename)
            if all('folded_doppler' in f['sensor_payload'] for f in history):
                filename = Path('radar')/f'{window_id}_folded_doppler.npy'
                np.save(output/filename, np.stack([f['sensor_payload']['folded_doppler'] for f in history]))
                radar['folded_doppler'] = str(filename)
            radar['time_offsets_s'] = [f['sensors']['radar']['timestamp_s']-current['timestamp_s'] for f in history]
            now_inverse = np.linalg.inv(np.asarray(current['ego']['pose']))
            poses, velocities = [], []
            for f in history:
                ego_transform = now_inverse @ np.asarray(f['ego']['pose'])
                calibration = np.asarray(f['sensors']['radar']['T_frame_ego_sensor'])
                poses.append((ego_transform @ calibration).tolist())
                sensor_v = f['sensors']['radar']['sensor_velocity_frame_ego']
                velocities.append(transform_vectors(sensor_v, ego_transform)[:2].tolist())
            radar['poses_current_ego'] = poses; radar['sensor_velocity_current_ego'] = velocities
            record['provenance'] = dict(raw_manifest=str(source), raw_sample_ids=[f['sample_id'] for f in history],
                                        future_annotation_source='explicit_tracks_or_same_scene_stable_id',
                                        radar_alignment='unwarped_sensor_grid_with_current_ego_geometry')
            records.append(record)
    if not records:
        raise ValueError('no complete valid history windows; check history_frames and timestamps')
    manifest = output/'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')
    report = dict(input_frames=len(frames), output_windows=len(records), dropped=dropped,
                  history_frames=history_frames, max_gap_s=max_gap_s, max_sensor_skew_s=max_sensor_skew_s,
                  max_future_gap_s=max_future_gap_s, future_times_s=list(future_times_s), supervision=supervision,
                  camera_lidar_role='validated_offline_teacher_assets; not_online_model_inputs')
    (output/'preparation_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return manifest
