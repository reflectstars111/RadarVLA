"""Convert validated flat v4 episodes to RadarVLA radar_frame_v2 JSONL.

All published frames are exported. Buffer frames provide causal history and
future annotation context; prepare-frames only emits rows marked sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

VERSION = 'v4_raw_export_4_source_backed_corridor'
FUTURE_STEPS = (5, 10, 15, 20, 25, 30)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def adapter_code_fingerprint():
    root = Path(__file__).resolve().parent
    paths = [root/name for name in ('v4_adapter.py', 'records.py', 'geometry.py')]
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode())
        value.update(digest(path).encode())
    return value.hexdigest()


def planar(full):
    pose = np.asarray(full, float)
    yaw = math.atan2(pose[1, 0], pose[0, 0])
    result = np.eye(4)
    result[:2, :2] = [[math.cos(yaw), -math.sin(yaw)],
                      [math.sin(yaw), math.cos(yaw)]]
    result[:3, 3] = pose[:3, 3]
    return result


def fixed(raw_xyz, inverse):
    return (inverse @ [raw_xyz['x'], -raw_xyz['y'], raw_xyz['z'], 1.])[:3].tolist()


def road_map(source):
    """Keep source road geometry; never choose a route by nearest centerline."""
    lines, bounds = [], []
    for road in source['roads']:
        for index, fragment in enumerate(road['centerline']):
            s = np.asarray(fragment['source_s_m'], float)
            points = np.asarray(fragment['points_xyz_m'], float)
            if len(s) < 2 or np.any(np.diff(s) <= 0):
                continue
            reverse = points[-1, 0] < points[0, 0]
            name = f"{road['id']}:{index}"
            line = {'id': name, 'points': (points[::-1] if reverse else points).tolist()}
            for side in ('left', 'right'):
                source_side = ('right' if side == 'left' else 'left') if reverse else side
                for other in road[source_side + '_boundary']:
                    bs = np.asarray(other['source_s_m'], float)
                    bp = np.asarray(other['points_xyz_m'], float)
                    if len(bs) < 2 or bs[0] > s[0]+1e-5 or bs[-1] < s[-1]-1e-5:
                        continue
                    sampled = np.column_stack([np.interp(s, bs, bp[:, j]) for j in range(3)])
                    ident = name + ':' + side
                    line[side + '_boundary_id'] = ident
                    bounds.append({'id': ident, 'points': (sampled[::-1] if reverse else sampled).tolist()})
                    break
            lines.append(line)
    return {'coordinate_frame': 'ego', 'lane_centerlines': lines,
            'lane_boundaries': bounds, 'traffic_elements': [],
            'road_topology': source.get('topology', [])}


def route_corridor(source, planned, samples=16):
    """Sample the actual global plan; mark width unknown without full source support."""
    path = np.asarray(planned['centerline_xyz_m'], float)
    if len(path) < 2:
        raise ValueError('planned route requires two points')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1))]
    unique = np.r_[True, np.diff(arc) > 1e-6]
    arc, path = arc[unique], path[unique]
    if len(path) < 2 or arc[-1] < 1.:
        raise ValueError('planned route corridor is too short')
    # The radar wedge often clips both outer borders at the sensor origin.
    # Begin at the first few metres of forward visible roadway, while the
    # complete planned route remains separately archived in map.route.
    start = min(3., max(0., float(arc[-1])-1.))
    end = min(float(arc[-1]), start+60.)
    grid = np.linspace(start, end, samples)
    points = np.column_stack([np.interp(grid, arc, path[:, j]) for j in range(3)])
    # Match each route point to a source-supported road segment. A turn may
    # traverse several roads; no nearest-road centerline is used as the route.
    candidates = []
    for road in source['roads']:
        for center in road['centerline']:
            cs = np.asarray(center['source_s_m'], float)
            cp = np.asarray(center['points_xyz_m'], float)
            if len(cs) < 2:
                continue
            side_points = []
            for side in ('left_boundary', 'right_boundary'):
                match = None
                for fragment in road[side]:
                    bs = np.asarray(fragment['source_s_m'], float)
                    if len(bs) >= 2 and bs[0] <= cs[0]+1e-5 and bs[-1] >= cs[-1]-1e-5:
                        bp = np.asarray(fragment['points_xyz_m'], float)
                        match = np.column_stack([np.interp(cs, bs, bp[:, j]) for j in range(2)])
                        break
                side_points.append(match)
            if any(side is None for side in side_points):
                continue
            candidates.append((cp[:, :2], side_points[0], side_points[1]))
    widths = []
    for route_point in points[:, :2]:
        choices = []
        for center, left, right in candidates:
            edges = np.diff(center, axis=0)
            norms = np.sum(edges*edges, axis=1)
            valid = norms > 1e-8
            t = np.divide(np.sum((route_point-center[:-1])*edges, axis=1), norms,
                          out=np.zeros(len(edges)), where=valid)
            t = np.clip(t, 0., 1.)
            projected = center[:-1]+t[:, None]*edges
            distances = np.linalg.norm(projected-route_point, axis=1)
            segment = int(distances.argmin())
            if not valid[segment]:
                continue
            tangent = edges[segment]/math.sqrt(norms[segment])
            normal = np.array([-tangent[1], tangent[0]])
            alpha = t[segment]
            l = left[segment]*(1-alpha)+left[segment+1]*alpha
            r = right[segment]*(1-alpha)+right[segment+1]*alpha
            lo, hi = sorted((float((l-projected[segment])@normal),
                             float((r-projected[segment])@normal)))
            offset = float((route_point-projected[segment])@normal)
            width_here = hi-lo
            if .5 < width_here < 50. and lo-1. <= offset <= hi+1.:
                choices.append((float(distances[segment]), width_here))
        if not choices:
            break
        widths.append(min(choices)[1])
    width = float(np.mean(widths)) if len(widths) == samples else None
    return {'source': 'planned_route', 'centerline': points.tolist(),
            'width_m': width, 'width_valid': width is not None,
            'support_start_m': start, 'support_length_m': end-start}


def actor_row(actor_id, raw, name, frames, index, pose, timestamps, entrant=False,
              present_in_archive=False):
    inverse = np.linalg.inv(pose)
    box = raw['bbox3d']
    yaw0 = math.atan2(pose[1, 0], pose[0, 0])
    def yaw(actor):
        return -math.radians(actor['heading_deg'] + actor['bbox3d']['rotation_local']['yaw'])-yaw0
    velocity = raw['velocity_world_mps']
    velocity = (inverse[:3, :3] @ [velocity['x'], -velocity['y'], 0.]).tolist()
    future = []
    for j in range(index+1, min(len(frames), index+31)):
        actor = frames[j].get(actor_id)
        if actor is not None:
            future.append({'timestamp_s': timestamps[j],
                           'position': fixed(actor['bbox3d']['center_world_m'], inverse),
                           'yaw': yaw(actor), 'valid': True})
        else:
            future.append({'timestamp_s': timestamps[j], 'position': [0., 0., 0.],
                           'yaw': 0., 'valid': False})
    row = {'id': name, 'coordinate_frame': 'ego', 'future_coordinate_frame': 'ego',
           'bbox3d': {'center': fixed(box['center_world_m'], inverse),
                      'size': [2*box['extent_m'][k] for k in ('x','y','z')], 'yaw': yaw(raw)},
           'velocity': velocity, 'future_trajectory': future}
    if entrant:
        row['_future_entrant'] = True
        row['_risk_present_valid'] = bool(present_in_archive)
        row['_known_absent_before_entry'] = not present_in_archive
    return row


def archive_coverage(raw, index, tracked_ids, steps=FUTURE_STEPS):
    """Whether the explicitly selected current risk actors stay archived."""
    coverage, reasons = [], []
    for step in steps:
        end = index + step
        if end >= len(raw):
            coverage.append(False)
            reasons.append({'time_s': step/10., 'reason': 'future_frame_missing'})
            continue
        lost = []
        for ident in set(tracked_ids):
            first = next((j for j in range(index, end+1) if ident in raw[j]), None)
            if first is not None and any(ident not in raw[j] for j in range(first+1, end+1)):
                lost.append(ident)
        coverage.append(not lost)
        if lost:
            reasons.append({'time_s': step/10., 'reason': 'tracked_actor_missing',
                            'actor_ids': sorted(lost)})
    return coverage, reasons


def corridor_source_map(road_graph, ego):
    """Unclipped source-backed road geometry, used only for route width supervision."""
    from geometry_v4 import world_to_fixed_ego
    location = ego['pose']['location']
    origin = np.asarray([location['x'], location['y']], float)
    roads = []
    for road in road_graph['roads']:
        center = np.asarray(road['centerline'], float)
        if np.min(np.linalg.norm(center[:, :2] - origin, axis=1)) > 100.:
            continue
        record = {'id': road['id']}
        for name in ('centerline', 'left_boundary', 'right_boundary'):
            record[name] = [{'source_s_m': road['source_s_m'],
                             'points_xyz_m': world_to_fixed_ego(road[name], ego).tolist()}]
        roads.append(record)
    return {'roads': roads}


def episode_rows(pub, work, split, max_agents):
    meta = yaml.safe_load((pub/'episode.yaml').read_text(encoding='utf-8'))
    if meta.get('schema_version') != 4:
        raise ValueError(f'{pub}: expected published schema 4')
    scenario = read(pub/'scenario.json')
    if not scenario.get('validation', {}).get('passed'):
        raise ValueError(f'{pub}: scenario validation has not passed')
    keys = [f'{i:06d}' for i in range(meta['frames'])]
    docs = {name: read(pub/(name+'.json'))['frames'] for name in ('frame','ego','agents','map','language')}
    if any(list(doc) != keys for doc in docs.values()):
        raise ValueError(f'{pub}: incomplete published frame documents')
    raw = []
    for key in keys:
        path = work/'frames'/key/'agents.json'
        raw.append({a['id']: a for a in read(path)['actors']})
    collector = Path(__file__).resolve().parent/'collection'/'collector'
    if str(collector) not in sys.path:
        sys.path.insert(0, str(collector))
    from road_vector import build_roads
    road_graph = build_roads(read(work/'map_source.json'))
    ids = sorted({a for frame in raw for a in frame})
    name_of = {ident: f'agent_{i+1:06d}' for i, ident in enumerate(ids)}
    id_of = {name: ident for ident, name in name_of.items()}
    cal = read(pub/'calibration.json')
    times = [docs['frame'][key]['timestamp_s'] for key in keys]
    rows, capacity, corridor_report = [], [], []
    for i, key in enumerate(keys):
        frame, ego = docs['frame'][key], docs['ego'][key]
        pose = planar(frame['T_world_ego'])
        current = []
        for entry in docs['agents'][key]:
            ident = id_of.get(entry['id'])
            if ident is None or ident not in raw[i]:
                raise ValueError(f'{pub}/{key}: current actor missing in archive')
            current.append(actor_row(ident, raw[i][ident], entry['id'], raw, i, pose, times))
        if frame['role'] == 'sample' and len(current) > max_agents:
            capacity.append({'sample_id': f'{pub.name}/{key}', 'agent_count': len(current),
                             'configured_model_capacity': max_agents})
        # Radar-conditioned risk uses exactly the current GT-FOV agent set.
        # Their archived futures continue even after leaving FOV; actors that
        # are invisible now are never introduced as future entrants.
        current_ids = {id_of[a['id']] for a in current}
        coverage, reasons = archive_coverage(raw, i, current_ids)
        source_map = docs['map'][key]
        planned = source_map.get('planned_route')
        if planned is None:
            raise ValueError(f'{pub}/{key}: planned route missing; rebuild labels')
        mapped = road_map(source_map)
        mapped['planned_route'] = planned
        clipped_corridor = route_corridor(source_map, planned)
        corridor = clipped_corridor
        support = 'radar_fov_roads'
        if corridor['width_m'] is None:
            source_ego = read(work/'frames'/key/'ego.json')
            corridor = route_corridor(corridor_source_map(road_graph, source_ego), planned)
            support = 'uncropped_map_source_roads' if corridor['width_m'] is not None else 'unknown'
        mapped['route_corridor'] = corridor
        if frame['role'] == 'sample':
            corridor_report.append({'sample_id': f'{pub.name}/{key}',
                                    'width_valid': corridor['width_m'] is not None,
                                    'support': support,
                                    'fov_width_valid': clipped_corridor['width_m'] is not None})
        mapped['route'] = {'centerline': planned['centerline_xyz_m'],
                           'goal_s_m': float(np.linalg.norm(np.diff(np.asarray(planned['centerline_xyz_m'])[:, :2], axis=0), axis=1).sum())}
        future = ego['future_trajectory']
        ego_future = [{'timestamp_s': times[i]+future['time_offsets_s'][j],
                       'position': future['xyz_m'][j] if future['valid'][j] else [0.,0.,0.],
                       'yaw': future['yaw_rad'][j] if future['valid'][j] else 0.,
                       'valid': bool(future['valid'][j])} for j in range(len(future['valid']))]
        def asset(folder, suffix='.npy'):
            return str((pub/folder/(key+suffix)).resolve())
        radar = {'power': asset('radar/power'), 'unfolded_doppler': asset('radar/unfolded_doppler'),
                 'doppler_valid': asset('radar/doppler_valid'),
                 'range_m': cal['radar_axes']['range_m'],
                 'azimuth_rad': np.deg2rad(cal['radar_axes']['azimuth_deg']).tolist(),
                 'timestamp_s': frame['radar_timestamp_s'], 'T_ego_sensor': cal['T_ego_radar'],
                 'doppler_status': {'values_present': True, 'algorithm_validity': 'qc.reliable_mask',
                                    'confidence': 'high_only', 'is_causal': False, 'uses_future': True}}
        rows.append({'schema_version': 'radar_frame_v2', 'sample_id': f'{pub.name}/{key}',
                     'scene_id': pub.name, 'split': split, 'sample_role': frame['role'],
                     'timestamp_s': times[i], 'coordinate_frame': 'ego',
                     'sensors': {'radar': radar,
                                 'camera': {'front_rgb': {'path': asset('camera', '.png'),
                                             'timestamp_s': frame['camera_timestamp_s'],
                                             'intrinsics': cal['camera']['intrinsic'],
                                             'T_ego_sensor': cal['camera']['T_ego_camera']}},
                                 'lidar': {'pointcloud': asset('lidar'),
                                           'timestamp_s': frame['lidar_timestamp_s'],
                                           'T_ego_sensor': np.eye(4).tolist()}},
                     'ego': {'pose': pose.tolist(), 'vector_frame': 'ego',
                             'velocity': ego['velocity_xyz_mps'][:2],
                             'acceleration': ego['acceleration_xyz_mps2'][:2],
                             'yaw_rate': ego['yaw_rate_rad_s'], 'box_size': ego['box_size_lwh_m'],
                             'future_coordinate_frame': 'ego', 'future_trajectory': ego_future},
                     'agents': current, '_future_entrants': [],
                     'tracking_coverage_valid': coverage,
                     'tracking_coverage_unknown_reasons': reasons,
                     'map': mapped, 'language': docs['language'][key],
                     'provenance': {'published_episode': str(pub), 'processing_episode': str(work),
                                    'adapter_version': VERSION,
                                    'adapter_code_sha256': adapter_code_fingerprint()}})
    return rows, capacity, corridor_report


def export_training(episodes, splits, output, max_agents=8, processing_root=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    episodes = [Path(p).resolve() for p in episodes]
    if not episodes or len({p.name for p in episodes}) != len(episodes):
        raise ValueError('Provide at least one uniquely named published episode')
    rows, capacity, sources, corridors = [], [], [], []
    for pub in episodes:
        split = splits.get(pub.name)
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'Explicit train/val/test split required for {pub.name}')
        if processing_root is None:
            work = pub.parent.parent/'processing'/pub.name
        else:
            supplied = Path(processing_root).resolve()
            work = supplied if supplied.name == pub.name and (supplied/'manifest.json').is_file() else supplied/pub.name
        if not (work/'manifest.json').is_file():
            raise FileNotFoundError(f'processing archive required for {pub.name}: {work}')
        metadata = yaml.safe_load((pub/'episode.yaml').read_text(encoding='utf-8'))
        source = read(work/'target_meta_v4.json')
        for key in ('source_sha256', 'radar_sha256', 'code_sha256'):
            if metadata['provenance'][key] != source[key]:
                raise ValueError(f'{pub.name}: source/publication fingerprint mismatch: {key}')
        collector = Path(__file__).resolve().parent/'collection'/'collector'
        if str(collector) not in sys.path:
            sys.path.insert(0, str(collector))
        from episode_v4 import validate as validate_published
        validate_published(pub, work)
        converted, conflicts, diagnostics = episode_rows(pub, work, split, max_agents)
        rows.extend(converted); capacity.extend(conflicts); corridors.extend(diagnostics)
        sources.append({'episode': pub.name, 'split': split,
                        'episode_metadata_sha256': digest(pub/'episode.yaml'),
                        'source_sha256': source['source_sha256'],
                        'radar_sha256': source['radar_sha256']})
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.'+output.name+'.export-', dir=output.parent))
    try:
        manifest = stage/'frames.jsonl'
        manifest.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
        report = {'schema': 'radar_frame_v2', 'adapter_version': VERSION,
                  'adapter_code_sha256': adapter_code_fingerprint(),
                  'source_episodes': sources, 'frames': len(rows),
                  'formal_samples': sum(r['sample_role'] == 'sample' for r in rows),
                  'capacity_conflicts': capacity, 'excluded': [],
                  'route_corridor_diagnostics': corridors,
                  'manifest_sha256': digest(manifest),
                  'doppler_status': 'offline_future_assisted; QC high confidence validity mask'}
        (stage/'conversion_report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        os.replace(stage, output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', action='append', type=Path)
    parser.add_argument('--dataset-root', type=Path)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-agents', type=int, default=8)
    parser.add_argument('--processing-root', type=Path,
                        help='Processing episode or directory containing same-named episodes')
    args = parser.parse_args()
    episodes = list(args.episode or [])
    if args.dataset_root:
        episodes.extend(sorted(p for p in args.dataset_root.iterdir() if (p/'episode.yaml').is_file()))
    print(json.dumps(export_training(episodes, read(args.splits), args.output,
                                     args.max_agents, args.processing_root), indent=2))


if __name__ == '__main__':
    main()
