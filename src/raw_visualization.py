"""CPU-only inspection of collected LiDAR, power and unfolded Doppler in BEV.

All display geometry uses the raw CARLA ego convention: x forward, y right.
This deliberately reads accepted raw frames, including preparation exclusions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


@lru_cache(maxsize=1)
def _geometry():
    # Reuse the collector's tested CARLA rotation, without importing CARLA or
    # changing sys.path. Works with the project's prescribed editable install.
    path = Path(__file__).resolve().parents[1] / 'collection/collector/road_map.py'
    spec = importlib.util.spec_from_file_location('_radarvla_collector_road_map', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotation(value):
    return _geometry().carla_rotation_matrix(value)


def world_to_current(points, ego_pose):
    return (np.asarray(points, float) - ego_pose['position_m']) @ rotation(ego_pose['rotation_deg'])


def sensor_to_ego(points, sensor):
    matrix = rotation([sensor['roll'], sensor['pitch'], sensor['yaw']])
    return np.asarray(points, float) @ matrix.T + [sensor['x'], sensor['y'], sensor['z']]


def _json(path):
    return json.loads(Path(path).read_text())


def _rows(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _check_frame(value, frame_id, path, key='carla_frame'):
    if value.get(key) != frame_id:
        raise ValueError(f'{path}: {key}={value.get(key)}, expected CARLA frame {frame_id}')


class RawSession:
    """Lightweight raw index; selected sensor arrays and futures load on demand."""

    def __init__(self, data_root):
        self.root = Path(data_root).expanduser().resolve()
        self.index = list(_rows(self.root / 'full/dataset_index.jsonl'))
        if not self.index:
            raise ValueError('raw dataset index is empty')
        self._anchor_indices = {}

    def select(self, *, split=None, category=None, weather=None, job_id=None,
               episode_id=None, carla_frame=None):
        filters = dict(split=split, category=category, weather=weather, job_id=job_id,
                       episode_id=episode_id, carla_frame=carla_frame)
        return [r for r in self.index if all(v is None or r.get(k) == v for k, v in filters.items())]

    def product_path(self, row):
        # Index paths can contain the collection machine's absolute prefix.
        # Keep the relative path inside this data root, including attempt ID.
        parts = Path(row['radar_product']).parts
        for i in range(len(parts) - 1):
            if parts[i:i+2] == ('full', 'jobs'):
                return self.root.joinpath(*parts[i:])
        raise ValueError(f"invalid radar_product path: {row['radar_product']}")

    @lru_cache(maxsize=16)
    def _metadata(self, episode):
        return _json(Path(episode) / 'metadata.json')

    @lru_cache(maxsize=16)
    def _radar_meta(self, product):
        path = Path(product)
        if not (path / 'complete.json').is_file():
            raise FileNotFoundError(path / 'complete.json')
        meta = _json(path / 'radar_meta.json')
        if meta.get('product_schema') not in ('dense_hann3d_v1', 'dense_hann3d_v2'):
            raise ValueError(f'{path}: unsupported radar schema')
        with np.load(path / 'radar_axes.npz', allow_pickle=False) as axes:
            ranges, azimuth = axes['range_m'].copy(), axes['azimuth_deg'].copy()
        for a in (ranges, azimuth):
            if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not (np.diff(a) > 0).all():
                raise ValueError(f'{path}: invalid radar axes')
        return meta, ranges, azimuth

    def _anchor(self, episode, frame_id):
        path = episode / 'anchors.jsonl'
        if episode not in self._anchor_indices:
            offsets = {}
            with path.open('rb') as stream:
                while True:
                    offset = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    match = re.search(rb'"carla_frame"\s*:\s*(\d+)', line[:160])
                    if not match:
                        raise ValueError(f'{path}: anchor without leading CARLA frame at byte {offset}')
                    fid = int(match[1])
                    if fid in offsets:
                        raise ValueError(f'{path}: duplicate CARLA frame {fid}')
                    offsets[fid] = offset
            self._anchor_indices[episode] = offsets
        offset = self._anchor_indices[episode].get(frame_id)
        if offset is None:
            raise ValueError(f'{path}: missing CARLA frame {frame_id}')
        with path.open('rb') as stream:
            stream.seek(offset)
            anchor = json.loads(stream.readline())
        _check_frame(anchor, frame_id, path)
        if anchor.get('future_steps') != 30 or not np.isclose(anchor.get('future_dt_s', 0), .1):
            raise ValueError(f'{path}: frame {frame_id}: expected 30 future points at 0.1 s')
        return anchor

    def load(self, row):
        fid = row['carla_frame']
        episode = self.root / 'full/jobs' / row['job_id'] / 'episode'
        folder = episode / 'frames' / f"{row['source_index']:06d}"
        try:
            meta = self._metadata(episode)
            if meta['episode_id'] != row['episode_id']:
                raise ValueError(f'{episode}/metadata.json: episode_id mismatch')
            frame = _json(folder / 'frame.json')
            for key in ('carla_frame', 'lidar_frame', 'camera_frame'):
                _check_frame(frame, fid, folder / 'frame.json', key)
            if frame['index'] != row['source_index'] or frame['episode_id'] != row['episode_id']:
                raise ValueError(f'{folder}/frame.json: source_index or episode_id mismatch')
            agents = _json(folder / frame['agents'])
            road = _json(folder / frame['map'])
            _check_frame(agents, fid, folder / frame['agents'])
            _check_frame(road, fid, folder / frame['map'])
            list(road_pieces(road['map']))  # Validate before allocating a figure.
            lidar = np.load(folder / frame['sensors']['semantic_lidar'], allow_pickle=False)
            if lidar.ndim != 2 or lidar.shape[1] != 6 or not np.isfinite(lidar).all():
                raise ValueError(f'{folder}: malformed semantic LiDAR')
            product = self.product_path(row)
            radar_meta, ranges, azimuth = self._radar_meta(product)
            ri = row['radar_frame_index']
            radar_frame = radar_meta['frames'][ri]
            _check_frame(radar_frame, fid, product / 'radar_meta.json')
            if radar_frame.get('source_index') != row['source_index']:
                raise ValueError(f'{product}: radar source_index mismatch')
            rf = product / 'frames' / f'{ri:06d}'
            with np.load(rf / 'folded_ra.npz', allow_pickle=False) as data:
                power = data['power_peak_ra'].copy()
            with np.load(rf / 'unfolded_ra.npz', allow_pickle=False) as data:
                speed = data['doppler_unfolded_dense_ra_mps'].copy()
                reliable = data['reliable_mask'].copy()
            shape = (len(ranges), len(azimuth))
            if any(a.shape != shape or not np.isfinite(a).all() for a in (power, speed, reliable)):
                raise ValueError(f'{rf}: invalid radar array shape or nonfinite value')
            if (power < 0).any() or not np.isin(reliable, [0, 1]).all():
                raise ValueError(f'{rf}: invalid power or reliability mask')
            peak = float(radar_meta['global_power_peak'])
            if not np.isfinite(peak) or peak <= 0:
                raise ValueError(f'{product}: invalid global_power_peak')
            anchor = self._anchor(episode, fid)
            # Validate futures for every visible actor, including those outside
            # a user's current zoom. No extrapolation for missing observations.
            for actor in [frame['ego'], *agents['agents']]:
                future_xy(anchor['futures_by_actor_id'][str(actor['actor_id'])],
                          actor['pose']['position_m'], frame['ego']['pose'])
            return dict(row=row, frame=frame, agents=agents['agents'], road=road['map'],
                        lidar=lidar, sensor=meta['sensors']['semantic_lidar'],
                        ranges=ranges, azimuth=azimuth, power=power, speed=speed,
                        reliable=reliable.astype(bool), peak=peak, anchor=anchor)
        except (OSError, ValueError, KeyError, IndexError) as exc:
            raise ValueError(f"job {row['job_id']}, CARLA frame {fid}, raw folder {folder}: {exc}") from exc


def future_xy(future, current_world_xyz, ego_pose):
    """Current point plus 30 observed futures; NaNs preserve invalid gaps."""
    points, valid = future['world_xyzyaw'], future['valid']
    if len(points) != 30 or len(valid) != 30:
        raise ValueError('future must have 30 points and validity flags')
    output = np.full((31, 3), np.nan)
    output[0] = world_to_current(current_world_xyz, ego_pose)
    for i, (point, known) in enumerate(zip(points, valid), 1):
        if type(known) is not bool or (point is not None) != known:
            raise ValueError(f'future point/mask mismatch at step {i}')
        if known:
            if len(point) != 4 or not np.isfinite(point).all():
                raise ValueError(f'invalid future point at step {i}')
            output[i] = world_to_current(point[:3], ego_pose)
    return output[:, :2]


def radar_vertices(ranges, azimuth, sensor):
    """Polar cell edges mapped to ego; plotting does not interpolate values."""
    def edges(axis):
        return np.r_[axis[0]-(axis[1]-axis[0])/2,
                     (axis[1:]+axis[:-1])/2, axis[-1]+(axis[-1]-axis[-2])/2]
    rr, aa = np.meshgrid(np.maximum(edges(ranges), 0), np.deg2rad(edges(azimuth)), indexing='ij')
    points = np.stack((rr*np.cos(aa), rr*np.sin(aa), np.zeros_like(rr)), axis=-1)
    return sensor_to_ego(points, sensor)


def road_pieces(road):
    """Yield raw ego corridor pieces without crossing invalid/topology gaps."""
    if road.get('target_schema') != 'ego_road_corridor_50m_v1':
        raise ValueError(f"unsupported road schema: {road.get('target_schema')}")
    fields = [road[k] for k in ('centerline_ego_m', 'valid', 'break_before', 'width_m', 'source_ids')]
    if any(len(a) != len(fields[0]) for a in fields):
        raise ValueError('road field lengths differ')
    points, widths, ids = [], [], None
    for point, valid, cut, width, source in zip(*fields):
        if type(valid) is not bool or (point is not None) != valid:
            raise ValueError('road point/mask mismatch')
        key = (source.get('road_id'), source.get('section_id')) if source else None
        if cut or not valid or key != ids:
            if points:
                yield np.asarray(points), np.asarray(widths), ids
            points, widths = [], []
        ids = key
        if valid:
            if len(point) != 2 or not np.isfinite(point).all():
                raise ValueError('invalid road point')
            points.append(point)
            widths.append(float(width) if width is not None else np.nan)
    if points:
        yield np.asarray(points), np.asarray(widths), ids


def _box_xy(actor, ego_pose):
    bbox = actor['bbox']
    ex, ey = bbox['extent_m'][:2]
    corners = np.array([[-ex, -ey, 0], [ex, -ey, 0], [ex, ey, 0], [-ex, ey, 0]])
    local = corners @ rotation(bbox['rotation_actor_deg']).T + bbox['center_actor_m']
    world = local @ rotation(actor['pose']['rotation_deg']).T + actor['pose']['position_m']
    return world_to_current(world, ego_pose)[:, :2]


# CARLA CityScapes semantic LiDAR palette, with labels in the exported figure.
_SEMANTIC_RGB = np.array([
    [0,0,0], [70,70,70], [100,40,40], [55,90,80], [220,20,60],
    [153,153,153], [157,234,50], [128,64,128], [244,35,232], [107,142,35],
    [0,0,142], [102,102,156], [220,220,0], [70,130,180], [81,0,81],
    [150,100,100], [230,150,140], [180,165,180], [250,170,30], [110,190,160],
    [170,120,50], [45,60,150], [145,170,100]], dtype=float) / 255


def render(sample, *, reliable_only=False, forward=60., rear=10., side=40., full_range=False):
    """Return a Matplotlib three-panel BEV figure with identical GT overlays."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.lines import Line2D
    if min(forward, rear, side) <= 0:
        raise ValueError('view distances must be positive')
    fig, axes = plt.subplots(1, 3, figsize=(21, 9), constrained_layout=True)
    ego = sample['frame']['ego']
    pose, sensor = ego['pose'], sample['sensor']
    if full_range:
        forward = float(sample['ranges'][-1]) + np.linalg.norm([sensor['x'], sensor['y']]) + 3
        rear, side = 10., forward
    xyz = sensor_to_ego(sample['lidar'][:, :3], sensor)
    tags = sample['lidar'][:, 5].astype(int)
    colors = _SEMANTIC_RGB[np.clip(tags, 0, len(_SEMANTIC_RGB)-1)]
    axes[0].scatter(xyz[:, 1], xyz[:, 0], c=colors, s=.35, alpha=.8, rasterized=True)
    vertices = radar_vertices(sample['ranges'], sample['azimuth'], sensor)
    db = 10*np.log10(np.maximum(sample['power'], np.finfo(np.float32).tiny)/sample['peak'])
    p = axes[1].pcolormesh(vertices[..., 1], vertices[..., 0], db,
                          shading='flat', cmap='viridis', vmin=-60, vmax=0, rasterized=True)
    speed = np.ma.masked_where(~sample['reliable'], sample['speed']) if reliable_only else sample['speed']
    d = axes[2].pcolormesh(vertices[..., 1], vertices[..., 0], speed,
                          shading='flat', cmap='RdBu_r', vmin=-35, vmax=35, rasterized=True)
    fig.colorbar(p, ax=axes[1], label='dB relative to radar-window peak', shrink=.7)
    fig.colorbar(d, ax=axes[2], label='Unfolded radial velocity (m/s; + receding)', shrink=.7)
    axes[0].set_title('Semantic LiDAR BEV')
    axes[1].set_title('Radar Power BEV')
    axes[2].set_title('Unfolded Doppler BEV\n%s | reliable %.2f%%' %
                      ('reliable cells only' if reliable_only else 'all saved values',
                       100*sample['reliable'].mean()))
    roads = list(road_pieces(sample['road']))
    actors = [ego, *sample['agents']]
    palette = plt.get_cmap('tab20')
    for ax in axes:
        ax.set_facecolor('#eeeeee')
        for points, widths, key in roads:
            ax.plot(points[:, 1], points[:, 0], color='#f09000', linewidth=2, zorder=5)
            if len(points) >= 2:
                tangent = np.gradient(points, axis=0)
                lengths = np.linalg.norm(tangent, axis=1)
                normal = np.c_[-tangent[:, 1], tangent[:, 0]] / np.maximum(lengths[:, None], 1e-9)
                widths = np.where((widths > 0) & np.isfinite(widths), widths, np.nan)
                for sign in (-1, 1):
                    edge = points + sign * normal * widths[:, None]/2
                    ax.plot(edge[:, 1], edge[:, 0], '--', color='#f09000', linewidth=1.1, zorder=5)
            ax.annotate(f'road {key[0] if key else "?"}', points[len(points)//2][::-1],
                        xytext=(4, 4), textcoords='offset points', fontsize=8, color='#8c4600', zorder=8, clip_on=True)
        for actor in actors:
            aid = actor['actor_id']
            is_ego = aid == ego['actor_id']
            color = '#00bddd' if is_ego else palette(2 * (int(aid) % 10))
            box = _box_xy(actor, pose)
            ax.add_patch(Polygon(box[:, ::-1], fill=False, edgecolor=color, linewidth=1.8, zorder=7))
            xy = world_to_current(actor['pose']['position_m'], pose)[:2]
            label = 'Ego' if is_ego else f"{actor.get('actor_type', 'agent')} {aid}"
            ax.annotate(label, xy[::-1], xytext=(4, 4), textcoords='offset points',
                        fontsize=8, color=color, zorder=8, clip_on=True,
                        bbox=dict(facecolor='white', alpha=.65, edgecolor='none', pad=1))
            trajectory = future_xy(sample['anchor']['futures_by_actor_id'][str(aid)],
                                   actor['pose']['position_m'], pose)
            ax.plot(trajectory[:, 1], trajectory[:, 0], color=color, linewidth=2 if is_ego else 1.3,
                    marker='.', markersize=2.5, zorder=6)
            direction = rotation(actor['pose']['rotation_deg'])[:, 0] @ rotation(pose['rotation_deg'])
            ax.arrow(xy[1], xy[0], direction[1]*2, direction[0]*2, width=.05,
                     head_width=.5, color=color, zorder=7, length_includes_head=True)
        for radius, style in [(50., '--'), (float(sample['ranges'][-1]), ':')]:
            angles = np.deg2rad(np.linspace(sample['azimuth'][0], sample['azimuth'][-1], 121))
            arc = sensor_to_ego(np.c_[radius*np.cos(angles), radius*np.sin(angles), np.zeros(121)], sensor)
            ax.plot(arc[:, 1], arc[:, 0], style, color='#555555', linewidth=.8, zorder=4)
        for angle in np.deg2rad([sample['azimuth'][0], sample['azimuth'][-1]]):
            edge = sensor_to_ego([[0, 0, 0], [sample['ranges'][-1]*np.cos(angle),
                                                      sample['ranges'][-1]*np.sin(angle), 0]], sensor)
            ax.plot(edge[:, 1], edge[:, 0], ':', color='#555555', linewidth=.8, zorder=4)
        ax.scatter(sensor['y'], sensor['x'], marker='x', color='black', s=22, zorder=9)
        ax.set(xlim=(-side, side), ylim=(-rear, forward), aspect='equal',
               xlabel='ego right (m)', ylabel='ego forward (m)')
        ax.grid(alpha=.15)
    legend = [Line2D([], [], color='#00bddd', label='Ego + observed future (3 s)'),
              Line2D([], [], color='tab:blue', label='Agent ID color + observed future (3 s)'),
              Line2D([], [], color='#f09000', label='Road center / dashed width boundaries'),
              Line2D([], [], color='#555555', linestyle='--', label='50 m annotation range'),
              Line2D([], [], color='#555555', linestyle=':', label='Radar FOV / range')]
    fig.legend(handles=legend, loc='outside lower center' if tuple(int(x) for x in plt.matplotlib.__version__.split('.')[:2]) >= (3, 7) else 'lower center',
               ncol=3, fontsize=9)
    row = sample['row']
    fig.suptitle(f"{row['job_id']} | CARLA {row['carla_frame']} | source {row['source_index']} | "
                 f"{row['split']} | {row['weather']} | agents {len(sample['agents'])}\n"
                 'Raw CARLA ego axes: x forward, y right. Future gaps are not interpolated.\n'
                 'LiDAR: road purple; road line light green; vehicle dark blue; pedestrian red; vegetation green.',
                 fontsize=12)
    return fig


def inventory(data_root):
    session = RawSession(data_root)
    root, rows = session.root, session.index
    products = {str(session.product_path(r)) for r in rows}
    prepared = []
    for folder in sorted((root / 'prepared').glob('*')):
        report = _json(folder / 'report.json') if (folder / 'report.json').is_file() else {}
        reasons = Counter()
        if (folder / 'exclusions.jsonl').is_file():
            for item in _rows(folder / 'exclusions.jsonl'):
                reasons[str(item.get('reason', item.get('reasons', 'unknown')))] += 1
        prepared.append(dict(name=folder.name, path=str(folder), status=report.get('status', 'no completion report'),
                             prepared_rows=report.get('prepared_rows'), excluded_rows=report.get('excluded_rows'),
                             split_counts=report.get('split_counts', {}), exclusion_reasons=dict(reasons)))
    jobs = []
    for job_id in sorted({r['job_id'] for r in rows}):
        selected = [r for r in rows if r['job_id'] == job_id]
        first = selected[0]
        jobs.append(dict(job_id=job_id, episode_id=first['episode_id'],
                         path=str(root/'full/jobs'/job_id), frames=len(selected),
                         category=first['category'], weather=first['weather'],
                         splits=sorted({r['split'] for r in selected}),
                         source_index_min=min(r['source_index'] for r in selected),
                         source_index_max=max(r['source_index'] for r in selected)))
    return dict(data_root=str(root), jobs=jobs, accepted_frames=len(rows),
                indexed_jobs=len({r['job_id'] for r in rows}),
                episodes=len({r['episode_id'] for r in rows}),
                job_directories=sum(p.is_dir() for p in (root / 'full/jobs').iterdir()),
                radar_windows=len(products), missing_completion_markers=[p for p in sorted(products) if not (Path(p)/'complete.json').is_file()],
                counts={key: dict(Counter(r[key] for r in rows)) for key in ('split', 'category', 'weather')},
                prepared=prepared,
                present_top_level=sorted(p.name for p in root.iterdir()),
                directories={
                    'full/dataset_index.jsonl': 'Accepted raw frame index; authoritative entry point',
                    'full/jobs/<job>/episode/frames/<source_index>/': 'frame.json, semantic_lidar.npy, agents.json, map.json, front_rgb.png',
                    'full/jobs/<job>/episode/anchors.jsonl': 'Observed ego/agent futures: 30 x 0.1 s with validity masks',
                    'full/jobs/<job>/radar/<window>/attempt_*/': 'radar_meta.json, radar_axes.npz, frames/*/{folded_ra,unfolded_ra}.npz, power_dra.npy',
                    'prepared/': 'Versioned model preparation products, not additional raw samples',
                    'smoke_02/': 'Collection smoke tests and scenario2 pilot',
                    'qualification/': 'Scenario qualification reports',
                    'model_runs/': 'Model outputs'},
                verification='Index/report/completion-marker inspection only; sensor arrays validated on selected-frame load.')


def inventory_markdown(report):
    lines = ['# RadarVLA data inventory', '', f"Data root: `{report['data_root']}`", '',
             f"Accepted frames: **{report['accepted_frames']}**; indexed jobs / episodes: "
             f"**{report['indexed_jobs']} / {report['episodes']}**; radar windows: **{report['radar_windows']}**.",
             f"All job directories (including failed/unselected): {report['job_directories']}. "
             f"Missing radar completion markers: {len(report['missing_completion_markers'])}.", '',
             'Present top-level entries: ' + ', '.join(f'`{p}`' for p in report['present_top_level']), '',
             'The path patterns below describe the data layout; optional pilot/model folders may be absent.', '',
             '| Relative path | Contents |', '|---|---|']
    lines += [f'| `{k}` | {v} |' for k, v in report['directories'].items()]
    for key, counts in report['counts'].items():
        lines += ['', f'## {key}', '', '| Value | Frames |', '|---|---:|']
        lines += [f'| {k} | {v} |' for k, v in counts.items()]
    lines += ['', '## Prepared versions', '', '| Directory | Status | Rows | Excluded |', '|---|---|---:|---:|']
    for p in report['prepared']:
        lines.append(f"| `{p['name']}` | {p['status']} | {p['prepared_rows']} | {p['excluded_rows']} |")
    lines += ['', '## Indexed jobs (accepted frames only)', '',
              '| Job / episode folder under full/jobs | Category | Weather | Split | Frames |',
              '|---|---|---|---|---:|']
    for job in report['jobs']:
        lines.append(f"| `{job['job_id']}` | {job['category']} | {job['weather']} | {', '.join(job['splits'])} | {job['frames']} |")
    for p in report['prepared']:
        if p['exclusion_reasons']:
            lines += ['', f"### {p['name']} exclusions", '']
            lines += [f'- {k}: {v}' for k, v in p['exclusion_reasons'].items()]
    return '\n'.join(lines + ['', report['verification'], ''])


def build_widget(session, output_dir=None):
    import ipywidgets as w
    from IPython.display import display
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir) if output_dir else session.root / 'visualizations/interactive'
    controls = {}
    for name in ('split', 'category', 'weather', 'episode_id'):
        controls[name] = w.Dropdown(options=[('All', None)] + [(x, x) for x in sorted({r[name] for r in session.index})],
                                   description=name, layout=w.Layout(width='340px'))
    job = w.Dropdown(description='job', layout=w.Layout(width='95%'))
    slider = w.IntSlider(description='Frame', min=0, max=0, continuous_update=False,
                         layout=w.Layout(width='80%'))
    previous, following, save = w.Button(description='Previous'), w.Button(description='Next'), w.Button(description='Save PNG')
    full = w.Checkbox(description='Full radar range', value=False)
    reliable = w.Checkbox(description='Reliable Doppler only', value=False)
    distances = {k: w.BoundedFloatText(value=v, min=1, max=300, description=k, layout=w.Layout(width='210px'))
                 for k, v in [('forward', 60), ('rear', 10), ('side', 40)]}
    out, status = w.Output(), w.HTML()
    state = {'rows': [], 'busy': False, 'figure': None, 'row': None, 'error': False}

    def draw(*_):
        if state['busy']:
            return
        if state['figure'] is not None:
            plt.close(state['figure'])
        state.update(figure=None, row=None, error=False)
        with out:
            out.clear_output(wait=True)
            if not state['rows']:
                print('No matching accepted frames.')
                return
            row = state['rows'][slider.value]
            try:
                sample = session.load(row)
                fig = render(sample, reliable_only=reliable.value, full_range=full.value,
                             **{k: v.value for k, v in distances.items()})
                state.update(figure=fig, row=row)
                display(fig)
                plt.close(fig)
                status.value = f"{len(state['rows'])} frames in selected job | CARLA {row['carla_frame']} | source {row['source_index']}"
            except Exception as exc:
                state['error'] = True
                print(f'{type(exc).__name__}: {exc}')

    def frames(*_):
        if state['busy']:
            return
        state['busy'] = True
        filters = {k: v.value for k, v in controls.items()}
        state['rows'] = sorted(session.select(**filters, job_id=job.value), key=lambda r: r['source_index']) if job.value else []
        slider.value = 0
        slider.max = max(0, len(state['rows'])-1)
        slider.disabled = not state['rows']
        state['busy'] = False
        draw()

    def jobs(*_):
        state['busy'] = True
        rows = session.select(**{k: v.value for k, v in controls.items()})
        options = sorted({r['job_id'] for r in rows})
        old = job.value
        job.options = options
        job.value = old if old in options else (options[0] if options else None)
        state['busy'] = False
        frames()

    def export(_):
        if state['figure'] is None:
            status.value = 'No valid figure to export.'
            return
        output_dir.mkdir(parents=True, exist_ok=True)
        row = state['row']
        suffix = '_reliable' if reliable.value else ''
        path = output_dir / f"{row['job_id']}_{row['carla_frame']}{suffix}.png"
        state['figure'].savefig(path, dpi=140)
        status.value = f'Saved: {path}'

    for control in controls.values():
        control.observe(jobs, names='value')
    job.observe(frames, names='value')
    for control in (slider, full, reliable, *distances.values()):
        control.observe(draw, names='value')
    previous.on_click(lambda _: setattr(slider, 'value', max(0, slider.value-1)))
    following.on_click(lambda _: setattr(slider, 'value', min(slider.max, slider.value+1)))
    save.on_click(export)
    widget = w.VBox([w.HBox(list(controls.values())[:2]), w.HBox(list(controls.values())[2:]), job,
                     slider, w.HBox([previous, following, save]), w.HBox([full, reliable]),
                     w.HBox(list(distances.values())), status, out])
    # Keep state inspectable for notebook diagnostics and offline smoke tests.
    widget.raw_bev_state = state
    jobs()
    return widget


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    inv = sub.add_parser('inventory', help='List raw/prepared data and exclusions')
    inv.add_argument('--data-root', type=Path, required=True)
    inv.add_argument('--output-dir', type=Path)
    ren = sub.add_parser('render', help='Export selected raw frames as three-panel BEV PNG')
    ren.add_argument('--data-root', type=Path, required=True)
    ren.add_argument('--output-dir', type=Path)
    for name in ('split', 'category', 'weather', 'job-id', 'episode-id'):
        ren.add_argument('--'+name)
    ren.add_argument('--carla-frame', type=int)
    ren.add_argument('--count', type=int, default=1)
    ren.add_argument('--reliable-only', action='store_true')
    ren.add_argument('--full-range', action='store_true')
    for key, default in [('forward', 60.), ('rear', 10.), ('side', 40.)]:
        ren.add_argument('--'+key, type=float, default=default)
    args = parser.parse_args(argv)
    if args.command == 'inventory':
        report = inventory(args.data_root)
        print(inventory_markdown(report))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir/'inventory.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
            (args.output_dir/'inventory.md').write_text(inventory_markdown(report))
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if args.count < 1:
        parser.error('--count must be positive')
    session = RawSession(args.data_root)
    rows = session.select(**{key: getattr(args, key) for key in
                             ('split', 'category', 'weather', 'job_id', 'episode_id', 'carla_frame')})
    if not rows:
        parser.error('no matching accepted frames')
    chosen = [rows[i] for i in np.linspace(0, len(rows)-1, min(args.count, len(rows)), dtype=int)]
    dest = args.output_dir or session.root/'visualizations/examples'
    dest.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in chosen:
        sample = session.load(row)
        fig = render(sample, reliable_only=args.reliable_only, full_range=args.full_range,
                     forward=args.forward, rear=args.rear, side=args.side)
        suffix = '_reliable' if args.reliable_only else ''
        path = dest/f"{row['job_id']}_{row['carla_frame']}{suffix}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        reports.append(dict(job_id=row['job_id'], carla_frame=row['carla_frame'], source_index=row['source_index'],
                            radar_product=str(session.product_path(row)), image=str(path),
                            reliable_fraction=float(sample['reliable'].mean()), reliable_only=args.reliable_only))
        print(path)
    (dest/'render_report.json').write_text(json.dumps(reports, indent=2)+'\n')


if __name__ == '__main__':
    main()
