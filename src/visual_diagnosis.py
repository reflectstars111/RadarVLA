"""Interactive validation-frame visualization for Grounding and SFT outputs."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .categories import category_name, record_class_schema
from .task_supervision import supervision_protocol, LEGACY_PROTOCOL
from .local_risk import risk_schema, LEGACY_RISK_SCHEMA, supervision_risk
from .data import RadarDataset, collate_batch
from .detection import decode_detections
from .annotation_domain import box_intersects_annotation_domain, ignored_region_mask
from .geometry import _corners
from .coordinates import transform_points
from .pipeline import autocast_context, load_models, select_risk, to_device


LEGACY_WARNING = ('这些权重使用旧 manifest 训练：0–50 m 有正例标签，但 50–118 m '
                  '未标注候选曾被当作背景。图中的 50 m 标注域用于解释和重新计分，'
                  '不会改变旧权重。')
FIGURE_WARNING = ('Legacy checkpoints: positives stop at 50 m, while unlabeled 50–118 m '
                  'candidates were trained as background. The displayed domain changes '
                  'interpretation/scoring only; it cannot change old weights.')


def manifest_index(path: Path, split: str = 'val') -> list[dict]:
    """Build a small seek index without retaining the large manifest in memory."""
    if split not in ('train', 'val'):
        raise ValueError('visual diagnosis intentionally supports only train or val')
    path = Path(path).resolve()
    rows = []
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('split') != split:
                continue
            risk = record.get('local_risk') if record.get('risk_supervision_schema') == 'local_agent_collision_v1' else record.get('risk_label')
            risk = risk or {}
            pcol, mask = risk.get('pcol', [0, 0, 0]), risk.get('pcol_mask', [False] * 3)
            collision = any(bool(known) and float(value) > .5 for value, known in zip(pcol, mask))
            rows.append({'offset': offset, 'sample_id': record['sample_id'],
                         'class_schema': record_class_schema(record), 'risk_schema': risk_schema(record), 'supervision_protocol': supervision_protocol(record),
                         'agent_selection_policy': record.get('agent_selection_policy'),
                         'episode_id': record['episode_id'], 'scene_id': record['scene_id'],
                         'agent_count': len(record.get('agents', [])),
                         'collision_positive': collision,
                         'teacher_mode': record.get('teacher_mode'),
                         'instruction': record['language']['instruction']})
    if not rows:
        raise ValueError(f'{path} has no {split} records')
    return rows


def read_record(path: Path, offset: int) -> dict:
    with Path(path).open('rb') as stream:
        stream.seek(offset)
        line = stream.readline()
    if not line:
        raise ValueError(f'no manifest record at byte offset {offset}')
    return json.loads(line)


def _single_sample(record: dict):
    """Use the canonical dataset loader while loading only one manifest row."""
    with tempfile.TemporaryDirectory(prefix='radarvla_visual_') as folder:
        manifest = Path(folder) / 'manifest.jsonl'
        manifest.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        dataset = RadarDataset(manifest, record['split'])
        return dataset[0]


def _cpu_output(output: dict) -> dict:
    return {key: value.detach().float().cpu() if torch.is_tensor(value) else value
            for key, value in output.items()}


def _infer_loaded(grounder, planner, state, checkpoint: Path, sample: dict,
                  device: str, *, generate: bool) -> dict:
    batch = to_device(collate_batch([sample]), device)
    precision = state['protocol'].get('precision', 'float32')
    with torch.no_grad(), autocast_context(device, precision):
        output = grounder(batch)
        plan = None
        if generate:
            if planner is None:
                raise ValueError('SFT checkpoint has no planner')
            risk = select_risk(output['risk'], batch, planner.config.risk_source)
            ids = planner.generate(output['radar_tokens'], risk, batch['instruction'],
                                   max_new_tokens=planner.config.max_length)[0]
            plan = planner.decode(ids, batch['future_times_s'][0].detach().cpu().tolist())
    result = {'output': _cpu_output(output), 'plan': plan,
              'config': grounder.config, 'checkpoint': str(Path(checkpoint).resolve()),
              'epoch': state.get('epoch'), 'stage': state['protocol']['stage'],
              'future_times_s': sample['future_times_s'].detach().cpu()}
    del batch, output
    return result


def _run_checkpoint(checkpoint: Path, sample: dict, device: str, *, generate: bool) -> dict:
    grounder, planner, state = load_models(checkpoint, device)
    result = _infer_loaded(grounder, planner, state, checkpoint, sample, device,
                           generate=generate)
    del grounder, planner, state
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()
    return result


def _decoded(result: dict, threshold: float) -> dict:
    config = replace(result['config'], detection_confidence=float(threshold))
    output = result['output']
    times = result['future_times_s'][None]
    return decode_detections(output, config, times)[0]


def match_detections(record: dict, detections: list[dict], distance_m: float = 2.,
                     *, radar_pose=None) -> dict:
    truth = record.get('agents', [])
    pairs = []
    if truth and detections:
        gt = np.asarray([a['position'] for a in truth], float)
        pred = np.asarray([d['position_m'] for d in detections], float)
        distances = np.linalg.norm(gt[:, None] - pred[None], axis=-1)
        cost = np.where(distances <= distance_m, distances,
                        (len(truth)+len(detections)+1)*(distance_m+1))
        gi, pi = linear_sum_assignment(cost)
        pairs = [(int(g), int(p), float(distances[g, p])) for g, p in zip(gi, pi)
                 if distances[g, p] <= distance_m]
    matched_g, matched_p = {g for g, _, _ in pairs}, {p for _, p, _ in pairs}
    unmatched = [i for i in range(len(detections)) if i not in matched_p]
    ignored = []
    domain = record.get('detection_annotation_domain')
    if domain is not None:
        if radar_pose is None:
            raise ValueError('explicit annotation domain requires radar_pose for matching')
        limits = [domain[key] for key in ('range_m', 'azimuth_min_rad', 'azimuth_max_rad')]
        ignored = [i for i in unmatched if not box_intersects_annotation_domain(
            detections[i]['position_m'], detections[i]['size_m'], detections[i]['heading_rad'],
            limits, radar_pose)]
    outside = list(ignored)
    if detections and record.get('ignored_agents'):
        boxes = torch.tensor([[[*a['position'], *a['size'], a['heading']]
                                for a in record['ignored_agents']]], dtype=torch.float64)
        points = torch.tensor([[d['position_m'] for d in detections]], dtype=torch.float64)
        ignored_removed = ignored_region_mask(points, dict(detection_ignore_boxes=boxes,
            detection_ignore_mask=torch.ones(boxes.shape[:2], dtype=torch.bool),
            detection_ignore_margin_m=torch.tensor([record.get('detection_ignore_margin_m',2.)])))[0]
        ignored = sorted(set(ignored) | {i for i in unmatched if bool(ignored_removed[i])})
    evaluated_unmatched = [i for i in unmatched if i not in set(ignored)]
    return {'pairs': pairs, 'tp': len(pairs), 'fp': len(evaluated_unmatched),
            'fn': len(truth) - len(pairs),
            'unmatched_truth': [i for i in range(len(truth)) if i not in matched_g],
            'unmatched_predictions': evaluated_unmatched,
            'ignored_predictions': ignored, 'ignored_outside_domain': outside}


def xy_to_ra(points) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, float)
    return np.degrees(np.arctan2(points[..., 1], points[..., 0])), np.linalg.norm(points, axis=-1)


def ego_to_radar_xy(points, radar_pose):
    """Project ego-frame ground geometry into the current radar ground plane."""
    return transform_points(points, np.linalg.inv(np.asarray(radar_pose, float)))[..., :2]


def _plot_ra_line(ax, points, radar_pose=None, **kwargs):
    points = np.asarray(points, float)
    valid = points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all(1)
    if not np.any(valid):
        return
    points = points[valid]
    if radar_pose is not None:
        points = ego_to_radar_xy(points, radar_pose)
    azimuth, radius = xy_to_ra(points)
    ax.plot(azimuth, radius, **kwargs)


def _plot_ra_box(ax, center, size, heading, radar_pose=None, **kwargs):
    corners = _corners(center, heading, size)
    _plot_ra_line(ax, np.vstack((corners, corners[0])), radar_pose=radar_pose, **kwargs)


def _plot_bev_box(ax, center, size, heading, **kwargs):
    corners = _corners(center, heading, size)
    ax.add_patch(Polygon(np.c_[corners[:, 1], corners[:, 0]], closed=True,
                         fill=False, **kwargs))


def _roads(record: dict):
    for road in record.get('road_segments', []):
        points = np.asarray(road.get('centerline_ego_m', []), float)
        if points.ndim == 2 and points.shape[1] == 2:
            yield points, float(road.get('width_m', 0.)), int(road.get('ordinal', 0))


def _predicted_roads(plan: dict | None):
    road = (plan or {}).get('road') or {}
    for segment in road.get('segments', []):
        points = segment.get('centerline_points')
        if points is not None:
            yield np.asarray(points, float), float(segment.get('width_m') or 0.)


def _draw_domain_ra(ax, max_range, range_limit=50., angle_limit=53.):
    ax.axhline(range_limit, color='white', linestyle='--', linewidth=1.2, alpha=.9)
    ax.axhspan(range_limit, max_range, color='0.4', alpha=.13)
    ax.text(-angle_limit + 2, range_limit + 3, '50–118 m: no complete Agent labels',
            color='white', fontsize=12, bbox={'facecolor': 'black', 'alpha': .45, 'pad': 3})


def _draw_domain_bev(ax, range_limit=50., angle_limit=53., radar_pose=None):
    angles = np.radians(np.linspace(-angle_limit, angle_limit, 120))
    pose = np.eye(4) if radar_pose is None else radar_pose
    arc = transform_points(np.c_[range_limit*np.cos(angles), range_limit*np.sin(angles)], pose)
    ax.plot(arc[:, 1], arc[:, 0],
            '--', color='0.35', linewidth=1)
    for angle in (-angle_limit, angle_limit):
        angle = math.radians(angle)
        edge = transform_points([[0, 0], [range_limit*math.cos(angle), range_limit*math.sin(angle)]], pose)
        ax.plot(edge[:, 1], edge[:, 0],
                '--', color='0.35', linewidth=1)


def render(record: dict, sample: dict, grounding: dict, sft: dict,
           threshold: float = .9):
    """Render raw RA inputs and GT/Grounding/SFT geometry for one frame."""
    ground_det = _decoded(grounding, threshold)['detections']
    sft_det = _decoded(sft, threshold)['detections']
    radar_pose = sample['radar_pose'][-1].numpy()
    ground_match = match_detections(record, ground_det, radar_pose=radar_pose)
    sft_match = match_detections(record, sft_det, radar_pose=radar_pose)
    radar = sample['radar'].numpy()
    power, doppler = radar[-1, 0], radar[-1, 1]  # Deliberately no doppler_valid mask.
    ranges = sample['range_m'].numpy()
    azimuth = np.degrees(sample['azimuth_rad'].numpy())
    dr = float(np.median(np.diff(ranges)))
    da = float(np.median(np.diff(azimuth)))
    extent = [azimuth[0]-da/2, azimuth[-1]+da/2, ranges[-1]+dr/2, ranges[0]-dr/2]
    power_db = 10 * np.log10(np.maximum(power, np.finfo(np.float32).tiny)
                             / max(float(power.max()), np.finfo(np.float32).tiny))
    finite_doppler = np.abs(doppler[np.isfinite(doppler)])
    vmax = max(1., float(np.percentile(finite_doppler, 99)))

    # Keep the raster panels large enough to inspect individual RA bins in a
    # notebook.  The diagnostics table gets the full lower row so that long
    # instructions and generation errors do not collide with neighbouring axes.
    fig = plt.figure(figsize=(30, 22), dpi=120)
    grid = fig.add_gridspec(2, 3, height_ratios=(2.25, 1),
                            left=.045, right=.975, bottom=.045, top=.925,
                            wspace=.27, hspace=.25)
    power_ax = fig.add_subplot(grid[0, 0])
    doppler_ax = fig.add_subplot(grid[0, 1])
    bev_ax = fig.add_subplot(grid[0, 2])
    table_ax = fig.add_subplot(grid[1, :])
    pimg = power_ax.imshow(power_db, extent=extent, origin='upper', aspect='auto',
                           cmap='viridis', vmin=-60, vmax=0, interpolation='nearest')
    dimg = doppler_ax.imshow(doppler, extent=extent, origin='upper', aspect='auto',
                             cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    fig.colorbar(pimg, ax=power_ax, label='dB relative to this frame peak', pad=.02)
    fig.colorbar(dimg, ax=doppler_ax, label='m/s, saved unfolded values', pad=.02)
    for ax, title in ((power_ax, 'Latest RA Power'),
                      (doppler_ax, 'Latest unfolded Doppler (saved values; no valid mask)')):
        ax.set(title=title, xlabel='azimuth (deg)', ylabel='range (m)')
        ax.tick_params(labelsize=11)
        ax.title.set_fontsize(15)
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        _draw_domain_ra(ax, float(ranges[-1]))

    colors = {'gt': '#35d04f', 'ground': '#ff9f1c', 'sft': '#d84cff', 'road': '#ffffff'}
    for agent in record.get('ignored_agents', []):
        for ax in (power_ax, doppler_ax):
            _plot_ra_box(ax, agent['position'], agent['size'], agent['heading'],
                         radar_pose=radar_pose, color='0.7', linewidth=1.8, linestyle='--')
        _plot_bev_box(bev_ax, agent['position'], agent['size'], agent['heading'],
                      edgecolor='0.5', linewidth=1.8, linestyle='--')
    if record.get('ignored_agents'):
        bev_ax.plot([], [], '--', color='0.5', label='Ignored pedestrian/cyclist (not GT positive)')
    for points, _, _ in _roads(record):
        for ax in (power_ax, doppler_ax):
            _plot_ra_line(ax, points, radar_pose=radar_pose, color=colors['road'], linewidth=2.0, alpha=.95)
        bev_ax.plot(points[:, 1], points[:, 0], color='#238b45', linewidth=3, label='GT road')
    for agent in record.get('agents', []):
        for ax in (power_ax, doppler_ax):
            _plot_ra_box(ax, agent['position'], agent['size'], agent['heading'],
                         radar_pose=radar_pose, color=colors['gt'], linewidth=1.6)
        _plot_bev_box(bev_ax, agent['position'], agent['size'], agent['heading'],
                      edgecolor=colors['gt'], linewidth=2)
        future = np.asarray(agent.get('future_xy', []), float)
        if len(future):
            bev_ax.plot(future[:, 1], future[:, 0], color=colors['gt'], linewidth=1, alpha=.65)
        bev_ax.annotate(f"GT {category_name(agent)}", (agent['position'][1], agent['position'][0]),
                        xytext=(4, 4), textcoords='offset points', color=colors['gt'], fontsize=11)

    for detections, match, color, label in ((ground_det, ground_match, colors['ground'], 'Grounding agent'),
                                             (sft_det, sft_match, colors['sft'], 'SFT-after agent')):
        matched = {p for _, p, _ in match['pairs']}
        ignored = set(match['ignored_predictions'])
        for number, detection in enumerate(detections):
            status = 'TP' if number in matched else ('ignored' if number in ignored else 'FP')
            draw_color = '0.55' if status == 'ignored' else color
            for ax in (power_ax, doppler_ax):
                _plot_ra_box(ax, detection['position_m'], detection['size_m'],
                             detection['heading_rad'], radar_pose=radar_pose,
                             color=draw_color, linewidth=1, linestyle=':')
            _plot_bev_box(bev_ax, detection['position_m'], detection['size_m'],
                          detection['heading_rad'], edgecolor=draw_color, linewidth=1.3, linestyle=':')
            bev_ax.annotate(f"{label} {status} {detection['confidence']:.2f}",
                            (detection['position_m'][1], detection['position_m'][0]),
                            xytext=(4, -13), textcoords='offset points', color=draw_color, fontsize=10)
            if number == 0:
                bev_ax.plot([], [], color=color, linestyle=':', linewidth=2, label=label)
        for gt_index, pred_index, _ in match['pairs']:
            gt, pred = record['agents'][gt_index]['position'], detections[pred_index]['position_m']
            bev_ax.plot([gt[1], pred[1]], [gt[0], pred[0]], color=color, linewidth=.7, alpha=.6)

    plan = sft.get('plan') or {}
    for points, width in _predicted_roads(plan):
        _plot_ra_line(power_ax, points, radar_pose=radar_pose, color='#00d7ff', linewidth=2, linestyle='--')
        _plot_ra_line(doppler_ax, points, radar_pose=radar_pose, color='#00d7ff', linewidth=2, linestyle='--')
        bev_ax.plot(points[:, 1], points[:, 0], '--', color='#00a6c7', linewidth=2.2,
                    label=f'SFT road ({width:.1f} m)')
    ego_trajectory = np.asarray(plan.get('ego_trajectory', []), float)
    if ego_trajectory.ndim == 2 and ego_trajectory.shape[1] == 2 and len(ego_trajectory):
        bev_ax.plot(ego_trajectory[:, 1], ego_trajectory[:, 0], color='#1565c0', linewidth=3,
                    marker='.', label='SFT ego trajectory')
    for agent in plan.get('agent_states', []):
        if agent.get('x') is not None and agent.get('y') is not None:
            bev_ax.scatter(agent['y'], agent['x'], marker='x', s=55, color='#7b1fa2',
                           label='SFT generated agent state')

    _draw_domain_bev(bev_ax, radar_pose=radar_pose)
    bev_ax.scatter([0], [0], marker='+', s=90, color='black', label='ego')
    bev_ax.scatter([radar_pose[1, 3]], [radar_pose[0, 3]], marker='x', s=50,
                   color='0.3', label='radar origin')
    bev_ax.set(xlim=(-60, 60), ylim=(-8, min(120., ranges[-1] + 2)), aspect='equal',
               xlabel='left y (m)', ylabel='forward x (m)', title='BEV: GT / Grounding / SFT')
    bev_ax.tick_params(labelsize=11)
    bev_ax.title.set_fontsize(15)
    bev_ax.xaxis.label.set_size(13)
    bev_ax.yaxis.label.set_size(13)
    handles, labels = bev_ax.get_legend_handles_labels()
    unique = OrderedDict(zip(labels, handles))
    bev_ax.legend(unique.values(), unique.keys(), fontsize=10, loc='upper right',
                  framealpha=.85)

    truth_risk = supervision_risk(record)
    ground_risk = grounding['output']['risk'][0].tolist()
    sft_risk = sft['output']['risk'][0].tolist()
    rows = [
        ['Sample', record['sample_id']],
        ['Episode / scene', f"{record['episode_id']} / {record['scene_id']}"],
        ['Instruction', record['language']['instruction']],
        ['GT agents', str(len(record.get('agents', [])))],
        [f'Grounding @ {threshold:.2f}', f"TP {ground_match['tp']}  FP {ground_match['fp']}  FN {ground_match['fn']}  ignored {len(ground_match['ignored_predictions'])}  raw {len(ground_det)}"],
        [f'SFT-after @ {threshold:.2f}', f"TP {sft_match['tp']}  FP {sft_match['fp']}  FN {sft_match['fn']}  ignored {len(sft_match['ignored_predictions'])}  raw {len(sft_det)}"],
        [f"GT collision p({'/'.join(f'{h:g}' for h in truth_risk.get('pcol_horizons_s',[1.,2.,3.]))}s)", str(truth_risk.get('pcol'))],
        ['Grounding risk', ', '.join(f'{v:.3f}' for v in ground_risk)],
        ['SFT-after risk', ', '.join(f'{v:.3f}' for v in sft_risk)],
        ['SFT free generation', f"mode={plan.get('mode')}  valid={plan.get('valid')}  errors={plan.get('errors', [])}"],
        ['Road semantics', ('SHORT grammar has no road output' if plan.get('mode') == 'short'
                            else f"generated segments={len(list(_predicted_roads(plan)))}")],
        ['Supervision domain', 'Agent/background: 0–50 m, ±53°; radar input extends to about 118 m'],
    ]
    table_ax.axis('off')
    table = table_ax.table(cellText=rows, colWidths=[.20, .78], cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.62)
    for cell in table.get_celld().values():
        cell.PAD = .025
    table_ax.set_title(FIGURE_WARNING, fontsize=13, color='#b22222', pad=10, wrap=True)
    fig.suptitle(f"RadarVLA visual diagnosis | Grounding epoch {grounding.get('epoch')} | "
                 f"SFT epoch {sft.get('epoch')}", fontsize=19, y=.975)
    return fig


class DiagnosisSession:
    """Lazy two-checkpoint inference with a small per-sample cache."""
    def __init__(self, manifest, grounding_checkpoint, sft_checkpoint,
                 device=None, split='val', cache_size=4):
        self.manifest = Path(manifest).resolve()
        self.grounding_checkpoint = Path(grounding_checkpoint).resolve()
        self.sft_checkpoint = Path(sft_checkpoint).resolve()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.index = manifest_index(self.manifest, split)
        self.by_sample = {row['sample_id']: row for row in self.index}
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()
        self._models = None

    def _load_models(self):
        if self._models is None:
            grounding = load_models(self.grounding_checkpoint, self.device)
            sft = load_models(self.sft_checkpoint, self.device)
            expected = {r['class_schema'] for r in self.index}
            if expected != {grounding[0].config.detection_class_schema} or expected != {sft[0].config.detection_class_schema}:
                raise ValueError('Visualization checkpoints and manifest class schemas differ')
            expected_risk = {r['risk_schema'] for r in self.index}
            for _, _, state in (grounding, sft):
                schema = state['protocol']['config'].get('data', {}).get('risk_supervision_schema', LEGACY_RISK_SCHEMA)
                if expected_risk != {schema}:
                    raise ValueError('Visualization checkpoint and manifest Risk schemas differ')
            expected_tasks={r['supervision_protocol'] for r in self.index}
            for _,_,state in (grounding,sft):
                if expected_tasks != {state['protocol']['config'].get('data',{}).get('supervision_protocol',LEGACY_PROTOCOL)}:raise ValueError('Visualization checkpoint task protocol differs')
                if {r.get('agent_selection_policy') for r in self.index} != {state['protocol']['config'].get('data',{}).get('agent_selection_policy')}:
                    raise ValueError('Visualization checkpoint Agent selection policy differs')
            if grounding[1] is not None or sft[1] is None:
                raise ValueError('expected a Grounding-only checkpoint and an SFT checkpoint')
            self._models = grounding, sft
        return self._models

    def diagnose(self, sample_id: str, threshold: float = .9):
        if sample_id not in self.by_sample:
            raise KeyError(sample_id)
        if sample_id not in self.cache:
            index = self.by_sample[sample_id]
            record = read_record(self.manifest, index['offset'])
            sample = _single_sample(record)
            grounding_models, sft_models = self._load_models()
            grounding = _infer_loaded(*grounding_models, self.grounding_checkpoint,
                                      sample, self.device, generate=False)
            sft = _infer_loaded(*sft_models, self.sft_checkpoint,
                                sample, self.device, generate=True)
            self.cache[sample_id] = record, sample, grounding, sft
            self.cache.move_to_end(sample_id)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        record, sample, grounding, sft = self.cache[sample_id]
        return render(record, sample, grounding, sft, threshold)

    def build_widget(self):
        import ipywidgets as widgets
        from IPython.display import display

        episodes = sorted({row['episode_id'] for row in self.index})
        subset = widgets.Dropdown(options=[('全部 validation', 'all'), ('有 GT Agent', 'agents'),
                                           ('无 GT Agent', 'empty'), ('碰撞正例', 'collision'),
                                           ('SFT teacher SHORT', 'short')],
                                  value='all', description='筛选')
        control_style = {'description_width': '100px'}
        episode = widgets.Dropdown(options=episodes, description='Episode', style=control_style,
                                   layout=widgets.Layout(width='100%'))
        sample = widgets.Dropdown(description='Sample', style=control_style,
                                  layout=widgets.Layout(width='100%'))
        threshold = widgets.FloatSlider(value=.9, min=.5, max=.99, step=.01,
                                        description='Threshold', continuous_update=False,
                                        readout_format='.2f', style=control_style,
                                        layout=widgets.Layout(width='75%'))
        run = widgets.Button(description='运行当前帧诊断', button_style='primary',
                             layout=widgets.Layout(width='220px', height='42px'))
        output = widgets.Output(layout=widgets.Layout(width='100%'))

        def selected_rows():
            if subset.value == 'agents':
                return [row for row in self.index if row['agent_count'] > 0]
            if subset.value == 'empty':
                return [row for row in self.index if row['agent_count'] == 0]
            if subset.value == 'collision':
                return [row for row in self.index if row['collision_positive']]
            if subset.value == 'short':
                return [row for row in self.index if row['teacher_mode'] == 'short']
            return self.index

        def update_episodes(*_):
            choices = sorted({row['episode_id'] for row in selected_rows()})
            episode.options = choices

        def update_samples(*_):
            choices = [row for row in selected_rows() if row['episode_id'] == episode.value]
            sample.options = [(f"{row['sample_id']} | agents={row['agent_count']} | "
                               f"collision={row['collision_positive']}", row['sample_id'])
                              for row in choices]

        def draw(_=None):
            with output:
                output.clear_output(wait=True)
                print(f'Loading {sample.value} on {self.device}; first SFT generation can take time...')
                fig = self.diagnose(sample.value, threshold.value)
                output.clear_output(wait=True)
                display(fig)
                plt.close(fig)

        subset.observe(update_episodes, names='value')
        episode.observe(update_samples, names='value')
        run.on_click(draw)
        update_episodes()
        update_samples()
        return widgets.VBox([widgets.HTML(f'<b>{LEGACY_WARNING}</b>'), subset, episode, sample,
                             widgets.HBox([threshold, run]), output])
