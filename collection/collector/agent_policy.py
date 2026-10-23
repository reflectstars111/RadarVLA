"""Motion membership shared by the Python 3.7 recorder and training adapter."""
import math

LEGACY_POLICY = 'moving_vehicle_01_background_v1'
POLICY = 'moving_vehicle_01_fullbbox50_background_v2'
SUPPORTED_POLICIES = (LEGACY_POLICY, POLICY)
MIN_SPEED_MPS = 0.1
BICYCLES = frozenset(('vehicle.bh.crossbike', 'vehicle.diamondback.century',
                      'vehicle.gazelle.omafiets'))


def exclusion_reason(actor):
    kind = actor['type_id']
    if kind.startswith('walker.pedestrian.'):
        return 'pedestrian'
    if kind in BICYCLES:
        return 'cyclist'
    if not kind.startswith('vehicle.'):
        return 'non_vehicle'
    velocity = actor['velocity_world_mps']
    if len(velocity) != 3 or not all(math.isfinite(float(v)) for v in velocity):
        raise ValueError('Missing/nonfinite world actor velocity')
    return 'stationary' if math.hypot(*velocity[:2]) < MIN_SPEED_MPS else None


def motion_metadata(policy=POLICY):
    if policy not in SUPPORTED_POLICIES:
        raise ValueError("Unknown moving policy")
    value = dict(policy=policy, min_speed_mps=MIN_SPEED_MPS,
                speed_frame='world_horizontal_current_frame', threshold_comparison='>=',
                range_m=50., half_fov_deg=53., background=['stationary', 'pedestrian', 'cyclist'],
                detection_agent_limit=None, risk_agent_limit=None)

    if policy == POLICY:
        value['distance_boundary'] = 'all_world_xy_bbox_vertices_strictly_inside_radar_circle'
        value['boundary_m'] = 50.
        value['future_range_clipping'] = False
    return value


def full_bbox_inside(actor, radar_position, range_m=50.):
    """All eight world BBox vertices projected to XY; circle centred on radar."""
    try:
        from road_map import carla_rotation_matrix
    except ImportError:
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('_motion_geometry', Path(__file__).with_name('road_map.py'))
        geometry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(geometry)
        carla_rotation_matrix = geometry.carla_rotation_matrix
    import numpy as np
    pose, box = actor['pose'], actor['bbox']
    rotation = carla_rotation_matrix(pose['rotation_deg'])
    box_rotation = rotation @ carla_rotation_matrix(box['rotation_actor_deg'])
    center = np.asarray(pose['position_m'], dtype=float) + rotation @ np.asarray(box['center_actor_m'], dtype=float)
    ex, ey, ez = box['extent_m']
    if not np.isfinite(center).all() or not np.isfinite(box_rotation).all() or any(not math.isfinite(float(v)) or v < 0 for v in (ex, ey, ez)):
        raise ValueError('Invalid BBox geometry')
    for dx in (-ex, ex):
        for dy in (-ey, ey):
            for dz in (-ez, ez):
                point = center + box_rotation @ np.array([dx, dy, dz])
                if math.hypot(point[0]-radar_position[0], point[1]-radar_position[1]) >= range_m - 1e-9:
                    return False
    return True


def label_exclusion_reason(actor, radar_position, policy=POLICY):
    if policy not in SUPPORTED_POLICIES:
        raise ValueError('Unknown moving policy')
    reason = exclusion_reason(actor)
    if reason is None and policy == POLICY and not full_bbox_inside(actor, radar_position):
        return 'range_boundary'
    return reason
