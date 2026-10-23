"""Motion membership shared by the Python 3.7 recorder and training adapter."""
import math

POLICY = 'moving_vehicle_01_background_v1'
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


def motion_metadata():
    return dict(policy=POLICY, min_speed_mps=MIN_SPEED_MPS,
                speed_frame='world_horizontal_current_frame', threshold_comparison='>=',
                range_m=50., half_fov_deg=53., background=['stationary', 'pedestrian', 'cyclist'],
                detection_agent_limit=None, risk_agent_limit=None)
