"""Turn the collected ego corridor into ordered, source backed road segments."""
from __future__ import annotations

import math
from statistics import median


def road_segments(map_record: dict, *, max_width_span_m: float = 0.5) -> list[dict]:
    source = map_record.get('map', map_record)
    points = source['centerline_ego_m']
    widths = source['width_m']
    valid = source['valid']
    breaks = source['break_before']
    ids = source['source_ids']
    if not (len(points) == len(widths) == len(valid) == len(breaks) == len(ids) == 16):
        raise ValueError('corridor arrays must have 16 aligned entries')
    segments: list[dict] = []
    current: list[tuple[int, list[float], float | None]] = []
    current_key = None

    def flush():
        if len(current) < 2:
            return
        usable = [w for _, _, w in current if w is not None and math.isfinite(w) and w > 0]
        span = max(usable) - min(usable) if usable else math.inf
        width_valid = len(usable) == len(current) and span <= max_width_span_m
        segments.append({
            'ordinal': len(segments),
            'centerline_ego_m': [point for _, point, _ in current],
            'source_point_indices': [i for i, _, _ in current],
            'width_m': float(median(usable)) if width_valid else None,
            'width_valid': width_valid,
            'road_id': current_key[0], 'section_id': current_key[1],
        })

    for i in range(16):
        identity = ids[i]
        key = None if identity is None else (identity.get('road_id'), identity.get('section_id'))
        p = points[i]
        usable = bool(valid[i]) and p is not None and key is not None and len(p) >= 2 and all(math.isfinite(v) for v in p[:2])
        if not usable:
            flush(); current = []; current_key = None
            continue
        if current and (breaks[i] or key != current_key):
            flush(); current = []
        current_key = key
        current.append((i, [float(p[0]), float(p[1])], widths[i]))
    flush()
    return segments
