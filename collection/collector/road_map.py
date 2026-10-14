"""Source backed road geometry and per-frame radar-FOV projection.

CARLA waypoints are sampled from OpenDRIVE.  Road sections are never joined by
spatial proximity; the explicit CARLA topology is retained separately.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np


def _xyz(location):
    return [float(location.x), float(location.y), float(location.z)]


def carla_rotation_matrix(rotation_deg):
    """CARLA/UE4 roll,pitch,yaw transform, checked against carla.Transform."""
    roll, pitch, yaw = (math.radians(float(x)) for x in rotation_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr-sy*cr, -cy*sp*cr-sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, -sy*sp*cr+cy*sr],
        [sp, -cp*sr, cp*cr],
    ])


def _lane_key(wp):
    return (int(wp.road_id), int(wp.section_id), int(wp.lane_id))


def build_static_map(world_map, spacing_m=2.0):
    """Sample lane strips, then group same-direction lanes into carriageways."""
    xodr = world_map.to_opendrive()
    lanes = defaultdict(list)
    for wp in world_map.generate_waypoints(spacing_m):
        if str(wp.lane_type).split('.')[-1] != "Driving" or wp.lane_width <= 0:
            continue
        yaw = math.radians(wp.transform.rotation.yaw)
        # CARLA uses x-forward, y-right: left of a waypoint faces -y at yaw=0.
        normal = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
        center = np.array(_xyz(wp.transform.location))
        half = normal * float(wp.lane_width) / 2
        lanes[_lane_key(wp)].append({
            "s_m": float(wp.s), "center_world_m": center.tolist(),
            "left_world_m": (center + half).tolist(),
            "right_world_m": (center - half).tolist(),
            "width_m": float(wp.lane_width), "junction_id": int(wp.junction_id)
            if wp.is_junction else None,
        })
    serialized_lanes = []
    groups = defaultdict(list)
    for key, points in sorted(lanes.items()):
        points.sort(key=lambda p: p["s_m"])
        # A repeated s is a distinct CARLA sample only if it has a distinct
        # position; retaining one prevents zero-length geometric edges.
        unique = []
        for point in points:
            if not unique or abs(point["s_m"] - unique[-1]["s_m"]) > 1e-3:
                unique.append(point)
        if len(unique) < 2:
            continue
        road_id, section_id, lane_id = key
        serialized_lanes.append({"road_id": road_id, "section_id": section_id,
                                 "lane_id": lane_id, "samples": unique})
        groups[(road_id, section_id, 1 if lane_id > 0 else -1)].append(
            (lane_id, unique))
    roads = []
    for (road_id, section_id, direction), members in sorted(groups.items()):
        # Match samples by OpenDRIVE s, never by nearest-road projection.
        s_values = sorted({round(p["s_m"], 3) for _, lane in members for p in lane})
        samples = []
        for s in s_values:
            aligned = []
            for lane_id, lane in members:
                match = min(lane, key=lambda p: abs(p["s_m"] - s))
                if abs(match["s_m"] - s) <= spacing_m * 0.55:
                    aligned.append((lane_id, match))
            if len(aligned) != len(members):
                continue
            # Lane IDs increase toward the left in the lane travel direction.
            # Negative IDs travel with OpenDRIVE s, positive against it.
            ordered = sorted(aligned, key=lambda item: abs(item[0]), reverse=direction > 0)
            left = np.array(ordered[0][1]["left_world_m"])
            right = np.array(ordered[-1][1]["right_world_m"])
            center = (left + right) / 2
            samples.append({"s_m": s, "center_world_m": center.tolist(),
                            "left_world_m": left.tolist(), "right_world_m": right.tolist(),
                            "width_m": float(np.linalg.norm(left - right)),
                            "junction_id": next((p["junction_id"] for _, p in ordered
                                                 if p["junction_id"] is not None), None)})
        if len(samples) >= 2:
            roads.append({"road_id": road_id, "section_id": section_id,
                          "direction": direction,
                          "travel_s_direction": -1 if direction > 0 else 1,
                          "lane_ids": [i for i, _ in members],
                          "samples": samples})
    links = []
    for start, end in world_map.get_topology():
        a, b = _lane_key(start), _lane_key(end)
        if a in lanes and b in lanes:
            links.append({"from": list(a), "to": list(b)})
    return {"source": "CARLA OpenDRIVE waypoints and get_topology",
            "opendrive_sha256": hashlib.sha256(xodr.encode()).hexdigest(),
            "sample_spacing_m": spacing_m, "lanes": serialized_lanes,
            "roads": roads, "topology": links}


def world_to_ego(point, ego_position, ego_yaw_deg):
    delta = np.asarray(point, dtype=float) - np.asarray(ego_position, dtype=float)
    rotation = (ego_yaw_deg if isinstance(ego_yaw_deg, np.ndarray)
                else carla_rotation_matrix([0, 0, ego_yaw_deg]))
    return (rotation.T @ delta).tolist()


def in_fov(point, max_range_m):
    x, y, z = point
    return (x > 0 and math.sqrt(x*x + y*y + z*z) <= max_range_m and
            abs(math.degrees(math.atan2(y, x))) <= 53.0 and
            abs(math.degrees(math.atan2(z, math.hypot(x, y)))) <= 18.0)


def radar_pose(ego_pose, sensor):
    position = np.asarray(ego_pose["position_m"], dtype=float)
    ego_rotation = carla_rotation_matrix(ego_pose["rotation_deg"])
    offset = np.array([sensor["x"], sensor["y"], sensor["z"]], dtype=float)
    position += ego_rotation @ offset
    sensor_rotation = carla_rotation_matrix([
        sensor.get("roll", 0), sensor.get("pitch", 0), sensor.get("yaw", 0)])
    return position.tolist(), ego_rotation @ sensor_rotation


def actor_intersects_fov(actor, position, yaw, max_range_m):
    pose = actor["pose"]
    box = actor["bbox"]
    actor_rotation = carla_rotation_matrix(pose["rotation_deg"])
    box_rotation = actor_rotation @ carla_rotation_matrix(box["rotation_actor_deg"])
    center = np.asarray(pose["position_m"]) + actor_rotation @ np.asarray(box["center_actor_m"])
    ex, ey, ez = box["extent_m"]
    for dx in (-ex, 0, ex):
        for dy in (-ey, 0, ey):
            for dz in (-ez, ez):
                point = center + box_rotation @ np.array([dx, dy, dz])
                if in_fov(world_to_ego(point, position, yaw), max_range_m):
                    return True
    return False


def project_frame(static_map, route_points, ego_pose, max_range_m, sensor,
                  progress_floor=0):
    position, yaw = radar_pose(ego_pose, sensor)
    visible = []
    for road in static_map["roads"]:
        samples = []
        for point in road["samples"]:
            center = world_to_ego(point["center_world_m"], position, yaw)
            left = world_to_ego(point["left_world_m"], position, yaw)
            right = world_to_ego(point["right_world_m"], position, yaw)
            if any(in_fov(p, max_range_m) for p in (center, left, right)):
                samples.append({"s_m": point["s_m"], "center_radar_m": center,
                                "left_radar_m": left, "right_radar_m": right,
                                "width_m": point["width_m"],
                                "junction_id": point["junction_id"]})
        if samples:
            record = {k: road[k] for k in ("road_id", "section_id", "direction",
                                           "travel_s_direction", "lane_ids")}
            record["samples"] = samples
            visible.append(record)
    route_xy = np.array([p["position_m"][:2] for p in route_points])
    ego_xy = np.asarray(ego_pose["position_m"][:2])
    if not 0 <= progress_floor < len(route_points):
        raise ValueError("official route progress floor out of range")
    progress = progress_floor + int(np.argmin(np.linalg.norm(
        route_xy[progress_floor:] - ego_xy, axis=1)))
    target = []
    corridor = []
    roads_by_key = {(r["road_id"], r["section_id"], r["direction"]): r
                    for r in static_map["roads"]}
    for point in route_points[progress:]:
        local = world_to_ego(point["position_m"], position, yaw)
        if in_fov(local, max_range_m):
            target.append({"position_radar_m": local, "road_option": point["road_option"]})
            key = (point.get("road_id"), point.get("section_id"),
                   1 if point.get("lane_id", 0) > 0 else -1)
            road = roads_by_key.get(key)
            width = None
            if road is not None:
                sample = min(road["samples"], key=lambda p: abs(p["s_m"] - point["s_m"]))
                if abs(sample["s_m"] - point["s_m"]) <= static_map["sample_spacing_m"]:
                    width = sample["width_m"]
            corridor.append({"position_radar_m": local, "road_option": point["road_option"],
                             "width_m": width, "width_valid": width is not None})
        if len(target) == 16:
            break
    maneuver = next((p["road_option"] for p in target
                     if p["road_option"] not in ("LANEFOLLOW", "VOID")), "LANEFOLLOW")
    visible_lanes = {(r["road_id"], r["section_id"], lane)
                     for r in visible for lane in r["lane_ids"]}
    topology = [link for link in static_map["topology"]
                if tuple(link["from"]) in visible_lanes and
                tuple(link["to"]) in visible_lanes]
    instructions = {"LEFT": "turn left", "RIGHT": "turn right",
                    "STRAIGHT": "go straight", "CHANGELANELEFT": "change lane left",
                    "CHANGELANERIGHT": "change lane right", "LANEFOLLOW": "follow lane"}
    return {"source_map_sha256": static_map["opendrive_sha256"],
            "coordinate_frame": "radar x-forward y-right z-up",
            "roads": visible, "topology": topology, "planned_route": target,
            "route_progress_index": progress,
            "route_corridor": corridor,
            "navigation_instruction": instructions.get(maneuver, "follow lane"),
            "navigation_road_option": maneuver,
            "instruction_source": "official dense route RoadOption"}


NEAR_ROAD_HORIZON_M = 50.0
NEAR_ROAD_SAMPLES = 16
MIN_ROAD_RETURNS = 3
ROUTE_MATCH_LIMIT_M = 2.0


def _route_key(point):
    return (int(point["road_id"]), int(point["section_id"]), int(point["lane_id"]))


def _project_xy(point, start, end):
    edge = end - start
    length2 = float(edge @ edge)
    if length2 <= 1e-8:
        return None
    fraction = float(np.clip((point - start) @ edge / length2, 0, 1))
    projected = start + fraction * edge
    return float(np.linalg.norm(point - projected)), fraction, projected


def _lane_projection(lanes_by_key, key, point, s_hint, max_s_delta):
    """Project only onto the route-identified lane near its route s."""
    samples = lanes_by_key.get(key)
    if not samples:
        return None
    choices = []
    for a, b in zip(samples, samples[1:]):
        if (abs(a["s_m"] - s_hint) > max_s_delta and
                abs(b["s_m"] - s_hint) > max_s_delta):
            continue
        if b["s_m"] - a["s_m"] > max_s_delta / 2:
            continue
        result = _project_xy(point, np.asarray(a["center_world_m"][:2]),
                             np.asarray(b["center_world_m"][:2]))
        if result is not None:
            distance, fraction, _ = result
            choices.append((distance, float(a["s_m"] + fraction *
                                            (b["s_m"] - a["s_m"]))))
    return min(choices) if choices else None


def _match_record(source=None, distance_to_route=None,
                  distance_to_candidate=None, valid=False):
    return {"source": source, "distance_to_route": distance_to_route,
            "distance_to_candidate": distance_to_candidate, "valid": bool(valid)}


def project_ego_road_corridor(static_map, route_points, ego_pose, sensor,
                              route_progress, ego_waypoint):
    """One route-connected whole-road envelope, sampled 0–50 m ahead.

    The official route selects the road. OpenDRIVE remains the geometry source.
    """
    if len(route_points) < 2 or not 0 <= route_progress < len(route_points):
        raise ValueError("official route progress missing")
    key = tuple(int(ego_waypoint[name]) for name in
                ("road_id", "section_id", "lane_id"))
    if key not in {(r["road_id"], r["section_id"], lane)
                   for r in static_map["roads"] for lane in r["lane_ids"]}:
        raise ValueError("ego waypoint absent from static driving roads")
    route_xyz = np.asarray([row["position_m"] for row in route_points], dtype=float)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route_xyz[:, :2], axis=0), axis=1))]
    ego_xy = np.asarray(ego_pose["position_m"][:2], dtype=float)
    links = {(tuple(link["from"]), tuple(link["to"]))
             for link in static_map["topology"]}
    lanes_by_key = {_route_key(lane): lane["samples"]
                    for lane in static_map.get("lanes", [])}
    spacing = float(static_map["sample_spacing_m"])
    candidates = []
    route_candidates = []
    route_projections = []
    for index in range(max(0, route_progress - 4),
                       min(len(route_points) - 1, route_progress + 5)):
        a, b = route_points[index:index+2]
        projected = _project_xy(ego_xy, route_xyz[index, :2], route_xyz[index+1, :2])
        if projected is None:
            continue
        distance, fraction, point_on_route = projected
        route_projections.append((distance, index, fraction))
        akey, bkey = _route_key(a), _route_key(b)
        if akey == bkey == key:
            candidates.append((distance, index, fraction))
        if distance < ROUTE_MATCH_LIMIT_M and arc[index+1] - arc[index] <= spacing * 2.5:
            # On an ID transition the official route's progress chooses the
            # endpoint. Spatially overlapping off-route branches never enter.
            if akey == bkey:
                chosen_key = akey
                hint = float(a["s_m"] * (1-fraction) + b["s_m"] * fraction)
            elif fraction < .5:
                chosen_key, hint = akey, float(a["s_m"])
            else:
                chosen_key, hint = bkey, float(b["s_m"])
            lane_at_route = _lane_projection(lanes_by_key, chosen_key,
                                             point_on_route, hint, spacing * 5)
            if lane_at_route is None or lane_at_route[0] >= ROUTE_MATCH_LIMIT_M:
                continue
            lane_at_ego = _lane_projection(lanes_by_key, chosen_key, ego_xy,
                                           hint, spacing * 5)
            if lane_at_ego is None:
                continue
            # Route progress precedes distance to keep overlapping junction
            # roads tied to the Leaderboard choice, not a nearest-road query.
            owner_index = index if akey == bkey or fraction < .5 else index + 1
            route_candidates.append((abs(owner_index-route_progress), distance,
                                     -owner_index, index, fraction, chosen_key,
                                     lane_at_ego[0], lane_at_ego[1]))
    exact = min(candidates) if candidates else None
    distance_to_route = min((item[0] for item in route_projections), default=float("inf"))
    if exact is not None and exact[0] < ROUTE_MATCH_LIMIT_M:
        distance, index, fraction = exact
        source_key = key
        start_s = float(ego_waypoint["s_m"])
        candidate_distance = None
        source = "exact_match"
    else:
        if distance_to_route >= ROUTE_MATCH_LIMIT_M:
            raise ValueError("ego is more than 2 m from official route")
        if not route_candidates:
            raise ValueError("ego has no route-constrained road corridor candidate")
        (_, distance, _, index, fraction, source_key,
         candidate_distance, start_s) = min(route_candidates)
        source = ("route_fallback" if source_key == key or
                  (key, source_key) in links else "nearest_route_segment")
    seed_key = source_key
    if candidate_distance is None:
        lane_at_ego = _lane_projection(lanes_by_key, seed_key, ego_xy,
                                       start_s, spacing * 5)
        candidate_distance = None if lane_at_ego is None else lane_at_ego[0]
    # The exact same-ID edge can end just before a junction. Keep its source
    # priority, but anchor the longitudinal samples at Ego's actual projection
    # onto the nearby official route instead of lagging behind that endpoint.
    nearby_route = [item for item in route_projections if abs(item[1] - index) <= 2]
    _, arc_index, arc_fraction = min(nearby_route) if nearby_route else (distance, index, fraction)
    start_arc = arc[arc_index] + arc_fraction * (arc[arc_index+1] - arc[arc_index])
    by_road = defaultdict(list)
    for strip in static_map["roads"]:
        by_road[(strip["road_id"], strip["section_id"])].append(strip)
    radar_position, radar_rotation = radar_pose(ego_pose, sensor)
    ego_position = ego_pose["position_m"]
    ego_rotation = carla_rotation_matrix(ego_pose["rotation_deg"])
    output = {"source_map_sha256": static_map["opendrive_sha256"],
              "target_schema": "ego_road_corridor_50m_v1", "horizon_m": 50.,
              "coordinate_frame": "ego x-forward y-right",
              "geometry_source": "CARLA OpenDRIVE outer Driving-lane boundaries",
              "ego_road": {"road_id": key[0], "section_id": key[1], "lane_id": key[2]},
              "centerline_ego_m": [None] * NEAR_ROAD_SAMPLES,
              "width_m": [None] * NEAR_ROAD_SAMPLES,
              "valid": [False] * NEAR_ROAD_SAMPLES,
              "break_before": [False] * NEAR_ROAD_SAMPLES,
              "source_ids": [None] * NEAR_ROAD_SAMPLES,
              "topology_links": [], "stop_reason": None,
              "matching": {"ego": _match_record(source, float(distance_to_route),
                                                 candidate_distance, True),
                           "points": [_match_record() for _ in range(NEAR_ROAD_SAMPLES)]}}
    previous_key = source_key
    for sample_index, distance_ahead in enumerate(np.linspace(0, 50, NEAR_ROAD_SAMPLES)):
        target_arc = start_arc + distance_ahead
        if target_arc > arc[-1] + 1e-6:
            output["stop_reason"] = "route_end"
            break
        if sample_index == 0:
            source_key, target_s = seed_key, start_s
            route_tangent = route_xyz[index+1, :2] - route_xyz[index, :2]
            route_reference = route_xyz[index, :2] + fraction * route_tangent
            point_source = source
            point_candidate_distance = candidate_distance
            point_distance_to_route = float(distance_to_route)
        else:
            j = int(np.searchsorted(arc, target_arc, side="right") - 1)
            j = min(j, len(route_points) - 2)
            a, b = route_points[j:j+2]
            source_key = (int(a["road_id"]), int(a["section_id"]), int(a["lane_id"]))
            end_key = (int(b["road_id"]), int(b["section_id"]), int(b["lane_id"]))
            if source_key != end_key:
                output["matching"]["points"][sample_index] = _match_record(
                    "nearest_route_segment", 0.0, None)
                continue  # A route interval crossing an ID boundary has no source geometry.
            interval = arc[j+1] - arc[j]
            if interval <= 1e-8:
                continue
            alpha = (target_arc - arc[j]) / interval
            target_s = float(a["s_m"] * (1-alpha) + b["s_m"] * alpha)
            route_tangent = route_xyz[j+1, :2] - route_xyz[j, :2]
            route_reference = route_xyz[j, :2] + alpha * route_tangent
            has_unlinked_transition = False
            for edge_index in range(index, j):
                u, v = route_points[edge_index:edge_index+2]
                ukey = (int(u["road_id"]), int(u["section_id"]), int(u["lane_id"]))
                vkey = (int(v["road_id"]), int(v["section_id"]), int(v["lane_id"]))
                if ukey[:2] != vkey[:2] and (ukey, vkey) not in links:
                    has_unlinked_transition = True
                if ukey[:2] != vkey[:2] and {"from": list(ukey), "to": list(vkey)} not in output["topology_links"]:
                    if (ukey, vkey) in links:
                        output["topology_links"].append({"from": list(ukey), "to": list(vkey)})
            point_source = ("nearest_route_segment" if has_unlinked_transition else
                            source if source_key == seed_key else "route_fallback")
            point_distance_to_route = None
            lane_at_route = _lane_projection(lanes_by_key, source_key,
                                             route_reference, target_s, spacing * 5)
            point_candidate_distance = None if lane_at_route is None else lane_at_route[0]
        output["matching"]["points"][sample_index] = _match_record(
            point_source, point_distance_to_route, point_candidate_distance)
        if source_key[:2] != previous_key[:2]:
            output["break_before"][sample_index] = True
            previous_key = source_key
        strips = by_road.get(source_key[:2], ())
        if not strips or source_key[2] not in {lane for strip in strips for lane in strip["lane_ids"]}:
            continue
        if (point_source != "exact_match" and sample_index > 0 and
                (point_candidate_distance is None or
                 point_candidate_distance >= ROUTE_MATCH_LIMIT_M)):
            continue
        matched = [_road_sample_at_s(strip["samples"], target_s, spacing * 2.5)
                   for strip in strips]
        if any(point is None for point in matched):
            continue
        tangent_norm = np.linalg.norm(route_tangent)
        if tangent_norm <= 1e-8:
            continue
        direction = route_tangent / tangent_norm
        normal = np.array([-direction[1], direction[0]])
        boundaries = [np.asarray(point[name], dtype=float)
                      for point in matched for name in ("left_world_m", "right_world_m")]
        projected = [float(boundary[:2] @ normal) for boundary in boundaries]
        left = boundaries[int(np.argmax(projected))]
        right = boundaries[int(np.argmin(projected))]
        center = (left + right) / 2
        width = float(np.linalg.norm(left[:2] - right[:2]))
        if not np.isfinite(width) or width <= 0:
            continue
        if sample_index > 0:
            output["matching"]["points"][sample_index]["distance_to_route"] = float(
                np.linalg.norm(center[:2] - route_reference))
        local_radar = world_to_ego(center, radar_position, radar_rotation)
        horizontal_range = math.hypot(local_radar[0], local_radar[1])
        if (local_radar[0] <= 0 or horizontal_range > 50 or
                abs(math.degrees(math.atan2(local_radar[1], local_radar[0]))) > 53):
            continue
        local_ego = world_to_ego(center, ego_position, ego_rotation)
        output["centerline_ego_m"][sample_index] = [round(float(local_ego[0]), 3),
                                                       round(float(local_ego[1]), 3)]
        output["width_m"][sample_index] = round(width, 3)
        output["valid"][sample_index] = True
        output["matching"]["points"][sample_index]["valid"] = True
        output["source_ids"][sample_index] = {"road_id": source_key[0],
                                               "section_id": source_key[1],
                                               "lane_id": source_key[2]}
    return output


def _road_sample_at_s(samples, s, max_gap_m):
    """Interpolate only inside a continuous, source-identified carriageway."""
    values = np.asarray([point["s_m"] for point in samples], dtype=float)
    index = int(np.searchsorted(values, s, side="left"))
    if index < len(values) and abs(values[index] - s) < 1e-6:
        return samples[index]
    if index == 0 or index == len(values) or values[index] - values[index-1] > max_gap_m:
        return None
    left, right = samples[index-1], samples[index]
    alpha = (s - values[index-1]) / (values[index] - values[index-1])
    return {key: (np.asarray(left[key]) * (1-alpha) + np.asarray(right[key]) * alpha).tolist()
            if key.endswith("_world_m") else
            float(left[key] * (1-alpha) + right[key] * alpha)
            if key == "width_m" else left[key]
            for key in ("center_world_m", "left_world_m", "right_world_m", "width_m", "junction_id")}


def project_near_roads(static_map, ego_pose, sensor, semantic_lidar):
    """Coarse 16-point labels for every source-identified road inside 50 m.

    Geometry comes from OpenDRIVE. Same-frame semantic-LiDAR road returns
    provide a separate support mask, not a gate on the geometry label.
    """
    position, rotation = radar_pose(ego_pose, sensor)
    lidar = np.asarray(semantic_lidar)
    if lidar.ndim != 2 or lidar.shape[1] != 6 or not np.isfinite(lidar).all():
        raise ValueError("semantic LiDAR must be finite N×6")
    road_returns = lidar[np.isin(lidar[:, 5].astype(np.int64), (6, 7)), :3]
    if len(road_returns):
        ranges = np.linalg.norm(road_returns, axis=1)
        angles = np.rad2deg(np.arctan2(road_returns[:, 1], road_returns[:, 0]))
        elevations = np.rad2deg(np.arctan2(road_returns[:, 2],
                                           np.hypot(road_returns[:, 0], road_returns[:, 1])))
        road_returns = road_returns[(road_returns[:, 0] > 0) &
                                    (ranges <= NEAR_ROAD_HORIZON_M) &
                                    (np.abs(angles) <= 53) & (np.abs(elevations) <= 18)]
    grouped = defaultdict(list)
    for strip in static_map["roads"]:
        grouped[(strip["road_id"], strip["section_id"])].append(strip)
    output = {"source_map_sha256": static_map["opendrive_sha256"],
              "coordinate_frame": "radar x-forward y-right z-up",
              "target_schema": "near_roads_50m_v2", "horizon_m": NEAR_ROAD_HORIZON_M,
              "geometry_source": "CARLA OpenDRIVE driving-road envelope",
              "observability_proxy": "same-frame semantic LiDAR road/roadline first-hit returns",
              "roads": []}
    spacing = float(static_map["sample_spacing_m"])
    for (road_id, section_id), strips in sorted(grouped.items()):
        reference = strips[0]["samples"]
        visible_s = [point["s_m"] for strip in strips for point in strip["samples"] if
                     any(in_fov(world_to_ego(point[key], position, rotation),
                                NEAR_ROAD_HORIZON_M)
                         for key in ("center_world_m", "left_world_m", "right_world_m"))]
        if not visible_s:
            continue
        s_start = max(max(strip["samples"][0]["s_m"] for strip in strips),
                      min(visible_s) - spacing)
        s_end = min(min(strip["samples"][-1]["s_m"] for strip in strips),
                    max(visible_s) + spacing)
        if s_end <= s_start:
            continue
        record = {"road_id": road_id, "section_id": section_id,
                  "centerline_radar_m": [None] * NEAR_ROAD_SAMPLES,
                  "centerline_valid": [False] * NEAR_ROAD_SAMPLES,
                  "lidar_support_mask": [False] * NEAR_ROAD_SAMPLES,
                  "lidar_road_return_count": [0] * NEAR_ROAD_SAMPLES,
                  "width_m": None, "width_valid": False,
                  "width_lidar_supported": False}
        widths, both_side_support = [], 0
        for index, target_s in enumerate(np.linspace(s_start, s_end, NEAR_ROAD_SAMPLES)):
            matched = [_road_sample_at_s(strip["samples"], float(target_s), spacing * 2.5)
                       for strip in strips]
            if any(point is None for point in matched):
                continue
            # Determine the road's local normal from its OpenDRIVE s samples.
            near = int(np.argmin([abs(point["s_m"] - target_s) for point in reference]))
            i0, i1 = max(0, near - 1), min(len(reference) - 1, near + 1)
            direction = (np.asarray(reference[i1]["center_world_m"][:2]) -
                         np.asarray(reference[i0]["center_world_m"][:2]))
            magnitude = np.linalg.norm(direction)
            if magnitude < 1e-3:
                continue
            normal = np.array([-direction[1], direction[0]]) / magnitude
            boundaries = [np.asarray(point[key], dtype=float)
                          for point in matched for key in ("left_world_m", "right_world_m")]
            origin = np.asarray(matched[0]["center_world_m"], dtype=float)
            lateral_values = [float((boundary[:2] - origin[:2]) @ normal)
                              for boundary in boundaries]
            left_world = boundaries[int(np.argmax(lateral_values))]
            right_world = boundaries[int(np.argmin(lateral_values))]
            center = np.asarray(world_to_ego((left_world + right_world) / 2,
                                             position, rotation))
            if not in_fov(center, NEAR_ROAD_HORIZON_M):
                continue
            left = np.asarray(world_to_ego(left_world, position, rotation))
            right = np.asarray(world_to_ego(right_world, position, rotation))
            lateral = left[:2] - right[:2]
            width = float(np.linalg.norm(lateral))
            if width <= 0 or not np.isfinite(width):
                continue
            lateral /= width
            tangent = np.array([lateral[1], -lateral[0]])
            delta = road_returns - center
            along = delta[:, :2] @ tangent
            cross = delta[:, :2] @ lateral
            nearby = ((np.abs(along) <= 2) & (np.abs(cross) <= min(width / 2, 2)) &
                      (np.abs(delta[:, 2]) <= .75))
            count = int(nearby.sum())
            record["lidar_road_return_count"][index] = count
            record["centerline_radar_m"][index] = [round(float(center[0]), 1),
                                                    round(float(center[1]), 1)]
            record["centerline_valid"][index] = True
            record["lidar_support_mask"][index] = count >= MIN_ROAD_RETURNS
            widths.append(width)
            side = (np.abs(along) <= 2) & (np.abs(delta[:, 2]) <= .75)
            if (np.count_nonzero(side & (cross >= width / 4) &
                                 (cross <= width / 2 + .5)) >= MIN_ROAD_RETURNS and
                np.count_nonzero(side & (cross <= -width / 4) &
                                 (cross >= -width / 2 - .5)) >= MIN_ROAD_RETURNS):
                both_side_support += 1
        if len(widths) >= 3:
            record["width_m"] = round(float(np.median(widths)) * 2) / 2
            record["width_valid"] = True
            record["width_lidar_supported"] = both_side_support >= 3
        output["roads"].append(record)
    return output


def project_navigation(route_points, ego_pose, progress_floor):
    """Keep official route instruction separate from observable road geometry."""
    route_xy = np.asarray([point["position_m"][:2] for point in route_points])
    ego_xy = np.asarray(ego_pose["position_m"][:2])
    if not 0 <= progress_floor < len(route_points):
        raise ValueError("official route progress floor out of range")
    progress = progress_floor + int(np.argmin(np.linalg.norm(
        route_xy[progress_floor:] - ego_xy, axis=1)))
    upcoming = route_points[progress:progress + 16]
    maneuver = next((point["road_option"] for point in upcoming
                     if point["road_option"] not in ("LANEFOLLOW", "VOID")), "LANEFOLLOW")
    instructions = {"LEFT": "turn left", "RIGHT": "turn right",
                    "STRAIGHT": "go straight", "CHANGELANELEFT": "change lane left",
                    "CHANGELANERIGHT": "change lane right", "LANEFOLLOW": "follow lane"}
    return progress, {"road_option": maneuver,
                      "instruction": instructions.get(maneuver, "follow lane"),
                      "source": "official dense route RoadOption"}
