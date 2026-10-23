"""CARLA independent episode finalization and validation."""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

SCHEMA = "radarvla_collection_v5"
PREVIOUS_SCHEMA = "radarvla_collection_v4"
OLDER_SCHEMA = "radarvla_collection_v3"
LEGACY_SCHEMA = "radarvla_collection_v2"
FUTURE_STEPS = 30


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def append_jsonl(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError("%s:%d: invalid JSON" % (path, number)) from exc


def frame_paths(root):
    return [path / "frame.json" for path in sorted(
        (item for item in (Path(root) / "frames").iterdir()
         if item.is_dir() and item.name.isdigit()), key=lambda item: int(item.name))]


def _pose(row):
    pose = row["pose"]
    return [*pose["position_m"], pose["rotation_deg"][2]]


def _future(anchor, lookahead_by_frame, actor_id, fps):
    points, valid = [], []
    expected_dt = 1.0 / fps
    for step in range(1, FUTURE_STEPS + 1):
        row = lookahead_by_frame.get(anchor["carla_frame"] + step)
        actor = None if row is None else row["ego"] if actor_id == anchor["ego"]["actor_id"] else next(
            (item for item in row["actors"] if item["actor_id"] == actor_id), None)
        matched = (row is not None and
                   row["carla_frame"] == anchor["carla_frame"] + step and
                   abs(row["timestamp_s"] - anchor["timestamp_s"] - step * expected_dt) <=
                   max(0.02, expected_dt * 0.2) and actor is not None)
        points.append(_pose(actor) if matched else None)
        valid.append(bool(matched))
    return {"world_xyzyaw": points, "valid": valid}


def build_anchors(root):
    """Use only observed future frames; keep a bounded 31-frame buffer."""
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    fps = float(metadata["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid episode fps")
    paths = iter(frame_paths(root))
    window = deque()

    def fill():
        while len(window) <= FUTURE_STEPS:
            try:
                path = next(paths)
            except StopIteration:
                break
            window.append(json.loads(path.read_text()))

    fill()
    count = 0
    output = root / "anchors.jsonl"
    staged = root / ".anchors.jsonl.inprogress"
    with staged.open("w") as stream:
        while window:
            anchor = window[0]
            following = {row["carla_frame"]: row for row in window}
            ids = [anchor["ego"]["actor_id"]] + [a["actor_id"] for a in anchor["actors"]]
            futures = {str(actor_id): _future(anchor, following, actor_id, fps)
                       for actor_id in ids}
            record = {"schema": metadata["schema"], "carla_frame": anchor["carla_frame"],
                      "timestamp_s": anchor["timestamp_s"], "future_steps": FUTURE_STEPS,
                      "future_dt_s": 1.0 / fps, "futures_by_actor_id": futures}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
            window.popleft()
            fill()
    staged.replace(output)
    return count


def merge_events(root):
    """Merge passive observations with ScenarioManager's recorded transition history."""
    root = Path(root)
    output = root / "events.jsonl"
    events = list(read_jsonl(output)) if output.exists() else []
    history = root / "traffic_event_history.jsonl"
    if history.exists():
        events.extend(dict(event, source="ScenarioManager") for event in read_jsonl(history))
    summary = root / "traffic_events.json"
    if summary.exists():
        selected = {item.get("index"): item for item in events
                    if item.get("event") == "scenario_selected"}
        for item in json.loads(summary.read_text()).get("events", []):
            detail = {"scenario_name": item.get("scenario_name"),
                      "scenario_type": item.get("scenario_type"),
                      "trigger_position": item.get("trigger_position"),
                      "other_actor_ids": item.get("other_actor_ids", [])}
            if item.get("index") in selected:
                selected[item.get("index")].update(detail)
            else:
                events.append({"event": "scenario_selected", "source": "Leaderboard",
                               "index": item.get("index"), **detail})
    events.sort(key=lambda item: (item.get("carla_frame") is not None,
                                  item.get("carla_frame") or -1))
    staged = root / ".events.jsonl.inprogress"
    with staged.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
    staged.replace(output)
    return len(events)


def validate(root, require_complete=False):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    route = json.loads((root / "route.json").read_text())
    schema = metadata.get("schema")
    if schema not in (SCHEMA, PREVIOUS_SCHEMA, OLDER_SCHEMA, LEGACY_SCHEMA) or route.get("schema") != schema:
        raise ValueError("raw schema mismatch")
    if schema in (SCHEMA, PREVIOUS_SCHEMA, OLDER_SCHEMA) and (not metadata.get("episode_id") or
                                               metadata.get("actor_id_scope") != "episode"):
        raise ValueError("episode-scoped actor ID missing")
    if schema in (SCHEMA, PREVIOUS_SCHEMA) and (metadata.get("agent_label_range_m") != 50 or
                            metadata.get("map_label_range_m") != 50):
        raise ValueError("near-field label range missing")
    if metadata.get("sensors") and (metadata["sensors"]["semantic_lidar"]["range_m"]
                                    < metadata["radar_range_m"]):
        raise ValueError("LiDAR does not cover radar range axis")
    if len(route.get("waypoints", [])) < 2 or any(
            not item.get("road_option") for item in route["waypoints"]):
        raise ValueError("dense route or RoadOption missing")
    static_map = json.loads((root / "static_map.json").read_text())
    if not static_map.get("opendrive_sha256") or not static_map.get("roads"):
        raise ValueError("OpenDRIVE road geometry missing")
    frames = frame_paths(root)
    if not frames:
        raise ValueError("no frames")
    previous_frame = None
    previous_time = None
    types = {}
    rows_by_frame = {}
    visible_by_frame = {}
    map_visible_road_frames = 0
    for path in frames:
        row = json.loads(path.read_text())
        fid, timestamp = row["carla_frame"], row["timestamp_s"]
        if previous_frame is not None and (fid <= previous_frame or timestamp <= previous_time):
            raise ValueError("nonmonotonic CARLA frame or timestamp")
        previous_frame, previous_time = fid, timestamp
        rows_by_frame[fid] = row
        if row["camera_frame"] != fid or row["lidar_frame"] != fid:
            raise ValueError("sensor frame mismatch at %s" % path)
        for key in ("camera", "semantic_lidar"):
            if not (path.parent / row["sensors"][key]).is_file():
                raise ValueError("missing %s at %s" % (key, path))
        agents_path, map_path = path.parent / row["agents"], path.parent / row["map"]
        if not agents_path.is_file() or not map_path.is_file():
            raise ValueError("agent or map file missing at %s" % path)
        agents = json.loads(agents_path.read_text())
        frame_map = json.loads(map_path.read_text())
        if agents["carla_frame"] != fid or frame_map["carla_frame"] != fid:
            raise ValueError("agent/map frame mismatch at %s" % path)
        current_ids = {a["actor_id"] for a in row["actors"]}
        if len({a["actor_id"] for a in agents["agents"]}) != len(agents["agents"]):
            raise ValueError("duplicate visible actor ID")
        if any(a["actor_id"] not in current_ids for a in agents["agents"]):
            raise ValueError("visible agent absent from full actor snapshot")
        by_id = {a["actor_id"]: a for a in row["actors"]}
        if any(a["type_id"] != by_id[a["actor_id"]]["type_id"] for a in agents["agents"]):
            raise ValueError("visible agent actor ID/type mismatch")
        if schema in (SCHEMA, PREVIOUS_SCHEMA) and metadata.get("sensors"):
            from road_map import actor_intersects_fov, radar_pose
            position, rotation = radar_pose(row["ego"]["pose"],
                                            metadata["sensors"]["semantic_lidar"])
            expected_agent_ids = {actor["actor_id"] for actor in row["actors"]
                                  if actor_intersects_fov(actor, position, rotation, 50)}
            if metadata.get('agent_selection'):
                from agent_policy import exclusion_reason, motion_metadata
                if metadata['agent_selection'] != motion_metadata():
                    raise ValueError('Unknown raw Agent motion policy')
                observed = agents.get('observed_agents', [])
                if len(observed) != len(expected_agent_ids) or {a['actor_id'] for a in observed} != expected_agent_ids:
                    raise ValueError('Full observed Agent archive incomplete')
                expected_agent_ids = {identifier for identifier in expected_agent_ids
                                      if exclusion_reason(by_id[identifier]) is None}
            actual_agent_ids = {actor["actor_id"] for actor in agents["agents"]}
            if actual_agent_ids != expected_agent_ids:
                raise ValueError("50 m visible agent set differs from world snapshot")
        visible_by_frame[fid] = [str(a["actor_id"]) for a in agents["agents"]]
        map_data = frame_map["map"]
        if map_data["source_map_sha256"] != static_map["opendrive_sha256"]:
            raise ValueError("map source hash mismatch")
        if schema == SCHEMA:
            if row.get("episode_id") != metadata["episode_id"]:
                raise ValueError("frame episode ID mismatch")
            if "roads" in map_data:
                raise ValueError("v5 map must contain one ego corridor, not a road list")
            if (map_data.get("target_schema") != "ego_road_corridor_50m_v1" or
                    map_data.get("horizon_m") != 50 or
                    map_data.get("coordinate_frame") != "ego x-forward y-right" or
                    map_data.get("geometry_source") !=
                    "CARLA OpenDRIVE outer Driving-lane boundaries"):
                raise ValueError("ego road corridor schema or provenance mismatch")
            origin = map_data.get("ego_road")
            if not isinstance(origin, dict) or any(
                    not isinstance(origin.get(name), int)
                    for name in ("road_id", "section_id", "lane_id")):
                raise ValueError("ego road source ID missing")
            arrays = [map_data.get(name) for name in
                      ("centerline_ego_m", "width_m", "valid", "source_ids", "break_before")]
            if any(not isinstance(a, list) or len(a) != 16 for a in arrays):
                raise ValueError("ego road corridor must have 16 samples")
            centers, widths, masks, sources, breaks = arrays
            if any(type(value) is not bool for value in breaks):
                raise ValueError("ego corridor break mask malformed")
            static_ids = {(r["road_id"], r["section_id"], lane)
                          for r in static_map["roads"] for lane in r.get("lane_ids", [])}
            if static_ids and (origin["road_id"], origin["section_id"], origin["lane_id"]) not in static_ids:
                raise ValueError("ego road ID absent from static map")
            for center, width, valid, source in zip(centers, widths, masks, sources):
                if type(valid) is not bool or any((item is None) == valid
                                                  for item in (center, width, source)):
                    raise ValueError("ego corridor value/mask mismatch")
                if not valid:
                    continue
                if (not isinstance(center, list) or len(center) != 2 or
                        not all(isinstance(x, (int, float)) and math.isfinite(x) for x in center)
                        or not isinstance(width, (int, float)) or
                        not math.isfinite(width) or width <= 0 or
                        not isinstance(source, dict)):
                    raise ValueError("invalid ego corridor geometry")
                source_id = tuple(source.get(name) for name in ("road_id", "section_id", "lane_id"))
                if (not all(isinstance(v, int) for v in source_id) or
                        (static_ids and source_id not in static_ids)):
                    raise ValueError("ego corridor source ID absent from static map")
                if metadata.get("sensors"):
                    from road_map import carla_rotation_matrix, radar_pose, world_to_ego
                    import numpy as np
                    ego_pose = row["ego"]["pose"]
                    sensor = metadata["sensors"]["semantic_lidar"]
                    sensor_position, sensor_rotation = radar_pose(ego_pose, sensor)
                    ego_rotation = carla_rotation_matrix(ego_pose["rotation_deg"])
                    world_point = (np.asarray(ego_pose["position_m"]) +
                                   ego_rotation @ np.asarray([*center, 0.]))
                    x, y, _ = world_to_ego(world_point, sensor_position, sensor_rotation)
                    if x <= 0 or math.hypot(x, y) > 50.1 or abs(math.degrees(math.atan2(y, x))) > 53.2:
                        raise ValueError("ego corridor point outside horizontal radar FOV")
            static_links = {(tuple(link["from"]), tuple(link["to"]))
                            for link in static_map.get("topology", [])}
            if not isinstance(map_data.get("topology_links"), list) or any(
                    (tuple(link["from"]), tuple(link["to"])) not in static_links
                    for link in map_data["topology_links"]):
                raise ValueError("ego corridor topology link absent from static map")
            if map_data.get("stop_reason") not in (None, "route_end", "unlinked_route_successor"):
                raise ValueError("ego corridor stop reason malformed")
            map_visible_road_frames += any(masks)
        elif schema in (PREVIOUS_SCHEMA, OLDER_SCHEMA):
            if row.get("episode_id") != metadata["episode_id"]:
                raise ValueError("frame episode ID mismatch")
            expected_map_schema = "near_roads_50m_v2" if schema == PREVIOUS_SCHEMA else "near_roads_50m_v1"
            if map_data.get("target_schema") != expected_map_schema or map_data.get("horizon_m") != 50:
                raise ValueError("near road target schema or horizon mismatch")
            if schema == PREVIOUS_SCHEMA and map_data.get("geometry_source") != "CARLA OpenDRIVE driving-road envelope":
                raise ValueError("near road geometry provenance missing")
            if not isinstance(map_data.get("roads"), list):
                raise ValueError("near roads missing")
            seen_road_keys = set()
            for road in map_data["roads"]:
                road_key = (road.get("road_id"), road.get("section_id"))
                if road_key in seen_road_keys or not any(
                        (source["road_id"], source["section_id"]) == road_key
                        for source in static_map["roads"]):
                    raise ValueError("near road source ID missing or duplicated")
                seen_road_keys.add(road_key)
                centers = road.get("centerline_radar_m", [])
                mask = road.get("centerline_valid", [])
                counts = road.get("lidar_road_return_count", [])
                lidar_mask = road.get("lidar_support_mask", []) if schema == PREVIOUS_SCHEMA else []
                arrays = (centers, mask, counts, lidar_mask) if schema == PREVIOUS_SCHEMA else (centers, mask, counts)
                if not all(len(a) == 16 for a in arrays):
                    raise ValueError("near road centerline grid mismatch")
                if any(type(valid) is not bool or (point is None) == valid or
                       (valid and (len(point) != 2 or not all(math.isfinite(v) for v in point) or
                                   math.hypot(*point) > 50.1 or
                                   point[0] < 0 or
                                   abs(math.degrees(math.atan2(point[1], point[0]))) > 53.2 or
                                   (schema == OLDER_SCHEMA and count < 3))) or
                       not isinstance(count, int) or count < 0
                       for point, valid, count in zip(centers, mask, counts)):
                    raise ValueError("near road centerline/mask/support mismatch")
                if schema == PREVIOUS_SCHEMA and any(
                        type(supported) is not bool or supported != (valid and count >= 3)
                        for supported, valid, count in zip(lidar_mask, mask, counts)):
                    raise ValueError("near road LiDAR support/mask mismatch")
                if (type(road.get("width_valid")) is not bool or
                        (road["width_m"] is None) == road["width_valid"] or
                        (road["width_valid"] and
                         (not math.isfinite(road["width_m"]) or road["width_m"] <= 0))):
                    raise ValueError("near road width/mask mismatch")
                if schema == PREVIOUS_SCHEMA and (type(road.get("width_lidar_supported")) is not bool or
                                         (road["width_lidar_supported"] and not road["width_valid"])):
                    raise ValueError("near road width LiDAR support mismatch")
            map_visible_road_frames += any(any(road["centerline_valid"])
                                           for road in map_data["roads"])
        if schema != LEGACY_SCHEMA:
            progress = row.get("route_progress_index")
            if not isinstance(progress, int) or not 0 <= progress < len(route["waypoints"]):
                raise ValueError("route progress out of range")
            upcoming = route["waypoints"][progress:progress + 16]
            maneuver = next((p["road_option"] for p in upcoming
                             if p["road_option"] not in ("LANEFOLLOW", "VOID")), "LANEFOLLOW")
            navigation = row.get("navigation", {})
            if navigation.get("road_option") != maneuver or navigation.get("source") != "official dense route RoadOption":
                raise ValueError("navigation differs from official RoadOption")
        else:
            map_visible_road_frames += bool(map_data["roads"])
            if not (0 <= map_data["route_progress_index"] < len(route["waypoints"])):
                raise ValueError("route progress out of range")
            if len(map_data["route_corridor"]) != len(map_data["planned_route"]):
                raise ValueError("route corridor/plan mismatch")
        if schema == LEGACY_SCHEMA and metadata.get("sensors"):
            from road_map import in_fov, radar_pose, world_to_ego
            radar_position, radar_yaw = radar_pose(
                row["ego"]["pose"], metadata["sensors"]["semantic_lidar"])
            expected = []
            for route_point in route["waypoints"][map_data["route_progress_index"]:]:
                local = world_to_ego(route_point["position_m"], radar_position, radar_yaw)
                if in_fov(local, metadata["radar_range_m"]):
                    expected.append((route_point["road_option"], local))
                if len(expected) == 16:
                    break
            actual = map_data["planned_route"]
            if len(actual) != len(expected) or any(
                    point["road_option"] != option or
                    math.sqrt(sum((a-b)**2 for a, b in zip(
                        point["position_radar_m"], local))) > 1e-3
                    for point, (option, local) in zip(actual, expected)):
                raise ValueError("map route differs from official dense route")
            maneuver = next((p["road_option"] for p in actual
                             if p["road_option"] not in ("LANEFOLLOW", "VOID")),
                            "LANEFOLLOW")
            if map_data.get("navigation_road_option") != maneuver:
                raise ValueError("navigation instruction differs from RoadOption")
        if schema == LEGACY_SCHEMA:
            for road in map_data["roads"]:
                for sample in road["samples"]:
                    if not math.isfinite(sample["width_m"]) or sample["width_m"] <= 0:
                        raise ValueError("invalid road width")
        ids = set()
        for actor in [row["ego"]] + row["actors"]:
            actor_id = actor["actor_id"]
            if actor_id in ids or (actor_id in types and types[actor_id] != actor["type_id"]):
                raise ValueError("duplicate or unstable actor ID at %s" % path)
            ids.add(actor_id)
            types[actor_id] = actor["type_id"]
    anchors = root / "anchors.jsonl"
    if not anchors.is_file():
        raise ValueError("anchors.jsonl missing")
    count = 0
    full_ego_future = 0
    full_visible_agent_future = 0
    visible_agent_anchors = 0
    for item in read_jsonl(anchors):
        count += 1
        anchor_frame = json.loads(frames[count - 1].read_text()) if count <= len(frames) else None
        if anchor_frame is None or item["carla_frame"] != anchor_frame["carla_frame"]:
            raise ValueError("anchor CARLA frame mismatch")
        if item.get("schema") != schema:
            raise ValueError("anchor schema mismatch")
        expected_ids = {str(actor["actor_id"]) for actor in
                        [anchor_frame["ego"]] + anchor_frame["actors"]}
        if set(item["futures_by_actor_id"]) != expected_ids:
            raise ValueError("anchor actor IDs differ from current frame")
        if item["future_steps"] != FUTURE_STEPS:
            raise ValueError("future horizon mismatch")
        ego_id = str(anchor_frame["ego"]["actor_id"])
        full_ego_future += all(item["futures_by_actor_id"][ego_id]["valid"])
        visible_ids = visible_by_frame[item["carla_frame"]]
        visible_agent_anchors += bool(visible_ids)
        full_visible_agent_future += any(
            all(item["futures_by_actor_id"][actor_id]["valid"])
            for actor_id in visible_ids)
        for future in item["futures_by_actor_id"].values():
            if len(future["valid"]) != FUTURE_STEPS or len(future["world_xyzyaw"]) != FUTURE_STEPS:
                raise ValueError("future length mismatch")
            if any(type(is_valid) is not bool or (point is None) == is_valid or
                   (point is not None and (not isinstance(point, list) or len(point) != 4))
                   for point, is_valid in zip(future["world_xyzyaw"], future["valid"])):
                raise ValueError("future point/mask mismatch")
        for actor_id, future in item["futures_by_actor_id"].items():
            if future != _future(anchor_frame, rows_by_frame, int(actor_id), float(metadata["fps"])):
                raise ValueError("future actor ID/trajectory reference mismatch")
    if count != len(frames):
        raise ValueError("anchor/frame count mismatch")
    if require_complete and metadata.get("status") != "complete":
        raise ValueError("episode is not complete")
    return {"schema": schema, "frames": len(frames), "anchors": count,
            "actors_seen": len(types), "status": metadata.get("status"),
            "map_visible_road_frames": map_visible_road_frames,
            "visible_agent_anchors": visible_agent_anchors,
            "full_ego_future_30": full_ego_future,
            "full_visible_agent_future_30": full_visible_agent_future}


def main():
    parser = argparse.ArgumentParser(description="Validate a raw collection episode")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.episode, args.require_complete), indent=2))


if __name__ == "__main__":
    main()
