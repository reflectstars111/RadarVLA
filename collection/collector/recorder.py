"""Passive, one-frame-at-a-time Leaderboard sensor and actor recorder."""
from __future__ import annotations

import json
import math
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from episode import SCHEMA, append_jsonl, build_anchors, merge_events, write_json
from road_map import (NEAR_ROAD_HORIZON_M, actor_intersects_fov, build_static_map, project_ego_road_corridor,
                      project_navigation,
                      radar_pose, world_to_ego)


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _rotation(value):
    return [float(value.roll), float(value.pitch), float(value.yaw)]


def _pose(transform):
    return {"position_m": _vec(transform.location),
            "rotation_deg": _rotation(transform.rotation)}


def _control(control):
    return {"throttle": float(control.throttle), "brake": float(control.brake),
            "steer": float(control.steer), "hand_brake": bool(control.hand_brake)}


def _actor(actor, snapshot):
    state = snapshot.find(actor.id)
    if state is None:
        return None
    box = actor.bounding_box
    record = {"actor_id": int(actor.id), "type_id": actor.type_id,
              "actor_type": "pedestrian" if actor.type_id.startswith("walker.pedestrian.") else "vehicle",
              "role_name": actor.attributes.get("role_name", ""),
              "pose": _pose(state.get_transform()),
              "velocity_world_mps": _vec(state.get_velocity()),
              "angular_velocity_world_degps": _vec(state.get_angular_velocity()),
              "acceleration_world_mps2": _vec(state.get_acceleration()),
              "bbox": {"center_actor_m": _vec(box.location),
                       "rotation_actor_deg": _rotation(box.rotation),
                       "extent_m": _vec(box.extent)}}
    if record["actor_type"] == "vehicle":
        record["applied_control"] = _control(actor.get_control())
    return record


def _route_points(dense_route, world_map):
    if len(dense_route) < 2:
        raise ValueError("Leaderboard did not supply a complete dense route")
    raw, expert = [], []
    for index, (transform, option) in enumerate(dense_route):
        if not getattr(option, "name", None):
            raise ValueError("RoadOption missing at route point %d" % index)
        location = transform.location
        waypoint = world_map.get_waypoint(location, project_to_road=False)
        if waypoint is None:
            raise ValueError("No road waypoint for official route point %d" % index)
        distance = math.sqrt(sum((a-b)**2 for a, b in zip(
            _vec(location), _vec(waypoint.transform.location))))
        if distance > 0.5:
            raise ValueError("Official route point %d projects %.3f m" % (index, distance))
        raw.append({"position_m": _vec(location), "rotation_deg": _rotation(transform.rotation),
                    "road_option": option.name, "road_id": int(waypoint.road_id),
                    "section_id": int(waypoint.section_id), "lane_id": int(waypoint.lane_id),
                    "s_m": float(waypoint.s)})
        expert.append((waypoint, option))
    return raw, expert


def _route_length(points):
    return sum(math.sqrt(sum((x-y)**2 for x, y in zip(
                   a["position_m"], b["position_m"])))
               for a, b in zip(points, points[1:]))


class Recorder:
    def __init__(self, output, config, dense_route, world, ego):
        self.root = Path(output)
        if self.root.exists():
            raise FileExistsError("episode already exists: %s" % self.root)
        self.root.mkdir(parents=True)
        (self.root / "frames").mkdir()
        self.config = config
        self.ego_id = int(ego.id)
        self.episode_id = uuid.uuid4().hex
        self.count = 0
        self.route_progress = 0
        self.seen = {}
        raw, self.drive_plan = _route_points(dense_route, world.get_map())
        self.route_points = raw
        self.static_map = build_static_map(world.get_map())
        write_json(self.root / "static_map.json", self.static_map)
        self.route_length_m = _route_length(raw)
        write_json(self.root / "route.json", {"schema": SCHEMA,
                   "source": "Leaderboard.set_global_plan full dense world route",
                   "length_m": self.route_length_m, "waypoints": raw})
        write_json(self.root / "metadata.json", {"schema": SCHEMA, "status": "recording",
                   "episode_id": self.episode_id, "actor_id_scope": "episode",
                   "fps": config["fps"], "seed": config["seed"], "ego_actor_id": self.ego_id,
                   "radar_range_m": config["radar_range_m"],
                   "agent_label_range_m": NEAR_ROAD_HORIZON_M,
                   "map_label_range_m": NEAR_ROAD_HORIZON_M,
                   "tick_owner": "Leaderboard ScenarioManager", "captured_frames": 0,
                   "route_xml": config["route_xml"], "scenario_json": config["scenario_json"],
                   "route_id": str(config["route_id"]),
                   "route_sha256": config["route_sha256"],
                   "scenario_sha256": config["scenario_sha256"],
                   "sensors": config["sensors"]})

    def record(self, world, ego, input_data, command):
        camera_frame, camera = input_data["front_rgb"]
        lidar_frame, lidar = input_data["semantic_lidar"]
        snapshot = world.get_snapshot()
        fid = int(snapshot.frame)
        if int(camera_frame) != fid or int(lidar_frame) != fid:
            raise RuntimeError("camera, semantic LiDAR and world snapshot must share one CARLA frame")
        if self.count and fid <= self.last_frame:
            raise RuntimeError("CARLA frame did not advance")
        ego_data = _actor(ego, snapshot)
        if ego_data is None:
            raise RuntimeError("ego absent from world snapshot")
        ego_data["commanded_control"] = _control(command)
        actors = []
        for actor in world.get_actors():
            if actor.id == ego.id or not (actor.type_id.startswith("vehicle.") or
                                         actor.type_id.startswith("walker.pedestrian.")):
                continue
            row = _actor(actor, snapshot)
            if row is not None:
                actors.append(row)
        actors.sort(key=lambda item: item["actor_id"])
        sensor = self.config["sensors"]["semantic_lidar"]
        points = np.asarray(lidar)
        if (points.ndim != 2 or points.shape[1] != 6 or
                not np.isfinite(points).all() or
                np.any(points[:, 4:] != np.floor(points[:, 4:])) or
                np.any(points[:, 4:] < 0) or
                np.any(points[:, 4:] > np.iinfo(np.uint32).max)):
            raise RuntimeError("invalid semantic LiDAR payload")
        self.route_progress, navigation = project_navigation(
            self.route_points, ego_data["pose"], self.route_progress)
        ego_waypoint = world.get_map().get_waypoint(ego.get_location(), project_to_road=False)
        if ego_waypoint is None:
            raise RuntimeError("ego has no current driving waypoint")
        map_data = project_ego_road_corridor(
            self.static_map, self.route_points, ego_data["pose"], sensor,
            self.route_progress,
            {"road_id": int(ego_waypoint.road_id),
             "section_id": int(ego_waypoint.section_id),
             "lane_id": int(ego_waypoint.lane_id), "s_m": float(ego_waypoint.s)})
        radar_position, radar_yaw = radar_pose(ego_data["pose"], sensor)
        visible_agents = []
        for actor in actors:
            local = world_to_ego(actor["pose"]["position_m"],
                                 radar_position, radar_yaw)
            if actor_intersects_fov(actor, radar_position, radar_yaw, NEAR_ROAD_HORIZON_M):
                visible_agents.append(dict(actor, position_radar_m=local))
        folder = self.root / "frames" / ("%06d" % self.count)
        staged = folder.with_name("." + folder.name + ".inprogress")
        staged.mkdir()
        data = np.asarray(camera)
        if data.ndim != 3 or data.shape[2] < 3 or not cv2.imwrite(
                str(staged / "front_rgb.png"), data[:, :, :3]):
            raise RuntimeError("invalid camera payload")
        # The local Leaderboard callback combines float coordinates with uint32
        # actor IDs. Float64 preserves every uint32 ID exactly.
        np.save(str(staged / "semantic_lidar.npy"), points.astype(np.float64), allow_pickle=False)
        write_json(staged / "agents.json", {"carla_frame": fid,
                                            "agents": visible_agents})
        write_json(staged / "map.json", {"carla_frame": fid, "map": map_data})
        row = {"schema": SCHEMA, "index": self.count, "carla_frame": fid,
               "episode_id": self.episode_id,
               "timestamp_s": float(snapshot.timestamp.elapsed_seconds),
               "camera_frame": int(camera_frame), "lidar_frame": int(lidar_frame),
               "sensors": {"camera": "front_rgb.png", "semantic_lidar": "semantic_lidar.npy"},
               "agents": "agents.json", "map": "map.json",
               "route_progress_index": self.route_progress, "navigation": navigation,
               "ego": ego_data, "actors": actors}
        write_json(staged / "frame.json", row)
        os.rename(str(staged), str(folder))
        for actor in [ego_data] + actors:
            actor_id = actor["actor_id"]
            z = actor["pose"]["position_m"][2]
            if actor_id not in self.seen:
                append_jsonl(self.root / "events.jsonl", {"event": "actor_first_observed",
                             "source": "world snapshot", "actor_id": actor_id,
                             "type_id": actor["type_id"], "carla_frame": fid,
                             "timestamp_s": row["timestamp_s"], "position_m": actor["pose"]["position_m"]})
            elif self.seen[actor_id] < -100 <= z:
                append_jsonl(self.root / "events.jsonl", {"event": "actor_surface_observed",
                             "source": "world snapshot", "actor_id": actor_id,
                             "carla_frame": fid, "timestamp_s": row["timestamp_s"],
                             "position_m": actor["pose"]["position_m"]})
            self.seen[actor_id] = z
        self.last_frame = fid
        self.count += 1

    def finalize(self, result=None):
        metadata_path = self.root / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        status = getattr(result, "status", None)
        metadata["route_result"] = str(status) if status is not None else None
        metadata["captured_frames"] = self.count
        try:
            metadata["anchors"] = build_anchors(self.root)
            metadata["events"] = merge_events(self.root)
            metadata["status"] = "complete" if str(status).startswith("Completed") else "failed"
        except Exception as exc:
            metadata["status"] = "partial"
            metadata["finalize_error"] = "%s: %s" % (type(exc).__name__, exc)
            write_json(metadata_path, metadata)
            raise
        write_json(metadata_path, metadata)
