"""Leaderboard expert: official dense route, BasicAgent, passive raw capture."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

from agents.navigation.basic_agent import BasicAgent
from leaderboard.autoagents.autonomous_agent_local import AutonomousAgent, Track
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from episode import write_json
from recorder import Recorder


def get_entry_point():
    return "RawCollectionAgent"


class RawCollectionAgent(AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None):
        self.track = Track.SENSORS
        self.config = json.loads(Path(path_to_conf_file).read_text())
        self.output = Path(os.environ["RADARVLA_EPISODE"]).resolve()
        self.dense_route = None
        self.expert = None
        self.recorder = None

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        # Base class keeps downsampled hints. Preserve the exact source first.
        self.dense_route = list(global_plan_world_coord)
        super().set_global_plan(global_plan_gps, global_plan_world_coord)

    def sensors(self):
        camera = self.config["sensors"]["camera"]
        lidar = self.config["sensors"]["semantic_lidar"]
        return [
            {"type": "sensor.camera.rgb", "id": "front_rgb",
             "x": camera["x"], "y": camera["y"], "z": camera["z"],
             "roll": camera["roll"], "pitch": camera["pitch"], "yaw": camera["yaw"],
             "width": camera["width"], "height": camera["height"], "fov": camera["fov"]},
            {"type": "sensor.lidar.ray_cast_semantic", "id": "semantic_lidar",
             "x": lidar["x"], "y": lidar["y"], "z": lidar["z"],
             "roll": lidar["roll"], "pitch": lidar["pitch"], "yaw": lidar["yaw"],
             "channels": lidar["channels"], "points_per_second": lidar["points_per_second"],
             "rotation_frequency": self.config["fps"], "range": lidar["range_m"],
             "upper_fov": lidar["upper_fov"], "lower_fov": lidar["lower_fov"]}
        ]

    def _initialize(self):
        world = CarlaDataProvider.get_world()
        ego = next((actor for actor in world.get_actors().filter("vehicle.*")
                    if actor.attributes.get("role_name") == "hero"), None)
        if ego is None:
            raise RuntimeError("Leaderboard hero actor missing")
        if self.dense_route is None:
            raise RuntimeError("Leaderboard full dense route missing")
        self.recorder = Recorder(self.output, self.config, self.dense_route, world, ego)
        self.expert = BasicAgent(ego, target_speed=self.config["target_speed_kmh"])
        self.expert._proximity_vehicle_threshold = self.config["vehicle_hazard_distance_m"]
        planner = self.expert._local_planner
        dt = 1.0 / self.config["fps"]
        planner._dt = dt
        planner._vehicle_controller._lon_controller._dt = dt
        planner._vehicle_controller._lat_controller._dt = dt
        # A short controller-only tail allows CARLA 0.9.10 RouteCompletionTest
        # to see the ego cross the final official waypoint. route.json stays exact.
        official = self.recorder.drive_plan
        tail_candidates = official[-1][0].next(6.0)
        if not tail_candidates:
            raise RuntimeError("Cannot extend controller past final official waypoint")
        yaw = official[-1][0].transform.rotation.yaw
        tail = min(tail_candidates, key=lambda item:
                   abs((item.transform.rotation.yaw - yaw + 180) % 360 - 180))
        planner.set_global_plan(official + [(tail, official[-1][1])])

    def run_step(self, input_data, timestamp, sensors=None):
        try:
            if self.expert is None:
                self._initialize()
            command = self.expert.run_step()  # Includes BasicAgent vehicle hazard detection.
            self.recorder.record(CarlaDataProvider.get_world(), self.expert._vehicle,
                                 input_data, command)
            return command
        except Exception as exc:
            if self.output.is_dir():
                write_json(self.output / "agent_error.json", {
                    "type": type(exc).__name__, "message": str(exc),
                    "traceback": traceback.format_exc()})
            raise

    def destroy(self, results=None):
        if self.recorder is not None:
            self.recorder.finalize(results)
