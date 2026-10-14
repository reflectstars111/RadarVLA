"""Offline checks for road structure, conservative velocity and RA unfolding."""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as N

import numpy as np

from radar import select_scatterers
from radar_source import prepare_scatterers
from road_map import (actor_intersects_fov, build_static_map, carla_rotation_matrix,
                      project_ego_road_corridor, project_frame, project_near_roads)
from unfold import unfold_frame


def wp(s, lane, y):
    return N(road_id=1, section_id=0, lane_id=lane, s=s, lane_type="Driving",
             lane_width=3.5, junction_id=None, is_junction=False,
             transform=N(location=N(x=float(s), y=y, z=0.0),
                         rotation=N(yaw=0.0)))


class FakeMap:
    def __init__(self):
        self.points = [wp(s, lane, y) for lane, y in ((-1, 1.75), (-2, 5.25))
                       for s in (0, 2, 4)]

    def to_opendrive(self):
        return "<OpenDRIVE/>"

    def generate_waypoints(self, spacing):
        return self.points

    def get_topology(self):
        return [(self.points[0], self.points[1])]


class VLAProductsTests(unittest.TestCase):
    def test_ego_whole_road_corridor_uses_linked_successor_and_point_widths(self):
        def strip(road_id, start, stop, center_y, width, lane_id):
            return {"road_id": road_id, "section_id": 0,
                    "direction": 1 if lane_id > 0 else -1,
                    "lane_ids": [lane_id],
                    "samples": [{"s_m": float(x),
                                 "center_world_m": [float(x), center_y, 0.],
                                 "left_world_m": [float(x), center_y-width/2, 0.],
                                 "right_world_m": [float(x), center_y+width/2, 0.],
                                 "width_m": width, "junction_id": None}
                                for x in range(start, stop+1, 2)]}
        def lane(road_id, start, stop, center_y, lane_id):
            return {"road_id": road_id, "section_id": 0, "lane_id": lane_id,
                    "samples": [{"s_m": float(x),
                                 "center_world_m": [float(x), center_y, 0.]}
                                for x in range(start, stop+1, 2)]}
        static = {"opendrive_sha256": "test", "sample_spacing_m": 2.,
                  "roads": [strip(1, 0, 10, 0., 4., 1),
                            strip(2, 10, 60, 0., 4., 1),
                            strip(2, 10, 60, 4., 4., -1),
                            strip(3, 10, 60, 0., 4., 1)],
                  "lanes": [lane(1, 0, 10, 0., 1),
                            lane(2, 10, 60, 0., 1),
                            lane(2, 10, 60, 4., -1),
                            lane(3, 10, 60, 0., 1)],
                  "topology": [{"from": [1, 0, 1], "to": [2, 0, 1]}]}
        route = [{"position_m": [float(x), 0., 0.], "road_id": 1 if x <= 10 else 2,
                  "section_id": 0, "lane_id": 1, "s_m": float(x)}
                 for x in range(0, 52)]
        result = project_ego_road_corridor(
            static, route, {"position_m": [2., 0., 0.], "rotation_deg": [0., 0., 0.]},
            {"x": 1.5, "y": 0., "z": 2.4}, 2,
            {"road_id": 1, "section_id": 0, "lane_id": 1, "s_m": 2.})
        self.assertEqual(result["ego_road"]["road_id"], 1)
        self.assertEqual(result["topology_links"], [{"from": [1, 0, 1], "to": [2, 0, 1]}])
        self.assertEqual({x["road_id"] for x in result["source_ids"] if x}, {1, 2})
        self.assertEqual({w for w in result["width_m"] if w is not None}, {4., 8.})
        self.assertTrue(any(result["break_before"]))
        self.assertFalse(any(x and x["road_id"] == 3 for x in result["source_ids"] if x))
        self.assertTrue(result["valid"][1])  # Near road survives the old elevation cut.
        self.assertEqual(result["matching"]["ego"]["source"], "exact_match")
        self.assertTrue(all(m["valid"] == v for m, v in
                            zip(result["matching"]["points"], result["valid"])))
        self.assertIn("route_fallback", {m["source"] for m in result["matching"]["points"]})
        static["topology"] = []
        route_only = project_ego_road_corridor(
            static, route, {"position_m": [2., 0., 0.], "rotation_deg": [0., 0., 0.]},
            {"x": 1.5, "y": 0., "z": 2.4}, 2,
            {"road_id": 1, "section_id": 0, "lane_id": 1, "s_m": 2.})
        self.assertIn(2, {x["road_id"] for x in route_only["source_ids"] if x})
        self.assertIn("nearest_route_segment",
                      {m["source"] for m in route_only["matching"]["points"]})
        self.assertEqual(route_only["topology_links"], [])

    def test_route_fallback_rejects_missing_or_distant_lane_source(self):
        def static_at(y):
            return {"opendrive_sha256": "test", "sample_spacing_m": 2.,
                    "topology": [{"from": [1, 0, 1], "to": [2, 0, 1]}],
                    "roads": [{"road_id": road_id, "section_id": 0,
                               "lane_ids": [1],
                               "samples": [{"s_m": float(x),
                                            "center_world_m": [float(x), center_y, 0.],
                                            "left_world_m": [float(x), center_y-2, 0.],
                                            "right_world_m": [float(x), center_y+2, 0.],
                                            "width_m": 4., "junction_id": None}
                                           for x in xs]}
                              for road_id, center_y, xs in
                              ((1, 0., range(0, 11, 2)), (2, y, range(12, 51, 2)))],
                    "lanes": [{"road_id": 2, "section_id": 0, "lane_id": 1,
                               "samples": [{"s_m": float(x),
                                            "center_world_m": [float(x), y, 0.]}
                                           for x in range(12, 51, 2)]}]}
        route = [{"position_m": [float(x), 0., 0.],
                  "road_id": 1 if x <= 10 else 2,
                  "section_id": 0, "lane_id": 1, "s_m": float(x)}
                 for x in range(0, 45)]
        sensor = {"x": 1.5, "y": 0., "z": 2.4}
        pose = {"position_m": [12.2, 0., 0.], "rotation_deg": [0., 0., 0.]}
        ego_key = {"road_id": 1, "section_id": 0, "lane_id": 1, "s_m": 10.}
        good = project_ego_road_corridor(static_at(0), route, pose, sensor, 12, ego_key)
        self.assertEqual(good["matching"]["ego"]["source"], "route_fallback")
        self.assertLess(good["matching"]["ego"]["distance_to_route"], 2.)
        no_topology = static_at(0)
        no_topology["topology"] = []
        route_only = project_ego_road_corridor(no_topology, route, pose, sensor, 12, ego_key)
        self.assertEqual(route_only["matching"]["ego"]["source"],
                         "nearest_route_segment")
        self.assertEqual(route_only["topology_links"], [])
        gap = static_at(0)
        gap["roads"][1]["samples"] = [sample for sample in gap["roads"][1]["samples"]
                                      if sample["s_m"] < 18 or sample["s_m"] > 24]
        partial = project_ego_road_corridor(gap, route, pose, sensor, 12, ego_key)
        self.assertFalse(partial["valid"][2])
        self.assertIsNone(partial["centerline_ego_m"][2])
        self.assertIsNone(partial["width_m"][2])
        self.assertFalse(partial["matching"]["points"][2]["valid"])
        with self.assertRaisesRegex(ValueError, "route-constrained"):
            project_ego_road_corridor(static_at(3), route, pose, sensor, 12, ego_key)
        missing = static_at(0)
        missing["lanes"] = []
        with self.assertRaisesRegex(ValueError, "route-constrained"):
            project_ego_road_corridor(missing, route, pose, sensor, 12, ego_key)
        far_pose = {"position_m": [12.2, 4., 0.], "rotation_deg": [0., 0., 0.]}
        with self.assertRaisesRegex(ValueError, "more than 2 m from official route"):
            project_ego_road_corridor(static_at(0), route, far_pose, sensor, 12, ego_key)

    def test_recorded_road_6_to_283_overlap_prefers_official_route(self):
        root = Path("/tmp/radarvla_v5_smoke_20261013/episode")
        if not (root / "frames/000181/frame.json").is_file():
            self.skipTest("single failed pilot is not available on this machine")
        route = json.loads((root / "route.json").read_text())["waypoints"]
        static = json.loads((root / "static_map.json").read_text())
        frame = json.loads((root / "frames/000181/frame.json").read_text())
        map_row = json.loads((root / "frames/000181/map.json").read_text())["map"]
        sensor = json.loads((root / "metadata.json").read_text())["sensors"]["semantic_lidar"]
        pose = copy.deepcopy(frame["ego"]["pose"])
        waypoint = {**map_row["ego_road"], "s_m": 36.}
        before = project_ego_road_corridor(static, route, pose, sensor, 17, waypoint)
        self.assertEqual(before["matching"]["ego"]["source"], "exact_match")
        self.assertEqual([x["road_id"] for x in before["source_ids"] if x][:1], [283])
        pose["position_m"][0] = 183.12  # Offline probe, not the missing CARLA frame.
        after = project_ego_road_corridor(static, route, pose, sensor, 17, waypoint)
        self.assertEqual(after["matching"]["ego"]["source"], "route_fallback")
        self.assertLess(after["matching"]["ego"]["distance_to_route"], .1)
        self.assertLess(after["matching"]["ego"]["distance_to_candidate"], 2.)
        self.assertTrue(any(x and x["road_id"] == 283 for x in after["source_ids"]))
        self.assertFalse(any(x and x["road_id"] == 252 for x in after["source_ids"][:2]))
        self.assertEqual([x["valid"] for x in after["matching"]["points"]], after["valid"])
        self.assertTrue(all(point["distance_to_candidate"] is not None and
                            point["distance_to_candidate"] < 2.
                            for point in after["matching"]["points"]
                            if point["valid"] and point["source"] != "exact_match"))

    def test_near_agent_bbox_intersection_uses_50m(self):
        actor = {"pose": {"position_m": [51, 0, 0],
                           "rotation_deg": [0, 0, 0]},
                 "bbox": {"center_actor_m": [0, 0, 0],
                          "rotation_actor_deg": [0, 0, 0],
                          "extent_m": [2, 1, 1]}}
        self.assertTrue(actor_intersects_fov(actor, [0, 0, 0], 0, 50))
        actor["pose"]["position_m"][0] = 54
        self.assertFalse(actor_intersects_fov(actor, [0, 0, 0], 0, 50))

    def test_near_roads_include_geometry_without_lidar_support(self):
        def strip(road_id, y):
            samples = []
            for x in range(0, 62, 2):
                samples.append({"s_m": float(x), "center_world_m": [float(x), y, 0],
                                "left_world_m": [float(x), y - 2, 0],
                                "right_world_m": [float(x), y + 2, 0],
                                "width_m": 4, "junction_id": None})
            return {"road_id": road_id, "section_id": 0, "direction": -1,
                    "lane_ids": [-1], "samples": samples}
        opposite = strip(1, 4)
        opposite["direction"] = 1
        opposite["lane_ids"] = [1]
        static = {"opendrive_sha256": "test", "sample_spacing_m": 2,
                  "roads": [strip(1, 0), opposite, strip(2, 8), strip(3, 80)]}
        pose = {"position_m": [0, 0, 0], "rotation_deg": [0, 0, 0]}
        sensor = {"x": 0, "y": 0, "z": 0, "yaw": 0}
        lidar = np.array([[x, y, 0, 1, 0, 7] for x in range(2, 49)
                          for y in (-1.5, 0, 1, 2, 3, 4, 5.5)], dtype=float)
        result = project_near_roads(static, pose, sensor, lidar)
        self.assertEqual([road["road_id"] for road in result["roads"]], [1, 2])
        one, two = result["roads"]
        self.assertGreater(sum(one["centerline_valid"]), 5)
        self.assertTrue(one["width_valid"])
        self.assertEqual(one["width_m"], 8)
        self.assertTrue(any(two["centerline_valid"]))
        self.assertFalse(any(two["lidar_support_mask"]))
        self.assertTrue(two["width_valid"])
        self.assertFalse(two["width_lidar_supported"])
        self.assertTrue(all(np.hypot(*point) <= 50.1 for point in
                            one["centerline_radar_m"] if point is not None))

    def test_near_road_does_not_interpolate_across_source_gap(self):
        samples = [{"s_m": float(x), "center_world_m": [float(x), 0, 0],
                    "left_world_m": [float(x), -2, 0],
                    "right_world_m": [float(x), 2, 0],
                    "width_m": 4, "junction_id": None}
                   for x in (2, 4, 6, 26, 28, 30)]
        static = {"opendrive_sha256": "gap", "sample_spacing_m": 2,
                  "roads": [{"road_id": 4, "section_id": 0, "direction": -1,
                             "lane_ids": [-1], "samples": samples}]}
        pose = {"position_m": [0, 0, 0], "rotation_deg": [0, 0, 0]}
        road = project_near_roads(static, pose, {"x": 0, "y": 0, "z": 0, "yaw": 0},
                                  np.zeros((0, 6)))["roads"][0]
        self.assertTrue(road["centerline_valid"][0])
        self.assertFalse(road["centerline_valid"][8])
        self.assertTrue(road["centerline_valid"][-1])

    def test_scatterer_budget_is_deterministic_and_keeps_dynamic_returns(self):
        points = np.zeros((5000, 6), dtype=float)
        points[:, 0] = np.linspace(2, 115, len(points))
        points[:, 1] = np.linspace(-25, 25, len(points))
        points[:, 5] = 1
        points[::5, 5] = 10
        speed = np.arange(len(points), dtype=float)
        a, av, report = select_scatterers(points, speed, 256)
        b, bv, _ = select_scatterers(points, speed, 256)
        self.assertLessEqual(len(a), 256)
        self.assertGreater(report["dynamic_scatterers_after_sampling"], 0)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(av, bv)
        self.assertLess(report["scatterers_in_radar_fov"], len(points))

    def test_carla_3d_transform_matches_installed_carla_binding(self):
        matrix = carla_rotation_matrix([10, 20, 30])
        expected = np.array([[.81379765, -.44096962, -.37852228],
                             [.46984631, .88256407, -.01802830],
                             [.34202012, -.16317591, .92541653]])
        np.testing.assert_allclose(matrix, expected, atol=1e-6)

    def test_carriageway_outer_boundaries_and_official_route(self):
        static = build_static_map(FakeMap())
        self.assertEqual(len(static["roads"]), 1)
        strip = static["roads"][0]["samples"][0]
        self.assertAlmostEqual(strip["width_m"], 7.0)
        route = [{"position_m": [float(x), 1.75, 0.0], "road_option": option,
                  "road_id": 1, "section_id": 0, "lane_id": -1, "s_m": float(x)}
                 for x, option in ((0, "LANEFOLLOW"), (2, "RIGHT"), (4, "LANEFOLLOW"))]
        pose = {"position_m": [0, 1.75, 0], "rotation_deg": [0, 0, 0]}
        frame_map = project_frame(static, route, pose, 10,
                                  {"x": 0, "y": 0, "z": 0, "yaw": 0})
        self.assertEqual([p["road_option"] for p in frame_map["planned_route"]],
                         ["RIGHT", "LANEFOLLOW"])
        self.assertEqual(frame_map["navigation_instruction"], "turn right")
        self.assertTrue(all(p["width_valid"] for p in frame_map["route_corridor"]))

    def test_dynamic_point_unique_bbox_and_static_radial_speed(self):
        actor = {"actor_type": "vehicle", "pose": {"position_m": [10, 0, 0],
                 "rotation_deg": [0, 0, 0]}, "velocity_world_mps": [5, 0, 0],
                 "bbox": {"center_actor_m": [0, 0, 0], "rotation_actor_deg": [0, 0, 0],
                          "extent_m": [2, 1, 1]}}
        ego = {"pose": {"position_m": [0, 0, 0], "rotation_deg": [0, 0, 0]},
               "velocity_world_mps": [2, 0, 0]}
        lidar = np.array([[10, 0, 0, 1, 999, 10], [8, 0, 0, 1, 888, 1],
                          [20, 0, 0, 1, 777, 10]], dtype=float)
        kept, speed, report = prepare_scatterers(lidar, {"ego": ego, "actors": [actor]},
                                                  {"x": 0, "y": 0, "z": 0,
                                                   "yaw": 0, "pitch": 0, "roll": 0})
        self.assertEqual(len(kept), 2)
        self.assertAlmostEqual(speed[0], 3.0)
        self.assertAlmostEqual(speed[1], -2.0)
        self.assertEqual(report["dynamic_rejected"], 1)

    def test_dense_unfold_has_finite_cells_and_explicit_reliability(self):
        shape = (256, 107)
        rng = np.random.default_rng(3)
        power = rng.random(shape).astype(np.float32)
        folded = np.zeros(shape, dtype=np.float32)
        frames = [{"carla_frame": i, "timestamp_s": i*0.1,
                   "ego": {"pose": {"position_m": [0, 0, 0],
                                    "rotation_deg": [0, 0, 0]}}} for i in range(3)]
        products = [{"power_peak_ra": power,
                     "doppler_at_peak_ra_mps": folded} for _ in frames]
        axes = {"range_m": np.arange(256, dtype=float),
                "azimuth_deg": np.arange(-53, 54, dtype=float),
                "velocity_mps": np.arange(-32, 32, dtype=float)*2}
        result = unfold_frame(1, frames, products, axes,
                              {"x": 0, "y": 0, "z": 0, "yaw": 0})
        self.assertEqual(result["doppler_unfolded_dense_ra_mps"].shape, shape)
        self.assertTrue(np.isfinite(result["doppler_unfolded_dense_ra_mps"]).all())
        self.assertTrue(np.all(result["reliable_mask"] <=
                               (result["source_code"] == 1)))


if __name__ == "__main__":
    unittest.main()
