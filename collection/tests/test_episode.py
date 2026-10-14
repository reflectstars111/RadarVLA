"""Offline contract checks; no CARLA import or server required."""
import json
import tempfile
import unittest
from pathlib import Path

from episode import SCHEMA, PREVIOUS_SCHEMA, OLDER_SCHEMA, LEGACY_SCHEMA, build_anchors, merge_events, read_jsonl, validate, write_json
from run import selected_route


def actor(actor_id, x, type_id="vehicle.test"):
    return {"actor_id": actor_id, "type_id": type_id,
            "pose": {"position_m": [float(x), 0.0, 0.0],
                     "rotation_deg": [0.0, 0.0, 0.0]}}


def episode(root, count, *, gap=None, missing_actor=None, schema=SCHEMA):
    write_json(root / "metadata.json", {"schema": schema, "fps": 10.0,
              "status": "complete", "episode_id": "test-episode",
              "actor_id_scope": "episode", "agent_label_range_m": 50,
              "map_label_range_m": 50})
    write_json(root / "route.json", {"schema": schema,
              "waypoints": [{"road_option": "LANEFOLLOW"},
                            {"road_option": "RIGHT"}]})
    write_json(root / "static_map.json", {"opendrive_sha256": "test-hash",
                                         "roads": [{"road_id": 1, "section_id": 0,
                                                    "lane_ids": [1], "samples": []}],
                                         "topology": []})
    frames = root / "frames"
    frames.mkdir()
    for index in range(count):
        frame = frames / ("%06d" % index)
        frame.mkdir()
        (frame / "front_rgb.png").write_bytes(b"camera")
        (frame / "semantic_lidar.npy").write_bytes(b"lidar")
        fid = index + (1 if gap is not None and index >= gap else 0)
        row = {"carla_frame": fid, "timestamp_s": fid / 10.0,
               "episode_id": "test-episode",
               "camera_frame": fid, "lidar_frame": fid,
               "sensors": {"camera": "front_rgb.png", "semantic_lidar": "semantic_lidar.npy"},
               "agents": "agents.json", "map": "map.json",
               "route_progress_index": 0,
               "navigation": {"road_option": "RIGHT", "instruction": "turn right",
                              "source": "official dense route RoadOption"},
               "ego": actor(1, fid),
               "actors": [] if missing_actor is not None and index == missing_actor
               else [actor(2, fid + 25)]}
        write_json(frame / "frame.json", row)
        write_json(frame / "agents.json", {"carla_frame": fid, "agents": row["actors"]})
        if schema == SCHEMA:
            map_data = {"source_map_sha256": "test-hash",
                        "target_schema": "ego_road_corridor_50m_v1",
                        "geometry_source": "CARLA OpenDRIVE outer Driving-lane boundaries",
                        "coordinate_frame": "ego x-forward y-right",
                        "horizon_m": 50, "ego_road": {"road_id": 1, "section_id": 0,
                                                    "lane_id": 1},
                        "centerline_ego_m": [None] * 16, "width_m": [None] * 16,
                        "valid": [False] * 16, "source_ids": [None] * 16,
                        "break_before": [False] * 16, "topology_links": [],
                        "stop_reason": None}
        elif schema in (PREVIOUS_SCHEMA, OLDER_SCHEMA):
            map_data = {"source_map_sha256": "test-hash",
                        "target_schema": "near_roads_50m_v2" if schema == PREVIOUS_SCHEMA else "near_roads_50m_v1",
                        "geometry_source": "CARLA OpenDRIVE driving-road envelope",
                        "horizon_m": 50,
                        "roads": []}
        else:
            map_data = {"source_map_sha256": "test-hash", "roads": [],
                        "route_progress_index": 0, "planned_route": [],
                        "route_corridor": []}
        write_json(frame / "map.json", {"carla_frame": fid, "map": map_data})


class RawEpisodeTests(unittest.TestCase):
    def test_v5_corridor_point_width_and_mask_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2)
            build_anchors(root)
            path = root / "frames/000000/map.json"
            row = json.loads(path.read_text())
            corridor = row["map"]
            corridor["centerline_ego_m"][1] = [4., 0.]
            corridor["width_m"][1] = 4.
            corridor["valid"][1] = True
            corridor["source_ids"][1] = {"road_id": 1, "section_id": 0, "lane_id": 1}
            write_json(path, row)
            self.assertEqual(validate(root)["map_visible_road_frames"], 1)
            corridor["width_m"][1] = None
            write_json(path, row)
            with self.assertRaisesRegex(ValueError, "value/mask"):
                validate(root)

    def test_future_and_missing_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 35, gap=5, missing_actor=8)
            self.assertEqual(build_anchors(root), 35)
            first = next(read_jsonl(root / "anchors.jsonl"))
            ego = first["futures_by_actor_id"]["1"]
            lead = first["futures_by_actor_id"]["2"]
            self.assertEqual(len(ego["valid"]), 30)
            self.assertEqual(ego["valid"][:4], [True] * 4)
            self.assertFalse(ego["valid"][4])  # Missing CARLA frame 5.
            self.assertEqual(ego["world_xyzyaw"][4], None)
            self.assertFalse(lead["valid"][8])  # Actor absent at index 8, CARLA frame 9.
            self.assertTrue(lead["valid"][9])
            self.assertEqual(validate(root, require_complete=True)["anchors"], 35)

    def test_sensor_sync_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2)
            build_anchors(root)
            frame = root / "frames/000001/frame.json"
            row = json.loads(frame.read_text())
            row["camera_frame"] += 1
            write_json(frame, row)
            with self.assertRaisesRegex(ValueError, "sensor frame mismatch"):
                validate(root)

    def test_legacy_episode_still_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2, schema=LEGACY_SCHEMA)
            build_anchors(root)
            self.assertEqual(validate(root)["schema"], LEGACY_SCHEMA)

    def test_previous_episode_still_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2, schema=PREVIOUS_SCHEMA)
            build_anchors(root)
            self.assertEqual(validate(root)["schema"], PREVIOUS_SCHEMA)

    def test_older_episode_still_validates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2, schema=OLDER_SCHEMA)
            build_anchors(root)
            self.assertEqual(validate(root)["schema"], OLDER_SCHEMA)

    def test_near_road_invalid_point_does_not_leak_geometry(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2, schema=PREVIOUS_SCHEMA)
            build_anchors(root)
            path = root / "frames/000000/map.json"
            row = json.loads(path.read_text())
            row["map"]["roads"] = [{"road_id": 1, "section_id": 0,
                                     "centerline_radar_m": [[12, 0]] + [None] * 15,
                                     "centerline_valid": [False] * 16,
                                     "lidar_support_mask": [False] * 16,
                                     "lidar_road_return_count": [0] * 16,
                                     "width_m": None, "width_valid": False,
                                     "width_lidar_supported": False}]
            write_json(path, row)
            with self.assertRaisesRegex(ValueError, "centerline/mask/support"):
                validate(root)

    def test_road_geometry_can_be_valid_without_lidar_support(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2, schema=PREVIOUS_SCHEMA)
            build_anchors(root)
            path = root / "frames/000000/map.json"
            row = json.loads(path.read_text())
            row["map"]["roads"] = [{"road_id": 1, "section_id": 0,
                                     "centerline_radar_m": [[12, 0]] + [None] * 15,
                                     "centerline_valid": [True] + [False] * 15,
                                     "lidar_support_mask": [False] * 16,
                                     "lidar_road_return_count": [0] * 16,
                                     "width_m": 4, "width_valid": True,
                                     "width_lidar_supported": False}]
            write_json(path, row)
            self.assertEqual(validate(root)["map_visible_road_frames"], 1)
            row["map"]["roads"][0]["lidar_support_mask"][0] = True
            write_json(path, row)
            with self.assertRaisesRegex(ValueError, "LiDAR support/mask"):
                validate(root)

    def test_future_reference_must_match_actor_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 32)
            build_anchors(root)
            path = root / "anchors.jsonl"
            rows = list(read_jsonl(path))
            rows[0]["futures_by_actor_id"]["2"]["world_xyzyaw"][0][0] += 1
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "future actor ID/trajectory"):
                validate(root)

    def test_actor_id_type_stability_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 2)
            build_anchors(root)
            frame = root / "frames/000001/frame.json"
            row = json.loads(frame.read_text())
            row["actors"][0]["type_id"] = "walker.pedestrian.test"
            write_json(frame, row)
            agents_file = frame.parent / "agents.json"
            agents = json.loads(agents_file.read_text())
            agents["agents"][0]["type_id"] = "walker.pedestrian.test"
            write_json(agents_file, agents)
            with self.assertRaisesRegex(ValueError, "unstable actor ID"):
                validate(root)

    def test_more_than_3000_frames_and_event_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 3001)
            self.assertEqual(build_anchors(root), 3001)
            (root / "events.jsonl").write_text('{"event":"actor_created","carla_frame":0}\n')
            (root / "traffic_event_history.jsonl").write_text(
                '{"event":"following_started","carla_frame":20}\n'
                '{"event":"braking_started","carla_frame":30}\n'
                '{"event":"scenario_completed","carla_frame":40}\n')
            write_json(root / "traffic_events.json", {"events": [{"index": 0,
                       "scenario_name": "Scenario2"}]})
            self.assertEqual(merge_events(root), 5)
            names = [row["event"] for row in read_jsonl(root / "events.jsonl")]
            self.assertEqual(names, ["scenario_selected", "actor_created",
                                     "following_started", "braking_started", "scenario_completed"])
            self.assertEqual(validate(root, require_complete=True)["frames"], 3001)

    def test_external_route_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "routes.xml"
            source.write_text('<routes><route id="1" town="Town01"/>'
                              '<route id="2" town="Town02"/></routes>')
            target = root / "selected.xml"
            selected_route(source, "2", target)
            self.assertEqual(target.read_text().count("<route "), 1)
            self.assertIn('id="2"', target.read_text())

    def test_scenario_selected_not_duplicated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "traffic_event_history.jsonl").write_text(
                '{"event":"scenario_selected","index":0,"carla_frame":1}\n')
            write_json(root / "traffic_events.json", {"events": [{"index": 0,
                       "scenario_name": "Scenario2", "other_actor_ids": [500],
                       "trigger_position": {"x": 1, "y": 2}}]})
            self.assertEqual(merge_events(root), 1)
            selected = next(read_jsonl(root / "events.jsonl"))
            self.assertEqual(selected["scenario_name"], "Scenario2")
            self.assertEqual(selected["other_actor_ids"], [500])
            self.assertEqual(selected["trigger_position"], {"x": 1, "y": 2})


if __name__ == "__main__":
    unittest.main()
