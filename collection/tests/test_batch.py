"""Offline checks for batch quotas, route preservation and event evidence."""
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "collector"))
import batch
from batch_selection import choose_with_quotas, episode_candidates


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_weather_copy_preserves_waypoints(self):
        source = self.root / "routes.xml"
        source.write_text('<routes><route id="3" town="Town02">'
                          '<waypoint x="1" y="2" z="0"/>'
                          '<waypoint x="9" y="8" z="0"/></route></routes>')
        result = self.root / "derived.xml"
        batch.prepare_route(source, "3", {"precipitation": 65}, result)
        import xml.etree.ElementTree as ET
        before = ET.parse(source).getroot().find("route")
        after = ET.parse(result).getroot().find("route")
        self.assertEqual([x.attrib for x in before.findall("waypoint")],
                         [x.attrib for x in after.findall("waypoint")])
        self.assertEqual(after.find("weather").get("precipitation"), "65")

    def test_official_route_slices_do_not_overlap(self):
        source = self.root / "routes.xml"
        source.write_text('<routes>' + ''.join(
            '<route id="%d" town="Town04"/>' % i for i in range(4)) + '</routes>')
        train = batch.route_entries({"xml": str(source), "max_routes": 2})
        test = batch.route_entries({"xml": str(source), "start_index": 2,
                                    "max_routes": 2})
        self.assertEqual([row["id"] for row in train], ["0", "1"])
        self.assertEqual([row["id"] for row in test], ["2", "3"])

    def episode(self, stages=True, moving=True):
        root = self.root / "episode"
        (root / "frames").mkdir(parents=True)
        anchors = []
        for index in range(60):
            folder = root / "frames" / ("%06d" % index)
            folder.mkdir()
            ego = {"actor_id": 1, "actor_type": "vehicle",
                   "pose": {"position_m": [index * .5, 0, 0]},
                   "velocity_world_mps": [5, 0, 0],
                   "commanded_control": {"brake": .2 if index == 25 else 0}}
            lead = {"actor_id": 2, "actor_type": "vehicle",
                    "pose": {"position_m": [index * .5 + 15, 0, 0]},
                    "velocity_world_mps": [3 if moving else 0, 0, 0]}
            (folder / "frame.json").write_text(json.dumps({
                "episode_id": "e1", "index": index, "carla_frame": 100 + index,
                "ego": ego, "actors": [lead]}))
            anchors.append({"carla_frame": 100 + index,
                            "futures_by_actor_id": {"1": {"valid": [index < 30] * 30}}})
        (root / "anchors.jsonl").write_text("".join(json.dumps(x) + "\n" for x in anchors))
        events = [{"event": "scenario_selected", "index": 0,
                   "scenario_name": "Scenario2", "other_actor_ids": [2],
                   "trigger_position": {"x": 5, "y": 0}}]
        for event, fid in (("actor_restored", 111), ("scenario_activated", 112),
                           ("following_started", 112), ("braking_started", 125),
                           ("scenario_completed", 130)):
            if stages or event != "braking_started":
                events.append({"event": event, "index": 0, "carla_frame": fid})
        (root / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in events))
        return root

    def test_scenario2_requires_history_motion_brake_and_future(self):
        rows, detail = episode_candidates(self.episode(), "scenario2", "rain", "train")
        self.assertTrue(rows)
        self.assertEqual(rows[0]["carla_frame"], 111)
        self.assertEqual(rows[-1]["carla_frame"], 129)
        self.assertEqual(len(detail["windows"]), 1)

    def test_missing_stage_or_stationary_actor_is_not_interaction(self):
        rows, _ = episode_candidates(self.episode(stages=False), "scenario2", "rain", "train")
        self.assertFalse(rows)
        self.temp.cleanup()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        rows, _ = episode_candidates(self.episode(moving=False), "scenario2", "rain", "train")
        self.assertFalse(rows)

    def test_quota_and_duplicate_guard(self):
        rows = [{"episode_id": "e", "carla_frame": i, "category": "normal",
                 "weather": "clear", "split": "train"} for i in range(5)]
        quota = {"normal": 2}
        picked = choose_with_quotas(rows, [], quota, {"train": 2}, {"clear": 2})
        self.assertEqual(len(picked), 2)
        self.assertFalse(choose_with_quotas(rows, picked, quota,
                                            {"train": 2}, {"clear": 2}))

    def test_radar_chunk_keeps_core_order(self):
        rows = [{"job_id": "a", "source_index": i, "carla_frame": 100 + i}
                for i in (1, 2, 3, 10, 11)]
        windows = list(batch.radar_windows(rows, core_limit=2))
        self.assertEqual([[x["source_index"] for x in core]
                          for _, core, _ in windows], [[1, 2], [3], [10, 11]])

    def test_finalized_partial_episode_can_be_selected_without_retry(self):
        episode = self.root / "partial"
        episode.mkdir()
        (episode / "metadata.json").write_text(json.dumps({"status": "failed"}))
        self.assertFalse(batch.finalized_episode(episode))
        (episode / "anchors.jsonl").write_text("")
        self.assertTrue(batch.finalized_episode(episode))
        (episode / "metadata.json").write_text(json.dumps({"status": "recording"}))
        self.assertFalse(batch.finalized_episode(episode))

    def test_parallel_radar_windows_keep_device_exclusive_and_index_order(self):
        rows = [{"job_id": "a", "episode_id": "e", "source_index": i,
                 "carla_frame": 100 + i} for i in range(4)]
        active = set()
        guard = threading.Lock()
        overlaps = []
        def fake_window(_config, _output, _job, core, _halo, device):
            with guard:
                if device in active:
                    overlaps.append(device)
                active.add(device)
            time.sleep(.02)
            with guard:
                active.remove(device)
            return ([{**core[0], "radar_product": str(device),
                      "radar_frame_index": 0}], None)
        config = {"runtime": {"radar_chunk_frames": 1, "radar_workers": 2,
                              "radar_devices": [2, 4]}}
        with patch.object(batch, "_process_radar_window", fake_window), \
             patch.object(batch, "tqdm", None):
            result, failures = batch.process_radar(config, self.root, rows)
        self.assertFalse(overlaps)
        self.assertFalse(failures)
        self.assertEqual([row["carla_frame"] for row in result], [100, 101, 102, 103])

    def test_scenario2_pilot_rejects_lead_behind_ego(self):
        episode = self.root / "pilot"
        (episode / "frames").mkdir(parents=True)
        for index in range(45):
            folder = episode / "frames" / ("%06d" % index)
            folder.mkdir()
            row = {"carla_frame": 100 + index,
                   "ego": {"actor_id": 1,
                           "pose": {"position_m": [index, 0, 0],
                                     "rotation_deg": [0, 0, 0]},
                           "bbox": {"extent_m": [2, 1, 1]},
                           "commanded_control": {"brake": .3 if index == 33 else 0}},
                   "actors": [{"actor_id": 2,
                               "pose": {"position_m": [index - 25, 0, 0]},
                               "bbox": {"extent_m": [2, 1, 1]},
                               "velocity_world_mps": [2, 0, 0]}]}
            (folder / "frame.json").write_text(json.dumps(row))
        history = [{"event": "scenario_selected", "index": 0,
                    "scenario_name": "Scenario2", "other_actor_ids": [2]}]
        for stage, fid in (("scenario_activated", 100), ("following_started", 100),
                           ("actor_restored", 101), ("braking_started", 133),
                           ("scenario_completed", 135)):
            history.append({"event": stage, "index": 0, "carla_frame": fid})
        (episode / "events.jsonl").write_text("".join(json.dumps(x) + "\n" for x in history))
        (episode / "anchors.jsonl").write_text("".join(json.dumps({
            "carla_frame": fid, "futures_by_actor_id": {
                "1": {"valid": [True] * 30}, "2": {"valid": [True] * 30}}}) + "\n"
            for fid in range(101, 106)))
        with self.assertRaisesRegex(ValueError, "unsafe measured lead placement"):
            batch.scenario2_pilot_evidence(episode, self.root / "annotation.json", "0")
        for index in range(45):
            path = episode / "frames" / ("%06d" % index) / "frame.json"
            row = json.loads(path.read_text())
            row["actors"][0]["pose"]["position_m"][0] = index + 25
            path.write_text(json.dumps(row))
        (self.root / "annotation.json").write_text('{}')
        episode.with_name("pilot.leaderboard_result.json").write_text(json.dumps({
            "_checkpoint": {"records": [{"status": "Failed - Route timeout",
                "infractions": {"collisions_vehicle": [
                    "Agent collided against object with type=vehicle.test and id=9 at (x=1000, y=1000, z=0)"]}}]}}))
        report = batch.scenario2_pilot_evidence(episode, self.root / "annotation.json", "0")
        self.assertAlmostEqual(report["restored_longitudinal_distance_m"], 25)
        self.assertTrue(report["ego_braked"])
        self.assertGreater(report["joint_future_30_anchors"], 0)
        self.assertEqual(report["route_status"], "Failed - Route timeout")


if __name__ == "__main__":
    unittest.main()
