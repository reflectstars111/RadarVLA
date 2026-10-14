"""Offline checks for real-episode window choice and raw/radar alignment."""
import json
import tempfile
import unittest
from pathlib import Path

from e2e_agent_sem_lidar2shenron_package.carla_kradar_episode import (
    select_frame_window, validate_source_window)
from episode import build_anchors, write_json
from smoke_check import select_moving_window
from test_episode import episode


class SmokeWindowTests(unittest.TestCase):
    def test_nonzero_window_and_single_sided_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 45)
            for i in range(45):
                path = root / "frames" / ("%06d" % i) / "frame.json"
                row = json.loads(path.read_text())
                row["ego"]["velocity_world_mps"] = [2., 0., 0.] if i >= 5 else [0., 0., 0.]
                row["actors"][0]["velocity_world_mps"] = [1., 0., 0.]
                write_json(path, row)
                agents_path = path.parent / "agents.json"
                agents = json.loads(agents_path.read_text())
                agents["agents"][0]["velocity_world_mps"] = [1., 0., 0.]
                write_json(agents_path, agents)
            build_anchors(root)
            self.assertEqual(select_moving_window(root, 7), 5)
            selected = select_frame_window(root, 5, 7)
            self.assertEqual([json.loads(p.read_text())["carla_frame"] for p in selected],
                             list(range(5, 12)))
            meta = {"source_start_index": 5,
                    "frames": [{"source_index": i, "carla_frame": i} for i in range(5, 12)]}
            self.assertEqual(validate_source_window(root, meta, 7), selected)
            meta["frames"][0]["source_index"] = 4
            with self.assertRaisesRegex(ValueError, "source index mismatch"):
                validate_source_window(root, meta, 7)
            meta["frames"][0]["source_index"] = 5
            meta["frames"][-1]["carla_frame"] = 999
            with self.assertRaisesRegex(ValueError, "CARLA frame mismatch"):
                validate_source_window(root, meta, 7)

    def test_missing_frame_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            episode(root, 7, gap=3)
            with self.assertRaisesRegex(ValueError, "missing CARLA frame"):
                select_frame_window(root, 0, 7)
            with self.assertRaisesRegex(ValueError, "outside"):
                select_frame_window(root, 7, 1)


if __name__ == "__main__":
    unittest.main()
