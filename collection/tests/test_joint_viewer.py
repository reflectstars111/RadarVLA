"""Offline checks for the synchronized notebook inputs and GT coordinates."""
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from joint_viewer import JointPreview, future_radar_xy, render


def _json(path, data):
    path.write_text(json.dumps(data))


def _future(first=None):
    points = [None] * 30
    valid = [False] * 30
    if first is not None:
        points[0] = first
        valid[0] = True
    return {"world_xyzyaw": points, "valid": valid}


def _fixture(root):
    raw, radar = root / "episode", root / "radar"
    (raw / "frames").mkdir(parents=True)
    (radar / "frames").mkdir(parents=True)
    _json(raw / "metadata.json", {"schema": "radarvla_collection_v4",
                                      "sensors": {"semantic_lidar":
                                                  {"x": 0, "y": 0, "z": 0,
                                                   "yaw": 0, "pitch": 0, "roll": 0}}})
    _json(radar / "radar_meta.json", {"product_schema": "dense_hann3d_v2",
                                         "global_power_peak": 1,
                                         "frames": [{"carla_frame": 100},
                                                    {"carla_frame": 101}]})
    np.savez(radar / "radar_axes.npz", range_m=np.arange(256),
             azimuth_deg=np.arange(-53, 54))
    for index, fid in enumerate((100, 101)):
        folder = raw / "frames" / ("%06d" % index)
        folder.mkdir()
        ego = {"actor_id": 1, "pose": {"position_m": [0, 0, 0],
                                         "rotation_deg": [0, 0, 0]}}
        agent = {"actor_id": 2, "pose": {"position_m": [10, 0, 0],
                                           "rotation_deg": [0, 0, 0]},
                 "bbox": {"center_actor_m": [0, 0, 0],
                          "rotation_actor_deg": [0, 0, 0],
                          "extent_m": [1, 1, 1]}}
        _json(folder / "frame.json", {"carla_frame": fid, "timestamp_s": fid/10,
                                       "ego": ego, "actors": [agent],
                                       "agents": "agents.json", "map": "map.json"})
        _json(folder / "agents.json", {"carla_frame": fid, "agents": [agent]})
        road = {"road_id": 7, "section_id": 0,
                "centerline_radar_m": [[10, 0]] + [None]*15,
                "centerline_valid": [True] + [False]*15,
                "lidar_support_mask": [False]*16,
                "width_m": None, "width_valid": False}
        _json(folder / "map.json", {"carla_frame": fid,
                                     "map": {"target_schema": "near_roads_50m_v2",
                                             "horizon_m": 50, "roads": [road]}})
        radar_folder = radar / "frames" / ("%06d" % index)
        radar_folder.mkdir()
        shape = (256, 107)
        np.savez(radar_folder / "folded_ra.npz",
                 power_peak_ra=np.ones(shape, np.float32),
                 doppler_at_peak_ra_mps=np.zeros(shape, np.float32),
                 peak_doppler_bin=np.zeros(shape, np.int16))
        np.savez(radar_folder / "unfolded_ra.npz",
                 doppler_unfolded_dense_ra_mps=np.zeros(shape, np.float32),
                 ambiguity_order=np.zeros(shape, np.int8),
                 source_code=np.full(shape, 5, np.uint8),
                 reliable_mask=np.zeros(shape, bool))
    with (raw / "anchors.jsonl").open("w") as stream:
        for fid in (100, 101):
            stream.write(json.dumps({"carla_frame": fid, "future_steps": 30,
                                     "futures_by_actor_id": {
                                         "1": _future([1, 0, 0, 0]),
                                         "2": _future([11, 0, 0, 0])}}) + "\n")
    return raw, radar


class JointViewerTests(unittest.TestCase):
    def test_v5_single_corridor_preview_and_break(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, radar = _fixture(Path(temp))
            metadata = json.loads((raw / "metadata.json").read_text())
            metadata["schema"] = "radarvla_collection_v5"
            _json(raw / "metadata.json", metadata)
            for folder in (raw / "frames").iterdir():
                path = folder / "map.json"
                row = json.loads(path.read_text())
                row["map"] = {"target_schema": "ego_road_corridor_50m_v1",
                              "horizon_m": 50,
                              "centerline_ego_m": [[3., 0.], [6., 0.], None,
                                                   [12., 2.]] + [None]*12,
                              "width_m": [4., 4., None, 8.] + [None]*12,
                              "valid": [True, True, False, True] + [False]*12,
                              "break_before": [False, False, False, True] + [False]*12,
                              "source_ids": [{"road_id": 1}, {"road_id": 1}, None,
                                             {"road_id": 2}] + [None]*12}
                _json(path, row)
            preview = JointPreview(raw, radar)
            fig = render(preview, 0)
            self.assertEqual(len(fig.axes), 6)
            self.assertIn("Ego road corridor", fig.axes[2].get_title())
            matplotlib.pyplot.close(fig)

    def test_future_uses_anchor_radar_pose_and_keeps_missing_steps(self):
        future = _future([11, 0, 0, 0])
        pose = {"position_m": [0, 0, 0], "rotation_deg": [0, 0, 90]}
        xy = future_radar_xy(future, [10, 0, 0], pose,
                             {"x": 0, "y": 0, "z": 0, "yaw": 0})
        np.testing.assert_allclose(xy[:2], [[-10, 0], [-11, 0]], atol=1e-6)
        self.assertTrue(np.isnan(xy[2:]).all())

    def test_synchronized_load_and_render(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, radar = _fixture(Path(temp))
            preview = JointPreview(raw, radar)
            for index, fid in enumerate((100, 101)):
                sample = preview.load(index)
                self.assertEqual(sample["carla_frame"], fid)
                self.assertEqual(sample["radar"]["dense"].shape, (256, 107))
                self.assertEqual(sample["map"]["roads"][0]["road_id"], 7)
                fig = render(preview, index)
                self.assertEqual(len(fig.axes), 6)  # Four panels and two colorbars.
                matplotlib.pyplot.close(fig)

    def test_mismatched_radar_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, radar = _fixture(Path(temp))
            meta = json.loads((radar / "radar_meta.json").read_text())
            meta["frames"][1]["carla_frame"] = 999
            _json(radar / "radar_meta.json", meta)
            with self.assertRaisesRegex(ValueError, "missing from full episode"):
                JointPreview(raw, radar)

    def test_mismatched_anchor_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, radar = _fixture(Path(temp))
            path = raw / "anchors.jsonl"
            lines = path.read_text().splitlines()
            row = json.loads(lines[0])
            row["carla_frame"] = 999
            lines[0] = json.dumps(row)
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "future anchor does not match"):
                JointPreview(raw, radar).load(0)


if __name__ == "__main__":
    unittest.main()
