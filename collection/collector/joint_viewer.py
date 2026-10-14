"""Jupyter explorer for synchronized CARLA radar, near roads, and future GT."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from episode import FUTURE_STEPS, frame_paths
from radar_viewer import load_frame as load_radar_frame
from road_map import (actor_intersects_fov, carla_rotation_matrix, project_near_roads,
                      radar_pose, world_to_ego)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _frame_index(root):
    result = {}
    for path in frame_paths(root):
        frame_id = _read_json(path)["carla_frame"]
        if frame_id in result:
            raise ValueError("duplicate CARLA frame %s in episode" % frame_id)
        result[frame_id] = path.parent
    if not result:
        raise ValueError("episode has no frames")
    return result


def _anchor_offsets(path, expected_count):
    offsets = []
    with path.open("rb") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if line.strip():
                offsets.append(start)
    if len(offsets) != expected_count:
        raise ValueError("anchor/frame count mismatch: %d != %d" %
                         (len(offsets), expected_count))
    return offsets


def future_radar_xy(future, current_world_xyz, ego_pose, sensor):
    """Return current point plus 30 observed future points in anchor radar axes.

    Columns are [right, forward]. Invalid observations remain NaN to break
    plotted lines; no motion interpolation or map matching is performed.
    """
    valid, points = future["valid"], future["world_xyzyaw"]
    if len(valid) != FUTURE_STEPS or len(points) != FUTURE_STEPS:
        raise ValueError("future must contain 30 points and masks")
    position, rotation = radar_pose(ego_pose, sensor)
    output = np.full((FUTURE_STEPS + 1, 2), np.nan, dtype=float)
    def place(index, xyz):
        local = world_to_ego(xyz, position, rotation)
        output[index] = [local[1], local[0]]
    place(0, current_world_xyz)
    for step, (is_valid, point) in enumerate(zip(valid, points), 1):
        if type(is_valid) is not bool or (point is None) == is_valid:
            raise ValueError("future point/mask mismatch at step %d" % step)
        if is_valid:
            if len(point) != 4 or not np.isfinite(point).all():
                raise ValueError("invalid future pose at step %d" % step)
            place(step, point[:3])
    return output


class JointPreview:
    """Index a full raw episode once; load selected radar/anchor frames lazily."""

    def __init__(self, episode, radar_output):
        self.episode = Path(episode)
        self.radar_output = Path(radar_output)
        self.metadata = _read_json(self.episode / "metadata.json")
        self.static_map = None
        self.radar_meta = _read_json(self.radar_output / "radar_meta.json")
        if self.radar_meta.get("product_schema") not in ("dense_hann3d_v1", "dense_hann3d_v2"):
            raise ValueError("dense unfolded Doppler product required")
        self.radar_frames = self.radar_meta.get("frames", [])
        if not self.radar_frames:
            raise ValueError("radar product has no frames")
        self.raw_by_id = _frame_index(self.episode)
        self.anchor_path = self.episode / "anchors.jsonl"
        if not self.anchor_path.is_file():
            raise ValueError("full episode with anchors.jsonl is required")
        self.anchor_offsets = _anchor_offsets(self.anchor_path, len(self.raw_by_id))
        self.anchor_offset_by_id = dict(zip(self.raw_by_id, self.anchor_offsets))
        if len({item["carla_frame"] for item in self.radar_frames}) != len(self.radar_frames):
            raise ValueError("duplicate CARLA frame in radar product")
        missing = [item["carla_frame"] for item in self.radar_frames
                   if item["carla_frame"] not in self.raw_by_id]
        if missing:
            raise ValueError("radar CARLA frames missing from full episode: %s" % missing)

    def _load_anchor(self, frame_id):
        with self.anchor_path.open("rb") as stream:
            stream.seek(self.anchor_offset_by_id[frame_id])
            anchor = json.loads(stream.readline())
        if anchor.get("carla_frame") != frame_id or anchor.get("future_steps") != FUTURE_STEPS:
            raise ValueError("future anchor does not match CARLA frame %s" % frame_id)
        return anchor

    def load(self, index):
        if not 0 <= index < len(self.radar_frames):
            raise IndexError(index)
        frame_id = self.radar_frames[index]["carla_frame"]
        folder = self.raw_by_id[frame_id]
        frame = _read_json(folder / "frame.json")
        agents = _read_json(folder / frame["agents"])
        if frame["carla_frame"] != frame_id or agents["carla_frame"] != frame_id:
            raise ValueError("raw/agent CARLA frame mismatch at %s" % frame_id)
        sensor = self.metadata["sensors"]["semantic_lidar"]
        position, rotation = radar_pose(frame["ego"]["pose"], sensor)
        near_agents = [agent for agent in agents["agents"]
                       if actor_intersects_fov(agent, position, rotation, 50)]
        if (self.metadata.get("schema") in ("radarvla_collection_v4", "radarvla_collection_v5") and
                len(near_agents) != len(agents["agents"])):
            raise ValueError("v4 agents.json contains an actor outside 50 m FOV")
        radar_meta, ranges, azimuths, radar = load_radar_frame(self.radar_output, index)
        if radar_meta["frames"][index]["carla_frame"] != frame_id:
            raise ValueError("radar CARLA frame mismatch at index %d" % index)
        raw_map = _read_json(folder / frame["map"])
        if raw_map["carla_frame"] != frame_id:
            raise ValueError("map CARLA frame mismatch at %s" % frame_id)
        road_map = raw_map["map"]
        if road_map.get("target_schema") not in ("near_roads_50m_v2",
                                                   "ego_road_corridor_50m_v1"):
            if self.metadata.get("schema") not in ("radarvla_collection_v2", "radarvla_collection_v3"):
                raise ValueError("unsupported near-road map schema")
            if self.static_map is None:
                self.static_map = _read_json(self.episode / "static_map.json")
            lidar = np.load(folder / frame["sensors"]["semantic_lidar"], allow_pickle=False)
            road_map = project_near_roads(self.static_map, frame["ego"]["pose"],
                                          sensor, lidar)
        anchor = self._load_anchor(frame_id)
        needed = {str(frame["ego"]["actor_id"])} | {
            str(agent["actor_id"]) for agent in near_agents}
        if not needed <= set(anchor["futures_by_actor_id"]):
            raise ValueError("visible actor future missing at %s" % frame_id)
        if road_map.get("horizon_m") != 50 or not (
                isinstance(road_map.get("roads"), list) or
                road_map.get("target_schema") == "ego_road_corridor_50m_v1"):
            raise ValueError("near-road label malformed at %s" % frame_id)
        return {"carla_frame": frame_id, "frame": frame, "agents": near_agents,
                "map": road_map, "anchor": anchor,
                "range_m": ranges, "azimuth_deg": azimuths, "radar": radar}


def _road_lines(ax, road_map, *, annotate, show_support=False, ego_pose=None, sensor=None):
    if road_map.get("target_schema") == "ego_road_corridor_50m_v1":
        if ego_pose is None or sensor is None:
            raise ValueError("ego corridor preview needs ego pose and radar sensor")
        centers, mask = road_map["centerline_ego_m"], road_map["valid"]
        if len(centers) != 16 or len(mask) != 16:
            raise ValueError("ego corridor must have 16 samples")
        ego_rotation = carla_rotation_matrix(ego_pose["rotation_deg"])
        radar_position, radar_rotation = radar_pose(ego_pose, sensor)
        xy = np.full((16, 2), np.nan)
        for i, (point, valid) in enumerate(zip(centers, mask)):
            if (point is None) == valid:
                raise ValueError("ego corridor center/mask mismatch")
            if valid:
                world = np.asarray(ego_pose["position_m"]) + ego_rotation @ np.asarray([*point, 0.])
                x, y, _ = world_to_ego(world, radar_position, radar_rotation)
                xy[i] = [y, x]
        plot_xy = []
        for i, point in enumerate(xy):
            if road_map["break_before"][i]:
                plot_xy.append([np.nan, np.nan])
            plot_xy.append(point)
        plot_xy = np.asarray(plot_xy)
        ax.plot(plot_xy[:, 0], plot_xy[:, 1], color="black", linewidth=2.5,
                marker=".", markersize=4)
        if annotate:
            previous_id = None
            for i, (valid, source) in enumerate(zip(mask, road_map["source_ids"])):
                if valid and source["road_id"] != previous_id:
                    ax.annotate("road %s  %.1f m" %
                                (source["road_id"], road_map["width_m"][i]),
                                xy[i], xytext=(4, 3), textcoords="offset points", fontsize=8)
                    previous_id = source["road_id"]
        return
    colors = plt.get_cmap("tab20")
    for road in road_map["roads"]:
        points = road["centerline_radar_m"]
        mask = road["centerline_valid"]
        if len(points) != 16 or len(mask) != 16:
            raise ValueError("road centerline must have 16 samples")
        if any((point is None) == valid for point, valid in zip(points, mask)):
            raise ValueError("road centerline/mask mismatch")
        xy = np.array([[point[1], point[0]] if valid else [np.nan, np.nan]
                       for point, valid in zip(points, mask)], dtype=float)
        color = colors(int(road["road_id"]) % 20)
        ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.2, marker=".",
                markersize=4)
        if show_support:
            support = np.asarray(road["lidar_support_mask"], dtype=bool)
            if len(support) != 16 or np.any(support & ~np.asarray(mask, dtype=bool)):
                raise ValueError("road LiDAR support mask malformed")
            ax.scatter(xy[support, 0], xy[support, 1], c=[color], s=18,
                       edgecolors="black", linewidths=.4, zorder=3)
        if annotate and np.any(np.isfinite(xy[:, 0])):
            mid = np.where(np.isfinite(xy[:, 0]))[0]
            x, y = xy[mid[len(mid)//2]]
            label = "road %s" % road["road_id"]
            if road["width_valid"]:
                label += "  %.1f m" % road["width_m"]
            ax.annotate(label, (x, y), xytext=(3, 3), textcoords="offset points",
                        color=color, fontsize=8)


def _fov(ax, distance=50):
    angles = np.deg2rad(np.linspace(-53, 53, 101))
    ax.plot(distance*np.sin(angles), distance*np.cos(angles),
            color="0.6", linestyle="--", linewidth=.8)
    for angle in (-53, 53):
        ax.plot([0, distance*np.sin(np.deg2rad(angle))],
                [0, distance*np.cos(np.deg2rad(angle))],
                color="0.6", linestyle="--", linewidth=.8)


def render(preview, index):
    sample = preview.load(index)
    frame = sample["frame"]
    sensor = preview.metadata["sensors"]["semantic_lidar"]
    anchor = sample["anchor"]
    ranges, azimuths, radar = (sample["range_m"], sample["azimuth_deg"],
                               sample["radar"])
    dr, da = np.median(np.diff(ranges)), np.median(np.diff(azimuths))
    extent = [azimuths[0]-da/2, azimuths[-1]+da/2,
              ranges[-1]+dr/2, ranges[0]-dr/2]
    peak = float(preview.radar_meta["global_power_peak"])
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("invalid global radar power peak")
    power_db = 10*np.log10(np.maximum(radar["power"], np.finfo(np.float32).tiny)/peak)
    fig, panels = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    power_ax, doppler_ax, map_ax, traj_ax = panels.flat
    power_image = power_ax.imshow(power_db, extent=extent, aspect="auto", origin="upper",
                                  cmap="viridis", vmin=-60, vmax=0, interpolation="nearest")
    speed_image = doppler_ax.imshow(radar["dense"], extent=extent, aspect="auto",
                                    origin="upper", cmap="RdBu_r", vmin=-35, vmax=35,
                                    interpolation="nearest")
    power_ax.set(title="RA power", xlabel="azimuth (deg)", ylabel="range (m)")
    doppler_ax.set(title="Dense unfolded Doppler candidate; reliable %.1f%%" %
                   (100*radar["reliable"].mean()), xlabel="azimuth (deg)", ylabel="range (m)")
    fig.colorbar(power_image, ax=power_ax, label="dB relative to run peak")
    fig.colorbar(speed_image, ax=doppler_ax, label="m/s (+ receding)")

    _road_lines(map_ax, sample["map"], annotate=True, show_support=True,
                ego_pose=frame["ego"]["pose"], sensor=sensor)
    _fov(map_ax)
    map_ax.scatter([0], [0], c="black", marker="+", s=80, label="radar")
    radar_position, radar_rotation = radar_pose(frame["ego"]["pose"], sensor)
    ex, ey, _ = world_to_ego(frame["ego"]["pose"]["position_m"],
                             radar_position, radar_rotation)
    map_ax.scatter([ey], [ex], c="red", marker="s", s=36, label="ego")
    map_ax.annotate("Ego", (ey, ex), xytext=(4, 3),
                    textcoords="offset points", fontsize=8, color="red")
    corridor = sample["map"].get("target_schema") == "ego_road_corridor_50m_v1"
    road_title = "Ego road corridor (OpenDRIVE prior)" if corridor else \
                 "OpenDRIVE road prior (dots = LiDAR support)"
    if corridor and sample["map"].get("stop_reason"):
        road_title += " | stop: " + sample["map"]["stop_reason"]
    map_ax.set(xlim=(-50, 50), ylim=(-5, 52), aspect="equal",
               title=road_title,
               xlabel="radar right (m)", ylabel="radar forward (m)")

    _fov(traj_ax)
    _road_lines(traj_ax, sample["map"], annotate=False,
                ego_pose=frame["ego"]["pose"], sensor=sensor)
    visible = [frame["ego"]] + sample["agents"]
    palette = plt.get_cmap("tab20")
    all_xy = []
    for number, actor in enumerate(visible):
        actor_id = actor["actor_id"]
        future = anchor["futures_by_actor_id"][str(actor_id)]
        xy = future_radar_xy(future, actor["pose"]["position_m"],
                             frame["ego"]["pose"], sensor)
        all_xy.append(xy)
        is_ego = number == 0
        color = "black" if is_ego else palette((number-1) % 20)
        traj_ax.plot(xy[:, 0], xy[:, 1], color=color,
                     linewidth=2.6 if is_ego else 1.2, alpha=1 if is_ego else .8)
        traj_ax.scatter(xy[0, 0], xy[0, 1], c=[color], s=45 if is_ego else 18,
                        marker="s" if is_ego else "o", zorder=3)
        traj_ax.annotate("ego %s" % actor_id if is_ego else str(actor_id),
                         xy[0], xytext=(3, 3), textcoords="offset points",
                         fontsize=8, color=color)
    finite = np.concatenate(all_xy)[np.isfinite(np.concatenate(all_xy)).all(axis=1)]
    x_extent = max(50, np.max(np.abs(finite[:, 0]))+5)
    y_low = min(-5, np.min(finite[:, 1])-5)
    y_high = max(52, np.max(finite[:, 1])+5)
    traj_ax.set(xlim=(-x_extent, x_extent), ylim=(y_low, y_high), aspect="equal",
                title="Observed future GT: ego + %d visible agents (30 frames)" %
                len(sample["agents"]), xlabel="radar right (m)",
                ylabel="radar forward (m)")
    fig.suptitle("CARLA frame %s | t=%.3f s | future NaNs break trajectories" %
                 (sample["carla_frame"], frame["timestamp_s"]))
    return fig


def build_explorer(episode, radar_output):
    """Build a Jupyter slider; each selection loads one synchronized sample."""
    import ipywidgets as widgets
    from IPython.display import display

    preview = JointPreview(episode, radar_output)
    slider = widgets.IntSlider(value=0, min=0, max=len(preview.radar_frames)-1,
                               step=1, description="Frame", continuous_update=False,
                               layout=widgets.Layout(width="80%"))
    output = widgets.Output()
    def draw(index):
        with output:
            output.clear_output(wait=True)
            fig = render(preview, index)
            display(fig)
            plt.close(fig)
    slider.observe(lambda change: draw(change["new"]), names="value")
    draw(0)
    return widgets.VBox([slider, output])
