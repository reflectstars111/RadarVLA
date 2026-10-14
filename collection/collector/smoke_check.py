"""Read-only selection and acceptance report for a v5 raw/radar smoke run."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from episode import SCHEMA, frame_paths, read_jsonl, validate


def _rows(episode):
    return [json.loads(path.read_text()) for path in frame_paths(episode)]


def select_moving_window(episode, count=7, min_speed_mps=1.0):
    """Prefer a moving interval containing a tracked visible actor."""
    if count < 3:
        raise ValueError("at least three frames are needed for temporal Doppler")
    rows = _rows(episode)
    anchors = {row["carla_frame"]: row for row in read_jsonl(Path(episode) / "anchors.jsonl")}
    best = None
    for start in range(len(rows) - count + 1):
        window = rows[start:start + count]
        if any(b["carla_frame"] != a["carla_frame"] + 1
               for a, b in zip(window, window[1:])):
            continue
        if not all(math.dist(row["ego"]["velocity_world_mps"], [0, 0, 0]) >= min_speed_mps
                   for row in window):
            continue
        if not all(all(anchors[row["carla_frame"]]["futures_by_actor_id"]
                       [str(row["ego"]["actor_id"])]["valid"]) for row in window):
            continue
        full_agents = 0
        moving_agents = 0
        for offset, row in enumerate(window):
            visible = json.loads((Path(episode) / "frames" / ("%06d" % (start + offset))
                                  / row["agents"]).read_text())["agents"]
            for actor in visible:
                future = anchors[row["carla_frame"]]["futures_by_actor_id"][str(actor["actor_id"])]
                full_agents += all(future["valid"])
                moving_agents += math.dist(actor["velocity_world_mps"], [0, 0, 0]) >= .5
        score = (moving_agents, full_agents)
        if best is None or score > best[0]:
            best = (score, start)
    if best is None:
        raise ValueError("no consecutive moving window with complete ego future")
    return best[1]


def report(episode, radar):
    from e2e_agent_sem_lidar2shenron_package.carla_kradar_episode import validate_products

    episode, radar = Path(episode), Path(radar)
    raw = validate(episode, require_complete=True)
    if raw["schema"] != SCHEMA:
        raise ValueError("smoke run is not a v5 episode")
    product = validate_products(episode, radar)
    metadata = json.loads((radar / "radar_meta.json").read_text())
    route = json.loads((episode / "route.json").read_text())
    raw_frames = _rows(episode)
    sampled_frames = metadata["frames"]
    valid_road_points = []
    for index, row in enumerate(raw_frames):
        path = episode / "frames" / ("%06d" % index) / row["map"]
        valid_road_points.append(sum(json.loads(path.read_text())["map"]["valid"]))
    dynamic_frames = sum(bool(row.get("moving_agent_ids_in_fov")) for row in sampled_frames)
    cells = product["frames"] * 256 * 107
    if dynamic_frames == 0:
        quality = "UNTESTED"
        reason = "no moving labeled agent in sampled 50 m FOV"
    elif product["reliable_cells"] == 0:
        quality = "FAIL"
        reason = "moving agents are present but no reliable Doppler cells exist"
    else:
        quality = "UNTESTED"
        reason = "reliable cells exist, but this smoke test does not attribute them to dynamic agents or verify speed error"
    return {
        "technical_chain": "PASS" if raw["full_ego_future_30"] > 0 and
            route["source"] == "Leaderboard.set_global_plan full dense world route" and
            product["product_schema"] == "dense_hann3d_v2" else "FAIL",
        "dynamic_speed_supervision": quality,
        "dynamic_speed_reason": reason,
        "raw": raw,
        "route": {"source": route["source"], "points": len(route["waypoints"]),
                  "road_options": sorted({point["road_option"] for point in route["waypoints"]})},
        "map": {"frames_with_valid_points": sum(bool(x) for x in valid_road_points),
                "valid_points_total": sum(valid_road_points),
                "mean_valid_points_per_frame": sum(valid_road_points) / len(valid_road_points)},
        "radar": {**product, "source_start_index": metadata["source_start_index"],
                  "carla_frame_ids": [row["carla_frame"] for row in sampled_frames],
                  "reliable_fraction": product["reliable_cells"] / cells,
                  "moving_agent_fov_frames": dynamic_frames},
        "agent_future_30": "PASS" if raw["full_visible_agent_future_30"] else "UNTESTED",
        "training_loader_ready": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pick = sub.add_parser("select-window")
    pick.add_argument("episode", type=Path)
    pick.add_argument("--count", type=int, default=7)
    check = sub.add_parser("report")
    check.add_argument("episode", type=Path)
    check.add_argument("radar", type=Path)
    args = parser.parse_args()
    if args.command == "select-window":
        print(select_moving_window(args.episode, args.count))
    else:
        result = report(args.episode, args.radar)
        print(json.dumps(result, indent=2))
        if result["technical_chain"] != "PASS":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
