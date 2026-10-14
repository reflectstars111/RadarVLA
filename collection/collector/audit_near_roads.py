"""Reproject existing raw frames into the v4 near-road target without CARLA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from road_map import actor_intersects_fov, project_near_roads, radar_pose


def audit(root, limit=None, start_index=0):
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    static_map = json.loads((root / "static_map.json").read_text())
    route_bytes = (root / "route.json").read_bytes()
    frames = sorted((root / "frames").glob("[0-9]*"))[start_index:]
    if limit is not None:
        frames = frames[:limit]
    types = {}
    ids_by_frame = {}
    rows = []
    for path in frames:
        frame = json.loads((path / "frame.json").read_text())
        lidar = np.load(path / frame["sensors"]["semantic_lidar"], allow_pickle=False)
        result = project_near_roads(static_map, frame["ego"]["pose"],
                                    metadata["sensors"]["semantic_lidar"], lidar)
        agents = json.loads((path / frame["agents"]).read_text())["agents"]
        radar_position, radar_rotation = radar_pose(
            frame["ego"]["pose"], metadata["sensors"]["semantic_lidar"])
        near_agents = [actor for actor in agents if
                       actor_intersects_fov(actor, radar_position, radar_rotation, 50)]
        actors = [frame["ego"]] + frame["actors"]
        ids = [actor["actor_id"] for actor in actors]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate actor ID in CARLA frame %s" % frame["carla_frame"])
        for actor in actors:
            actor_id = actor["actor_id"]
            if actor_id in types and types[actor_id] != actor["type_id"]:
                raise ValueError("actor ID changed type: %s" % actor_id)
            types[actor_id] = actor["type_id"]
        ids_by_frame[frame["carla_frame"]] = set(ids)
        valid = sum(sum(road["centerline_valid"]) for road in result["roads"])
        supported = sum(sum(road["lidar_support_mask"]) for road in result["roads"])
        total = 16 * len(result["roads"])
        if any((point is None) == mask for road in result["roads"]
               for point, mask in zip(road["centerline_radar_m"], road["centerline_valid"])):
            raise ValueError("invalid or hidden geometry leaked into near road target")
        if any(point is not None and np.linalg.norm(point) > 50.1
               for road in result["roads"] for point in road["centerline_radar_m"]):
            raise ValueError("label outside 50 m")
        if any(supported and not valid for road in result["roads"]
               for supported, valid in zip(road["lidar_support_mask"],
                                           road["centerline_valid"])):
            raise ValueError("LiDAR support without road geometry")
        rows.append({"carla_frame": frame["carla_frame"],
                     "road_count": len(result["roads"]),
                     "source_agent_count": len(agents),
                     "near_50m_agent_count": len(near_agents),
                     "near_50m_agent_ids": [actor["actor_id"] for actor in near_agents],
                     "geometry_valid_points": valid,
                     "lidar_supported_points": supported,
                     "total_centerline_points": total,
                     "width_geometry_valid_roads": sum(road["width_valid"] for road in result["roads"]),
                     "width_lidar_supported_roads": sum(road["width_lidar_supported"]
                                                        for road in result["roads"]),
                     "roads": [{"road_id": road["road_id"], "section_id": road["section_id"],
                                "geometry_valid_points": sum(road["centerline_valid"]),
                                "lidar_supported_points": sum(road["lidar_support_mask"]),
                                "width_valid": road["width_valid"],
                                "width_lidar_supported": road["width_lidar_supported"]}
                               for road in result["roads"]]})
    anchors_path = root / "anchors.jsonl"
    anchor_refs_checked = 0
    if anchors_path.exists():
        wanted = set(ids_by_frame)
        with anchors_path.open() as stream:
            for line in stream:
                anchor = json.loads(line)
                fid = anchor["carla_frame"]
                if fid not in wanted:
                    continue
                if set(map(int, anchor["futures_by_actor_id"])) != ids_by_frame[fid]:
                    raise ValueError("anchor ID mismatch in CARLA frame %s" % fid)
                anchor_refs_checked += 1
                if anchor_refs_checked == len(wanted):
                    break
    route_unchanged = route_bytes == (root / "route.json").read_bytes()
    if not route_unchanged:
        raise ValueError("source route changed during audit")
    valid_points = sum(row["geometry_valid_points"] for row in rows)
    supported_points = sum(row["lidar_supported_points"] for row in rows)
    total_points = sum(row["total_centerline_points"] for row in rows)
    valid_widths = sum(row["width_geometry_valid_roads"] for row in rows)
    supported_widths = sum(row["width_lidar_supported_roads"] for row in rows)
    total_roads = sum(row["road_count"] for row in rows)
    return {"source_episode": str(root), "source_schema": metadata["schema"],
            "source_route_sha256": hashlib.sha256(route_bytes).hexdigest(),
            "route_and_roadoptions_unchanged": route_unchanged,
            "frames": len(rows), "actor_ids_seen": len(types),
            "episode_actor_key": "(episode_id, actor_id)",
            "anchor_actor_id_refs_checked": anchor_refs_checked,
            "source_agent_instances": sum(row["source_agent_count"] for row in rows),
            "near_50m_agent_instances": sum(row["near_50m_agent_count"] for row in rows),
            "geometry_valid_centerline_points": valid_points,
            "lidar_supported_centerline_points": supported_points,
            "total_centerline_points": total_points,
            "centerline_geometry_coverage": valid_points / total_points if total_points else 0,
            "lidar_support_fraction_of_geometry": supported_points / valid_points if valid_points else 0,
            "width_geometry_valid_roads": valid_widths,
            "width_lidar_supported_roads": supported_widths,
            "total_roads": total_roads,
            "width_geometry_valid_rate": valid_widths / total_roads if total_roads else 0,
            "per_frame": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.episode, args.limit, args.start_index)
    formatted = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(formatted)
    print(formatted)


if __name__ == "__main__":
    main()
