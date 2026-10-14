"""Select one Scenario2 trigger from an unchanged official Leaderboard route.

Run under the CARLA Python environment while a CARLA server is online. A
candidate is only a proposal; the batch smoke run must verify its real actor
placement and full scenario history before release.
"""
from __future__ import print_function

import argparse
import json
import math
import xml.etree.ElementTree as ET

import carla
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_parser import RouteParser
from srunner.tools.scenario_helper import get_waypoint_in_distance


def horizontal(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def candidate(world, route, min_before=40.0, min_after=70.0):
    road_map = world.get_map()
    _, dense = interpolate_trajectory(world, route.trajectory)
    progress = [0.0]
    for previous, current in zip(dense, dense[1:]):
        progress.append(progress[-1] + horizontal(previous[0].location,
                                                 current[0].location))
    rows = []
    for index, (pose, road_option) in enumerate(dense):
        along = progress[index]
        if along < min_before or progress[-1] - along < min_after:
            continue
        # A trigger on the first meter after a junction can map back to an
        # intersection in FollowLeadingVehicle's default get_waypoint() call.
        # The 25 m helper then returns immediately and restores the lead on Ego.
        previous_turn = next((j for j in range(index, -1, -1)
                              if dense[j][1].name != "LANEFOLLOW"), None)
        if previous_turn is not None and along - progress[previous_turn] < 12.0:
            continue
        waypoint = road_map.get_waypoint(pose.location, project_to_road=False)
        if waypoint is None or waypoint.is_intersection:
            continue
        if horizontal(waypoint.transform.location, pose.location) > 1.0:
            continue
        # Use exactly the projection mode that Scenario2 uses at runtime.
        scenario_waypoint = road_map.get_waypoint(pose.location)
        if (scenario_waypoint is None or scenario_waypoint.is_intersection or
                (scenario_waypoint.road_id, scenario_waypoint.lane_id) !=
                (waypoint.road_id, waypoint.lane_id)):
            continue
        forward, travelled = get_waypoint_in_distance(scenario_waypoint, 25.0)
        if not 24.0 <= travelled <= 27.0 or forward.is_intersection:
            continue
        if (waypoint.road_id, waypoint.lane_id) != (forward.road_id, forward.lane_id):
            continue
        nearby = [(j, p) for j, (p, _) in enumerate(dense)
                  if 23.0 <= progress[j] - along <= 27.0 and
                  horizontal(p.location, forward.transform.location) < 2.0]
        if not nearby:
            continue
        if any(dense[j][1].name != "LANEFOLLOW" for j in range(index,
                min(nearby[0][0] + 1, len(dense)))):
            continue
        # The lead must have room to travel toward the next junction and brake.
        cursor = forward
        remaining = 0.0
        while remaining < 35.0 and not cursor.is_intersection:
            nxt = cursor.next(1.0)
            if len(nxt) != 1:
                break
            remaining += horizontal(cursor.transform.location,
                                    nxt[0].transform.location)
            cursor = nxt[0]
        if remaining < 30.0:
            continue
        rows.append({"index": index, "route_progress_m": along,
                     "lead_forward_m": travelled,
                     "scenario_waypoint_road_id": scenario_waypoint.road_id,
                     "scenario_waypoint_lane_id": scenario_waypoint.lane_id,
                     "lead_to_junction_min_m": remaining,
                     "road_id": waypoint.road_id, "lane_id": waypoint.lane_id,
                     "transform": {"x": round(pose.location.x, 3),
                                   "y": round(pose.location.y, 3),
                                   "z": round(pose.location.z, 3),
                                   "yaw": round(pose.rotation.yaw, 3), "pitch": 0}})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True)
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15184)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    configurations = RouteParser.parse_routes_file(args.routes, "", args.route_id)
    if len(configurations) != 1:
        raise ValueError("exactly one official route is required")
    client = carla.Client(args.host, args.port)
    client.set_timeout(30)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != configurations[0].town:
        world = client.load_world(configurations[0].town)
    rows = candidate(world, configurations[0])
    if not rows:
        raise ValueError("no route-constrained 25 m Scenario2 trigger with straight clearance")
    # Prefer a middle-of-route trigger so pre- and post-event observations fit.
    chosen = min(rows, key=lambda row: abs(row["route_progress_m"] -
                                            rows[-1]["route_progress_m"] / 2))
    annotation = {"available_scenarios": [{configurations[0].town: [{
        "scenario_type": "Scenario2", "available_event_configurations": [
            {"transform": chosen["transform"]}]}]}]}
    with open(args.output, "w") as stream:
        json.dump({"route_id": args.route_id, "town": configurations[0].town,
                   "chosen": chosen, "candidate_count": len(rows),
                   "annotation": annotation}, stream, indent=2)
    print(json.dumps({"candidate_count": len(rows), "chosen": chosen}))


if __name__ == "__main__":
    main()
