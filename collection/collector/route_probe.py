"""Ask the installed Leaderboard whether a route contains a scenario trigger."""
from __future__ import print_function

import argparse
import json

import carla
from leaderboard.utils.route_manipulation import interpolate_trajectory
from leaderboard.utils.route_parser import RouteParser


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    configurations = RouteParser.parse_routes_file(args.route, args.scenarios)
    if len(configurations) != 1:
        raise ValueError("probe requires exactly one route")
    route = configurations[0]
    if world.get_map().name.split("/")[-1] != route.town:
        client.load_world(route.town)
        world = client.get_world()
    _, dense = interpolate_trajectory(world, route.trajectory)
    annotations = RouteParser.parse_annotations_file(args.scenarios)
    matches, _ = RouteParser.scan_route_for_scenarios(route.town, dense, annotations)
    print(json.dumps({"town": route.town, "route_points": len(dense),
                      "matched_triggers": len(matches),
                      "scenario_names": sorted(set(item["name"] for values in
                                                   matches.values() for item in values))}))


if __name__ == "__main__":
    main()
