"""Run one explicit external Leaderboard route into a fresh raw episode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


HERE = Path(__file__).resolve().parent
CARLA_ROOT = HERE.parents[2] / "Carla"


def selected_route(source, route_id, destination):
    root = ET.parse(str(source)).getroot()
    matches = [item for item in root.findall("route") if item.get("id") == str(route_id)]
    if len(matches) != 1:
        raise ValueError("route id %s occurs %d times" % (route_id, len(matches)))
    single = ET.Element("routes")
    single.append(matches[0])
    ET.ElementTree(single).write(str(destination), encoding="UTF-8", xml_declaration=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config(args):
    return {"fps": 10.0, "seed": args.seed, "target_speed_kmh": args.target_speed_kmh,
            "radar_range_m": 118.037109375,
            "vehicle_hazard_distance_m": args.vehicle_hazard_distance_m,
            "route_xml": str(args.routes.resolve()), "route_id": str(args.route_id),
            "scenario_json": str(args.scenarios.resolve()),
            "route_sha256": sha256(args.routes),
            "scenario_sha256": sha256(args.scenarios),
            "sensors": {
                "camera": {"x": 1.5, "y": 0.0, "z": 2.4, "roll": 0.0, "pitch": 0.0,
                           "yaw": 0.0, "width": 800, "height": 600, "fov": 90.0},
                "semantic_lidar": {"x": 1.5, "y": 0.0, "z": 2.4,
                                   "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                                   "channels": 64, "points_per_second": 1300000,
                                   "range_m": 118.037109375, "upper_fov": 10.0,
                                   "lower_fov": -30.0}}}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, required=True, help="External Leaderboard route XML")
    parser.add_argument("--route-id", required=True)
    parser.add_argument("--scenarios", type=Path, required=True,
                        help="External ScenarioRunner scenario JSON")
    parser.add_argument("--output", type=Path, required=True, help="New episode directory")
    parser.add_argument("--carla-python", type=Path, required=True)
    parser.add_argument("--carla-root", type=Path, default=CARLA_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-speed-kmh", type=float, default=20.0)
    parser.add_argument("--vehicle-hazard-distance-m", type=float, default=10.0)
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate files and route selection without starting CARLA")
    return parser.parse_args(argv)


def run(args):
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("episode already exists: %s" % output)
    for path in (args.routes, args.scenarios, args.carla_python):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.target_speed_kmh <= 0 or args.vehicle_hazard_distance_m <= 0:
        raise ValueError("target speed and hazard distance must be positive")
    json.loads(args.scenarios.read_text())
    with tempfile.TemporaryDirectory(prefix="raw_collection_") as temp:
        temp = Path(temp)
        route = temp / "selected.xml"
        selected_route(args.routes, args.route_id, route)
        settings = config(args)
        (temp / "agent.json").write_text(json.dumps(settings))
        if args.check_inputs:
            print(json.dumps({"status": "PASS", "route_id": args.route_id,
                              "route_sha256": settings["route_sha256"],
                              "scenario_sha256": settings["scenario_sha256"]}, indent=2))
            return
        carla_root = args.carla_root.resolve()
        paths = [carla_root / "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg",
                 carla_root / "PythonAPI/carla", carla_root / "C-Shenron/scenario_runner_0910",
                 carla_root / "C-Shenron/leaderboard"]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(path)
        checkpoint = output.with_name(output.name + ".leaderboard_result.json")
        if checkpoint.exists():
            raise FileExistsError(checkpoint)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(map(str, paths + [HERE]))
        environment["RADARVLA_EPISODE"] = str(output)
        environment["RADARVLA_TRAFFIC_EVENTS"] = str(output / "traffic_events.json")
        environment["DATAGEN"] = "1"
        # The local evaluator contains existing passive event reporting and
        # ScenarioManager remains its only world tick owner.
        script = ("from leaderboard.leaderboard_evaluator_local import "
                  "LeaderboardEvaluator, main; LeaderboardEvaluator.frame_rate=10.0; main()")
        command = [str(args.carla_python.resolve()), "-c", script,
                   "--host", args.host, "--port", str(args.port),
                   "--trafficManagerPort", str(args.tm_port),
                   "--trafficManagerSeed", str(args.seed),
                   "--routes", str(route), "--scenarios", str(args.scenarios.resolve()),
                   "--repetitions", "1", "--agent", str(HERE / "agent.py"),
                   "--agent-config", str(temp / "agent.json"),
                   "--track", "SENSORS", "--checkpoint", str(checkpoint)]
        print("COMMAND " + json.dumps(command), flush=True)
        subprocess.run(command, check=True, env=environment, cwd=str(paths[-1]))
        metadata = output / "metadata.json"
        if not metadata.exists() or json.loads(metadata.read_text()).get("status") != "complete":
            raise RuntimeError("capture did not finish a complete Leaderboard route")


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print("ERROR [raw-collection]: %s" % exc, file=sys.stderr)
        sys.exit(1)
