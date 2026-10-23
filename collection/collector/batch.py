"""One-command smoke or resumable CARLA -> raw -> K-Radar collection."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

from batch_selection import CATEGORIES, choose_with_quotas, episode_candidates
from episode import validate as validate_episode

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
RADAR_MODULE = "e2e_agent_sem_lidar2shenron_package.carla_kradar_episode"
CODE_FILES = [HERE / name for name in ("batch.py", "batch_selection.py", "route_probe.py",
                                      "scenario2_candidate.py", "run.py", "agent.py",
                                      "episode.py", "road_map.py", "recorder.py", "agent_policy.py")]
CODE_FILES += [WORKSPACE / "Carla/C-Shenron/team_code/e2e_agent_sem_lidar2shenron_package" / name
               for name in ("carla_kradar_episode.py", "kradar_dense_unfold.py",
                            "kradar_dra_products.py", "lidar.py",
                            "shenron/heatmap_gen_fast.py")]


def load_manifest(path):
    data = path.read_text()
    try:
        import yaml
    except ImportError:
        value = json.loads(data)  # JSON is valid YAML; supports the stdlib launcher.
    else:
        value = yaml.safe_load(data)
    if value.get("schema") != "radarvla_batch_v1":
        raise ValueError("unknown batch manifest schema")
    return value


def digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def signature(manifest_path):
    config = load_manifest(manifest_path)
    inputs = {resolve(row["xml"]) for row in config["routes"]}
    inputs.update(resolve(value) for value in config["scenarios"].values())
    inputs.add(resolve(config["scenario2"]["route_xml"]))
    return digest([manifest_path] + CODE_FILES + sorted(inputs))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".inprogress")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)


def atomic_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".inprogress")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    tmp.replace(path)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else (WORKSPACE / path).resolve()


def frame_quota(config):
    quotas = config["quotas"]
    categories = quotas["categories"]
    splits = quotas["splits"]
    weathers = quotas["weather"]
    total = int(quotas["total"])
    if set(categories) != {"normal", *CATEGORIES}:
        raise ValueError("category quotas do not cover the collection classes")
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("split quotas must be train/val/test")
    if set(weathers) != set(config["weather"]):
        raise ValueError("weather quotas and profiles differ")
    if any(sum(int(x) for x in group.values()) != total for group in
           (categories, splits, weathers)):
        raise ValueError("all quota dimensions must sum to total")
    joint = quotas.get("category_weather")
    if joint is not None:
        expected = {c + '/' + w for c in categories for w in weathers}
        if set(joint) != expected or any(type(v) is not int or v < 0 for v in joint.values()):
            raise ValueError('invalid joint quotas')
        if any(sum(joint[c+'/'+w] for w in weathers) != categories[c] for c in categories) or any(sum(joint[c+'/'+w] for c in categories) != weathers[w] for w in weathers):
            raise ValueError('joint quotas disagree with marginal quotas')
    return categories, splits, weathers, total


def verify_scenario2(config):
    scene = config["scenario2"]
    if not all(scene.get(key) is not None for key in
               ("annotation_json", "pilot_report", "route_id")):
        raise ValueError("Scenario2 route ID, annotation and real pilot report are required")
    annotation, report = resolve(scene["annotation_json"]), resolve(scene["pilot_report"])
    if not annotation.is_file() or not report.is_file():
        raise FileNotFoundError("Scenario2 annotation/pilot report missing")
    pilot = json.loads(report.read_text())
    required = ("actor_restored", "scenario_activated", "following_started",
                "braking_started", "scenario_completed")
    if pilot.get("scenario") != "Scenario2" or not all(
            pilot.get("events", {}).get(key) for key in required):
        raise ValueError("Scenario2 pilot lacks real event evidence")
    if pilot.get("annotation_sha256") != hashlib.sha256(annotation.read_bytes()).hexdigest():
        raise ValueError("Scenario2 pilot does not certify this annotation")
    if str(pilot.get("route_id")) != str(scene["route_id"]):
        raise ValueError("Scenario2 pilot does not certify this route")
    return annotation


def preflight(config, manifest, root, mode, skip_smoke=False):
    frame_quota(config)
    maps = WORKSPACE / "Carla/CarlaUE4/Content/Carla/Maps"
    required_towns = {item["town"] for item in config["routes"]}
    required_towns.add(config["scenario2"]["town"])
    missing_maps = sorted(town for town in required_towns
                          if not (maps / (town + ".umap")).is_file())
    if missing_maps:
        raise FileNotFoundError("CARLA maps are not installed: " + ", ".join(missing_maps))
    for name in ("carla_python", "radar_python", "carla_server"):
        if not resolve(config["runtime"][name]).is_file():
            raise FileNotFoundError(config["runtime"][name])
    for item in config["routes"]:
        if item["split"] not in ("train", "val", "test"):
            raise ValueError("invalid route split")
        if not resolve(item["xml"]).is_file():
            raise FileNotFoundError(item["xml"])
    for name in ("normal", "official"):
        if not resolve(config["scenarios"][name]).is_file():
            raise FileNotFoundError(config["scenarios"][name])
    if mode == "full":
        certificate = root / "qualification" / "scenario2_pilot.json"
        if not certificate.is_file():
            raise FileNotFoundError("Scenario2 pilot certificate is missing")
        pilot = json.loads(certificate.read_text())
        config["scenario2"].update({"route_id": pilot.get("route_id"),
                                    "annotation_json": pilot.get("annotation_json"),
                                    "pilot_report": str(certificate)})
        verify_scenario2(config)
    if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
        raise PermissionError("data root must exist and be writable: " + str(root))
    available = shutil.disk_usage(root).free
    minimum = int(config["runtime"]["minimum_free_gb"]) * 1024 ** 3
    if available < minimum:
        raise OSError("insufficient storage: %.1f GiB free, %.1f GiB required" %
                      (available / 1024 ** 3, minimum / 1024 ** 3))
    stamp = signature(manifest)
    if mode == "full" and not skip_smoke:
        release = root / "qualification" / "smoke_release.json"
        if not release.is_file() or \
                json.loads(release.read_text()).get("signature") != stamp or \
                json.loads(release.read_text()).get("status") != "PASS":
            raise ValueError("matching PASS smoke release is required before full collection")
    return stamp


def route_entries(item):
    tree = ET.parse(str(resolve(item["xml"])))
    routes = tree.getroot().findall("route")
    start = int(item.get("start_index", 0))
    limit = int(item.get("max_routes", len(routes) - start))
    if start < 0 or limit < 1 or start + limit > len(routes):
        raise ValueError("route slice is outside official XML")
    return [{"id": route.get("id")} for route in routes[start:start + limit]]


def prepare_route(source, route_id, weather, destination):
    matches = [route for route in ET.parse(str(source)).getroot().findall("route")
               if route.get("id") == str(route_id)]
    if len(matches) != 1:
        raise ValueError("route id is not unique")
    original = matches[0]
    before = [dict(point.attrib) for point in original.findall("waypoint")]
    for node in list(original.findall("weather")):
        original.remove(node)
    ET.SubElement(original, "weather", {key: str(value) for key, value in weather.items()})
    if [dict(point.attrib) for point in original.findall("waypoint")] != before:
        raise ValueError("weather edit changed official waypoints")
    root = ET.Element("routes")
    root.append(original)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(destination), encoding="utf-8", xml_declaration=True)


def prepare_scenarios(source, category, destination):
    data = json.loads(source.read_text())
    if category != "normal":
        wanted = CATEGORIES[category]
        data["available_scenarios"] = [{town: [
            entry for entry in scenarios if entry["scenario_type"] in wanted]}
            for block in data["available_scenarios"] for town, scenarios in block.items()]
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, data)
    return sum(len(entry.get("available_event_configurations", []))
               for block in data["available_scenarios"]
               for entries in block.values() for entry in entries)


def jobs(config, mode):
    if mode == "smoke":
        for row in config["smoke_jobs"]:
            if row["category"] == "scenario2" and row["route_id"] is None:
                yield dict(row, route_id=str(config["scenario2"]["route_id"]))
            else:
                yield row
        return
    profiles = list(config["weather"])
    scene = config["scenario2"]
    for category in ("scenario2", "scenario3", "scenario4", "scenario7_9",
                     "scenario10", "normal"):
        if config["quotas"]["categories"][category] == 0:
            continue
        if category == "scenario2" and scene["route_id"] is None:
            continue  # preflight reports the missing certified input explicitly.
        route_sets = ([{"town": scene["town"], "xml": scene["route_xml"],
                        "split": "train", "route_id": scene["route_id"]}]
                      if category == "scenario2" else config["routes"])
        for repetition in range(int(config["runtime"]["repetitions"])):
            for route_set in route_sets:
                entries = ([{"id": str(route_set["route_id"])}] if category == "scenario2"
                           else route_entries(route_set))
                for offset, route in enumerate(entries):
                    weather = profiles[(offset + repetition + int(route["id"])) % len(profiles)]
                    yield {"category": category, "town": route_set["town"],
                           "route_xml": route_set["xml"], "route_id": str(route["id"]),
                           "split": route_set["split"], "weather": weather,
                           "seed": repetition}


def job_key(job):
    payload = json.dumps(job, sort_keys=True)
    prefix = "%s_%s_%s_%s" % (job["category"], job["town"],
                              job["route_id"], job["weather"])
    return prefix + "_" + hashlib.sha256(payload.encode()).hexdigest()[:10]


def port_open(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


def server_ready(config):
    runtime = config["runtime"]
    if not port_open(runtime["host"], int(runtime["port"])):
        return False
    env = os.environ.copy()
    env["PYTHONPATH"] = str(WORKSPACE / "Carla/PythonAPI/carla/dist/"
                            "carla-0.9.10-py3.7-linux-x86_64.egg")
    code = ("import carla; c=carla.Client(%r,%d); c.set_timeout(5); "
            "print(c.get_world().get_map().name)" %
            (runtime["host"], int(runtime["port"])))
    try:
        result = subprocess.run([str(resolve(runtime["carla_python"])), "-c", code],
                                env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class CarlaServer:
    def __init__(self, config, output):
        self.config, self.output, self.process = config, output, None
        self.display_process = None

    def display_environment(self, runtime):
        env = os.environ.copy()
        if not runtime.get('xvfb'):
            return env
        import secrets
        number = next((i for i in range(15184, 15284)
                       if not Path('/tmp/.X11-unix/X%d' % i).exists()
                       and not Path('/tmp/.X%d-lock' % i).exists()), None)
        if number is None:
            raise RuntimeError('No unused virtual display number')
        display = ':%d' % number
        auth = self.output/'.Xauthority'
        auth.touch(mode=0o600, exist_ok=True)
        auth.chmod(0o600)
        subprocess.run(['xauth', '-f', str(auth), 'add', display,
                        'MIT-MAGIC-COOKIE-1', secrets.token_hex(16)], check=True)
        env.update(DISPLAY=display, XAUTHORITY=str(auth),
                   XDG_RUNTIME_DIR='/run/user/'+str(os.getuid()))
        if runtime.get('vulkan_icd'):
            env['VK_ICD_FILENAMES'] = str(runtime['vulkan_icd'])
        with (self.output/'xvfb.log').open('a') as log:
            self.display_process = subprocess.Popen([str(runtime['xvfb']), display,
                '-screen', '0', '1024x768x24', '-nolisten', 'tcp', '-auth', str(auth)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(30):
            if self.display_process.poll() is not None:
                raise RuntimeError('Virtual display exited; see xvfb.log')
            if subprocess.run(['xdpyinfo'], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                return env
            time.sleep(.2)
        raise RuntimeError('Virtual display not ready')

    def __enter__(self):
        runtime = self.config["runtime"]
        host, port = runtime["host"], int(runtime["port"])
        if port_open(host, port):
            raise RuntimeError("CARLA port already occupied; refusing to control another server")
        quality = runtime.get('quality_level', 'Low')
        backend = runtime.get('graphics_api')
        if quality not in ('Low', 'Epic') or backend not in (None, 'vulkan', 'opengl'):
            raise ValueError('Unknown CARLA rendering configuration')
        rendering = ['-windowed', '-ResX=800', '-ResY=600'] if runtime.get('xvfb') else ['-RenderOffScreen']
        command = [str(resolve(runtime["carla_server"])), "CarlaUE4", *rendering,
                   "-nosound", "-quality-level=" + quality, "-carla-rpc-port=" + str(port)]
        if backend:
            command.append('-' + backend)
        if 'carla_device' in runtime:
            command.append('-graphicsadapter=' + str(runtime['carla_device']))
        try:
            env = self.display_environment(runtime)
            with (self.output / 'carla_server.log').open('a') as log:
                self.process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                env=env, start_new_session=True)
            deadline = time.monotonic() + int(runtime.get("server_timeout_s", 120))
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("CARLA exited during startup; see carla_server.log")
                if server_ready(self.config):
                    libraries = Path('/proc/%d/maps' % self.process.pid).read_text()
                    if backend == 'vulkan' and 'libvulkan' not in libraries:
                        raise RuntimeError('CARLA did not load requested Vulkan backend')
                    atomic_json(self.output/'rendering.json', {'quality_level':quality,
                        'requested_graphics_api':backend, 'vulkan_loaded':'libvulkan' in libraries,
                        'virtual_display':bool(runtime.get('xvfb')),
                        'xvfb_sha256':digest([Path(runtime['xvfb'])]) if runtime.get('xvfb') else None,
                        'command':command, 'pid':self.process.pid})
                    return self
                time.sleep(2)
            raise TimeoutError("CARLA did not open its RPC port")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        for process in (self.process, self.display_process):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def command_env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(WORKSPACE / "Carla/C-Shenron/team_code"),
                                         str(HERE), env.get("PYTHONPATH", "")))
    return env


def finalized_episode(episode):
    """A route may end early; only a finalized, internally valid capture is usable."""
    metadata_path = episode / "metadata.json"
    if not metadata_path.is_file() or not (episode / "anchors.jsonl").is_file():
        return False
    return json.loads(metadata_path.read_text()).get("status") in ("complete", "failed", "partial")


def run_logged(command, logfile, env=None, cwd=None, progress=None):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as stream:
        stream.write("COMMAND " + json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   env=env, cwd=cwd, start_new_session=True)
        try:
            while process.poll() is None:
                if progress:
                    progress()
                time.sleep(2)
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, command)


def probe_route(config, route_path, scenario_path, job, folder):
    runtime = config["runtime"]
    carla_root = WORKSPACE / "Carla"
    paths = [carla_root / "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg",
             carla_root / "PythonAPI/carla", carla_root / "C-Shenron/scenario_runner_0910",
             carla_root / "C-Shenron/leaderboard", HERE]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
    command = [str(resolve(runtime["carla_python"])), str(HERE / "route_probe.py"),
               "--route", str(route_path), "--scenarios", str(scenario_path),
               "--host", runtime["host"], "--port", str(runtime["port"])]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    (folder / "route_probe.log").write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError("route probe failed: " + str(folder / "route_probe.log"))
    return json.loads(result.stdout.strip().splitlines()[-1])


def scenario2_pilot_evidence(episode, annotation, route_id):
    """Certify a generated trigger using measured actor placement and event history."""
    from batch_selection import distance_xy, read_jsonl, speed
    episode = Path(episode)
    events = list(read_jsonl(episode / "events.jsonl"))
    chosen = next((item for item in events if item.get("event") == "scenario_selected" and
                   item.get("scenario_name") == "Scenario2"), None)
    if chosen is None or not chosen.get("other_actor_ids"):
        raise ValueError("Scenario2 was not selected with a lead actor")
    index = chosen["index"]
    stages = ("actor_restored", "scenario_activated", "following_started",
              "braking_started", "scenario_completed")
    history = {stage: [item for item in events if item.get("index") == index and
               item.get("event") == stage and item.get("carla_frame") is not None]
               for stage in stages}
    if any(not rows for rows in history.values()):
        raise ValueError("Scenario2 did not complete all stages")
    stage_frame = {stage: min(row["carla_frame"] for row in history[stage])
                   for stage in stages}
    # The ScenarioManager can record activation/following on the tick before
    # the actor's restored surface pose first appears in a world snapshot.
    if not (stage_frame["scenario_activated"] <= stage_frame["following_started"] <
            stage_frame["braking_started"] <= stage_frame["scenario_completed"] and
            stage_frame["scenario_activated"] <= stage_frame["actor_restored"] <=
            stage_frame["braking_started"]):
        raise ValueError("Scenario2 stage history is out of order")
    restored = min(row["carla_frame"] for row in history["actor_restored"])
    completed = min(row["carla_frame"] for row in history["scenario_completed"])
    lead_id = int(chosen["other_actor_ids"][0])
    frames = [json.loads(path.read_text()) for path in sorted(
        (episode / "frames").glob("[0-9]*/frame.json"))]
    first = next((row for row in frames if row["carla_frame"] >= restored and
                  any(actor["actor_id"] == lead_id and
                      actor["pose"]["position_m"][2] > -100 for actor in row["actors"])), None)
    if first is None:
        raise ValueError("lead actor absent after restoration")
    lead = next(actor for actor in first["actors"] if actor["actor_id"] == lead_id)
    center = distance_xy(first["ego"]["pose"]["position_m"],
                         lead["pose"]["position_m"])
    ego_position = first["ego"]["pose"]["position_m"]
    lead_position = lead["pose"]["position_m"]
    yaw = math.radians(first["ego"]["pose"]["rotation_deg"][2])
    longitudinal = ((lead_position[0] - ego_position[0]) * math.cos(yaw) +
                    (lead_position[1] - ego_position[1]) * math.sin(yaw))
    lateral = abs(-(lead_position[0] - ego_position[0]) * math.sin(yaw) +
                  (lead_position[1] - ego_position[1]) * math.cos(yaw))
    gap = center - float(first["ego"]["bbox"]["extent_m"][0]) - \
        float(lead["bbox"]["extent_m"][0])
    if not 20.0 <= longitudinal <= 32.0 or lateral > 3.0 or gap < 14.0:
        raise ValueError("unsafe measured lead placement: ahead %.1f m, lateral %.1f m, bbox gap %.1f m" %
                         (longitudinal, lateral, gap))
    interval = [row for row in frames if restored <= row["carla_frame"] <= completed]
    trace = [(row, next((actor for actor in row["actors"]
                         if actor["actor_id"] == lead_id), None)) for row in interval]
    visible = [(row, actor) for row, actor in trace if actor is not None]
    moved = max((distance_xy(actor["pose"]["position_m"],
                             lead["pose"]["position_m"]) for _, actor in visible), default=0)
    braked = any(actor is not None and
                 distance_xy(row["ego"]["pose"]["position_m"],
                             actor["pose"]["position_m"]) <= 50 and
                 float((row["ego"].get("commanded_control") or {}).get("brake", 0)) > .05
                 for row, actor in trace)
    if moved < 2.0 or not braked or not any(speed(actor) >= .5 for _, actor in visible):
        raise ValueError("lead motion or ego braking missing in Scenario2 pilot")
    full_future_anchors = []
    for anchor in read_jsonl(episode / "anchors.jsonl"):
        fid = anchor["carla_frame"]
        if not restored <= fid <= completed - 30:
            continue
        futures = anchor.get("futures_by_actor_id", {})
        ego_id = str(first["ego"]["actor_id"])
        if all(futures.get(ego_id, {}).get("valid", [])) and \
                all(futures.get(str(lead_id), {}).get("valid", [])) and \
                len(futures.get(str(lead_id), {}).get("valid", [])) == 30:
            full_future_anchors.append(fid)
    if not full_future_anchors:
        raise ValueError("Scenario2 has no joint Ego/lead future-30 anchor")
    checkpoint = episode.with_name(episode.name + ".leaderboard_result.json")
    if not checkpoint.is_file():
        raise ValueError("Scenario2 pilot Leaderboard result missing")
    records = json.loads(checkpoint.read_text())["_checkpoint"]["records"]
    if len(records) != 1:
        raise ValueError("Scenario2 pilot has no unique Leaderboard result")
    # Leaderboard infractions may occur after the completed interaction. Reject
    # collisions involving the lead or spatially overlapping the event trace.
    collisions = [str(item) for name in ("collisions_vehicle", "collisions_pedestrian",
                                           "collisions_layout")
                  for item in records[0]["infractions"].get(name, [])]
    event_ego_positions = [row["ego"]["pose"]["position_m"] for row in interval]
    for item in collisions:
        match_id = re.search(r"\bid=(\d+)\b", item)
        if match_id and int(match_id.group(1)) == lead_id:
            raise ValueError("Scenario2 lead collision: " + item)
        match_xy = re.search(r"\bx=([-+\d.]+), y=([-+\d.]+)", item)
        if match_xy is None:
            raise ValueError("collision time/location cannot be isolated: " + item)
        position = (float(match_xy.group(1)), float(match_xy.group(2)))
        if any(distance_xy(position, ego_position) < 6.0
               for ego_position in event_ego_positions):
            raise ValueError("collision overlaps Scenario2 event window: " + item)
    return {"scenario": "Scenario2", "route_id": str(route_id),
            "annotation_json": str(annotation),
            "annotation_sha256": hashlib.sha256(annotation.read_bytes()).hexdigest(),
            "events": {key: min(row["carla_frame"] for row in value)
                       for key, value in history.items()},
            "lead_actor_id": lead_id, "restored_center_distance_m": center,
            "restored_longitudinal_distance_m": longitudinal,
            "restored_lateral_distance_m": lateral,
            "restored_bbox_gap_m": gap, "lead_displacement_m": moved,
            "ego_braked": braked, "joint_future_30_anchors": len(full_future_anchors),
            "first_joint_future_anchor": full_future_anchors[0],
            "route_status": records[0]["status"],
            "route_completion_percent": records[0].get("scores", {}).get("score_route"),
            "outside_event_collisions": collisions, "episode": str(episode)}


def prepare_scenario2_for_smoke(config, output, root):
    """One route-constrained candidate and one real pilot; no automatic retries."""
    certificate = root / "qualification" / "scenario2_pilot.json"
    if certificate.is_file():
        pilot = json.loads(certificate.read_text())
        config["scenario2"].update({"route_id": pilot["route_id"],
                                    "annotation_json": pilot["annotation_json"],
                                    "pilot_report": str(certificate)})
        verify_scenario2(config)
        return
    scene = config["scenario2"]
    folder = output / "scenario2_pilot"
    folder.mkdir(parents=True, exist_ok=True)
    route_source = resolve(scene["route_xml"])
    candidate_file = folder / "candidate.json"
    if not candidate_file.is_file():
        carla_root = WORKSPACE / "Carla"
        paths = [carla_root / "PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg",
                 carla_root / "PythonAPI/carla", carla_root / "C-Shenron/scenario_runner_0910",
                 carla_root / "C-Shenron/leaderboard", HERE]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
        for route in route_entries({"xml": scene["route_xml"],
                                    "max_routes": scene["max_candidate_routes"]}):
            command = [str(resolve(config["runtime"]["carla_python"])),
                       str(HERE / "scenario2_candidate.py"), "--routes", str(route_source),
                       "--route-id", str(route["id"]), "--host", config["runtime"]["host"],
                       "--port", str(config["runtime"]["port"]),
                       "--output", str(candidate_file)]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            with (folder / "candidate_search.log").open("a") as stream:
                stream.write("ROUTE %s\n%s%s\n" %
                             (route["id"], result.stdout, result.stderr))
            if result.returncode == 0:
                break
        if not candidate_file.is_file():
            raise ValueError("no safe route-constrained Scenario2 candidate; see candidate_search.log")
    candidate = json.loads(candidate_file.read_text())
    annotation = folder / "scenario2_annotation.json"
    atomic_json(annotation, candidate["annotation"])
    route = folder / "route.xml"
    prepare_route(route_source, candidate["route_id"],
                  config["weather"][scene.get("pilot_weather", "clear_day")], route)
    pilot_episode = folder / "episode"
    if pilot_episode.exists() and not (pilot_episode / "metadata.json").is_file():
        raise RuntimeError("incomplete Scenario2 pilot preserved; no retry")
    if not pilot_episode.exists():
        runtime = config["runtime"]
        command = [str(resolve(runtime["carla_python"])), str(HERE / "run.py"),
                   "--routes", str(route), "--route-id", str(candidate["route_id"]),
                   "--scenarios", str(annotation), "--output", str(pilot_episode),
                   "--carla-python", str(resolve(runtime["carla_python"])),
                   "--carla-root", str(WORKSPACE / "Carla"),
                   "--host", runtime["host"], "--port", str(runtime["port"]),
                   "--tm-port", str(runtime["tm_port"])]
        try:
            run_logged(command, folder / "pilot_collect.log")
        except subprocess.CalledProcessError:
            # A route may fail after a complete, usable event. The finalized
            # episode and event-local evidence below still have to validate.
            if not (pilot_episode / "metadata.json").is_file():
                raise
    validate_episode(pilot_episode, require_complete=False)
    pilot = scenario2_pilot_evidence(pilot_episode, annotation, candidate["route_id"])
    atomic_json(certificate, pilot)
    scene.update({"route_id": pilot["route_id"], "annotation_json": str(annotation),
                  "pilot_report": str(certificate)})
    verify_scenario2(config)


def accepted_from_jobs(root):
    """Read only existing selections; the manifest can contain many unused jobs."""
    accepted = []
    for selection in sorted((root / "jobs").glob("*/selection.json")):
        accepted.extend(json.loads(selection.read_text())["accepted"])
    keys = {(row["episode_id"], row["carla_frame"]) for row in accepted}
    if len(keys) != len(accepted):
        raise ValueError("resumed selections contain duplicate episode/frame anchors")
    return accepted


def deficits(accepted, config):
    categories, splits, weathers, total = frame_quota(config)
    groups = {"categories": categories, "splits": splits, "weather": weathers}
    fields = {"categories": "category", "splits": "split", "weather": "weather"}
    return {name: {key: max(0, int(target) - sum(row[fields[name]] == key for row in accepted))
                   for key, target in quota.items()} for name, quota in groups.items()} | {
                       "total": max(0, total - len(accepted))}


def radar_windows(accepted, halo=2, core_limit=48):
    """Group selected indices; include measured temporal neighbors as halos."""
    by_episode = {}
    for row in accepted:
        by_episode.setdefault(row["job_id"], []).append(row)
    for job_id, rows in by_episode.items():
        rows.sort(key=lambda row: row["source_index"])
        group = []
        for row in rows:
            if group and (row["source_index"] - group[-1]["source_index"] > 1 or
                          len(group) >= core_limit):
                yield job_id, group, halo
                group = []
            group.append(row)
        if group:
            yield job_id, group, halo


def _process_radar_window(config, output, job_id, core, halo, device):
    runtime = config["runtime"]
    jobdir = output / "jobs" / job_id
    episode = jobdir / "episode"
    metadata = json.loads((episode / "metadata.json").read_text())
    count = int(metadata["captured_frames"])
    start = max(0, core[0]["source_index"] - halo)
    end = min(count - 1, core[-1]["source_index"] + halo)
    name = "%06d_%06d" % (core[0]["source_index"], core[-1]["source_index"])
    base = jobdir / "radar" / name
    base.mkdir(parents=True, exist_ok=True)
    completed = sorted(base.glob("attempt_*/complete.json"))
    try:
        if completed:
            product = completed[0].parent
            marker = json.loads(completed[0].read_text())
            if (marker.get("source_start_index") != start or
                marker.get("source_end_index") != end or
                marker.get("core_frames") != [row["carla_frame"] for row in core]):
                raise ValueError("completed radar window does not match accepted anchors")
        else:
            attempt = len(list(base.glob("attempt_*"))) + 1
            product = base / ("attempt_%03d" % attempt)
            env = command_env()
            if device is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(device)
            command = [str(resolve(runtime["radar_python"])), "-m", RADAR_MODULE,
                       str(episode), "--output", str(product), "--start-index", str(start),
                       "--limit", str(end - start + 1)]
            run_logged(command, base / "radar.log", env=env)
            run_logged([str(resolve(runtime["radar_python"])), "-m", RADAR_MODULE,
                        str(episode), "--output", str(product), "--validate"],
                       base / "validate.log", env=env)
            atomic_json(product / "complete.json", {"source_start_index": start,
                                                     "source_end_index": end,
                                                     "core_frames": [row["carla_frame"] for row in core],
                                                     "cuda_visible_devices": device})
        return ([{**row, "radar_product": str(product),
                  "radar_frame_index": row["source_index"] - start} for row in core], None)
    except Exception as exc:
        failure = {"job_id": job_id, "window": name,
                   "error": "%s: %s" % (type(exc).__name__, exc)}
        atomic_json(base / "failure.json", failure)
        print("FAILED radar %s/%s: %s" % (job_id, name, exc),
              file=sys.stderr, flush=True)
        return [], failure


def process_radar(config, output, accepted):
    runtime = config["runtime"]
    windows = list(radar_windows(accepted, core_limit=int(runtime["radar_chunk_frames"])))
    devices = runtime.get("radar_devices") or [None]
    workers = int(runtime.get("radar_workers", 1))
    if workers < 1 or workers > len(devices):
        raise ValueError("radar_workers must be between 1 and number of radar_devices")
    started, progress, total = time.monotonic(), 0, len(accepted)
    index_rows, failures = [], []
    bar = tqdm(total=total, desc="radar frames", unit="frame") if tqdm else None
    try:
        # One queue per physical GPU prevents two windows from racing onto the
        # same device when adjacent windows take different amounts of time.
        with ExitStack() as stack:
            pools = [stack.enter_context(ThreadPoolExecutor(max_workers=1))
                     for _ in range(workers)]
            futures = {pools[index % workers].submit(
                           _process_radar_window, config, output, job_id, core,
                           halo, devices[index % workers]): len(core)
                       for index, (job_id, core, halo) in enumerate(windows)}
            for future in as_completed(futures):
                rows, failure = future.result()
                index_rows.extend(rows)
                if failure:
                    failures.append(failure)
                progress += futures[future]
                elapsed = max(time.monotonic() - started, 0.001)
                eta = elapsed / max(progress, 1) * (total - progress)
                if bar is not None:
                    bar.update(futures[future])
                    bar.set_postfix_str("ETA %.1f min" % (eta / 60))
                else:
                    print("radar %d/%d frames | %.1f frame/s | ETA %.1f min" %
                          (progress, total, progress / elapsed, eta / 60), flush=True)
    finally:
        if bar is not None:
            bar.close()
    order = {(row["episode_id"], row["carla_frame"]): index for index, row in enumerate(accepted)}
    index_rows.sort(key=lambda row: order[(row["episode_id"], row["carla_frame"])])
    atomic_jsonl(output / "dataset_index.jsonl", index_rows)
    return index_rows, failures


def collect(config, mode, output):
    runtime = config["runtime"]
    ordered = list(jobs(config, mode))
    accepted = accepted_from_jobs(output)
    categories, splits, weathers, target = frame_quota(config)
    counts = {field: {} for field in ("category", "split", "weather")}
    for row in accepted:
        for field in counts:
            value = row[field]
            counts[field][value] = counts[field].get(value, 0) + 1
    started = time.monotonic()
    failures = [json.loads(path.read_text()) for job in ordered
                for path in [output / "jobs" / job_key(job) / "failure.json"] if path.is_file()]
    bar = tqdm(total=target if mode == "full" else
               int(config["smoke_frames_per_job"]) * len(ordered),
               initial=len(accepted), desc="accepted anchors", unit="frame") if tqdm else None
    try:
      with CarlaServer(config, output):
        for number, job in enumerate(ordered, 1):
            if mode == "full" and len(accepted) >= target:
                break
            if mode == "full" and counts["category"].get(job["category"], 0) >= categories[job["category"]]:
                continue
            if mode == "full" and counts["split"].get(job["split"], 0) >= splits[job["split"]]:
                continue
            if mode == "full" and counts["weather"].get(job["weather"], 0) >= weathers[job["weather"]]:
                continue
            key = job_key(job)
            folder = output / "jobs" / key
            if (folder / "selection.json").is_file() or (folder / "failure.json").is_file():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            print("episode %d/%d %s | accepted %d/%d" %
                  (number, len(ordered), key, len(accepted), target), flush=True)
            try:
                if job["route_id"] == "auto":
                    # Try distinct official routes, never retry the same route.
                    chosen = None
                    scene_source = resolve(config["scenarios"]["official"])
                    scene_path = folder / "scenarios.json"
                    prepare_scenarios(scene_source, job["category"], scene_path)
                    for route in route_entries(next(item for item in config["routes"]
                                                    if item["town"] == job["town"])):
                        candidate = folder / ("probe_%s.xml" % route["id"])
                        prepare_route(resolve(job["route_xml"]), route["id"],
                                      config["weather"][job["weather"]], candidate)
                        probe = probe_route(config, candidate, scene_path, job, folder)
                        if probe["matched_triggers"]:
                            chosen = str(route["id"])
                            break
                    if chosen is None:
                        atomic_json(folder / "selection.json", {"accepted": [],
                                                                 "reason": "no matching smoke route"})
                        continue
                    job = dict(job, route_id=chosen)
                if job["category"] == "scenario2":
                    scene_source = verify_scenario2(config)
                else:
                    scene_source = resolve(config["scenarios"][
                        "normal" if job["category"] == "normal" else "official"])
                route_path = folder / "route.xml"
                prepare_route(resolve(job["route_xml"]), job["route_id"],
                              config["weather"][job["weather"]], route_path)
                scene_path = folder / "scenarios.json"
                count = prepare_scenarios(scene_source, job["category"], scene_path)
                if job["category"] != "normal" and not count:
                    atomic_json(folder / "selection.json", {"accepted": [], "reason": "no annotation"})
                    continue
                if job["category"] != "normal":
                    probe = probe_route(config, route_path, scene_path, job, folder)
                    if not probe["matched_triggers"]:
                        atomic_json(folder / "selection.json", {"accepted": [], "reason": "no route trigger"})
                        continue
                episode = folder / "episode"
                if not finalized_episode(episode):
                    if episode.exists():
                        raise RuntimeError("incomplete episode preserved; no automatic retry")
                    command = [str(resolve(runtime["carla_python"])), str(HERE / "run.py"),
                               "--routes", str(route_path), "--route-id", str(job["route_id"]),
                               "--scenarios", str(scene_path), "--output", str(episode),
                               "--carla-python", str(resolve(runtime["carla_python"])),
                               "--carla-root", str(WORKSPACE / "Carla"),
                               "--host", runtime["host"], "--port", str(runtime["port"]),
                               "--tm-port", str(runtime["tm_port"]), "--seed", str(job["seed"])]
                    episode_bar = tqdm(desc="episode " + key, unit="frame") if tqdm else None
                    seen = [0]
                    def episode_progress():
                        frames = episode / "frames"
                        count = (sum(path.is_dir() and path.name.isdigit()
                                     for path in frames.iterdir()) if frames.is_dir() else 0)
                        if episode_bar is not None:
                            episode_bar.update(max(0, count - seen[0]))
                        elif count > seen[0]:
                            print("episode %s: %d captured frames" % (key, count), flush=True)
                        seen[0] = count
                    try:
                        try:
                            run_logged(command, folder / "collect.log", progress=episode_progress)
                        except subprocess.CalledProcessError:
                            if not finalized_episode(episode):
                                raise
                        episode_progress()
                    finally:
                        if episode_bar is not None:
                            episode_bar.close()
                # A route timeout or later unrelated collision does not erase valid
                # earlier anchors. Validate the complete recorded prefix instead.
                validation = validate_episode(episode, require_complete=False)
                if mode == "smoke":
                    atomic_json(folder / "episode_validation.json", validation)
                candidates, evidence = episode_candidates(episode, job["category"],
                                                          job["weather"], job["split"])
                if mode == "smoke":
                    candidates = candidates[:int(config["smoke_frames_per_job"])]
                picked = choose_with_quotas(candidates, accepted, categories, splits, weathers, config["quotas"].get("category_weather"))
                picked = [dict(row, job_id=key) for row in picked]
                atomic_json(folder / "selection.json", {"job": job, "accepted": picked,
                                                         "evidence": evidence,
                                                         "route_status": validation["status"]})
                accepted.extend(picked)
                for row in picked:
                    for field in counts:
                        value = row[field]
                        counts[field][value] = counts[field].get(value, 0) + 1
                atomic_jsonl(output / "accepted_anchors.jsonl", accepted)
                elapsed = max(time.monotonic() - started, 0.001)
                eta = elapsed / max(len(accepted), 1) * max(target - len(accepted), 0)
                if bar is not None:
                    bar.update(len(picked))
                    bar.set_postfix_str("estimated ETA %.1f h; yield-dependent" % (eta / 3600))
                else:
                    print("accepted %d/%d | ETA ~%.1f h (yield-dependent)" %
                          (len(accepted), target, eta / 3600), flush=True)
            except Exception as exc:
                failure = {"job_id": key, "error": "%s: %s" % (type(exc).__name__, exc)}
                failures.append(failure)
                atomic_json(folder / "failure.json", failure)
                print("FAILED %s: %s" % (key, exc), file=sys.stderr, flush=True)
                if mode == "smoke":
                    break
    finally:
        if bar is not None:
            bar.close()
    return accepted, failures


def smoke_dynamic_audit(output, products):
    """Independent CARLA scatterer oracle on a few event frames, for audit only."""
    sys.path.insert(0, str(WORKSPACE / "Carla/C-Shenron/team_code"))
    import numpy as np
    from e2e_agent_sem_lidar2shenron_package.carla_kradar_episode import select_scatterers
    from e2e_agent_sem_lidar2shenron_package.carla_radar_source import prepare_scatterers
    from e2e_agent_sem_lidar2shenron_package.kradar_dra_oracle import (
        oracle_from_scatterers, scatterers_from_frame)

    event_rows = [row for row in products if row["category"] != "normal"]
    if not event_rows:
        return {"supported_cells": 0, "reliable_overlap_cells": 0, "sampled_frames": 0}
    sample = [event_rows[i] for i in np.linspace(
        0, len(event_rows) - 1, min(12, len(event_rows)), dtype=int)]
    supported = overlap = moving = 0
    errors = []
    for row in sample:
        episode = output / "jobs" / row["job_id"] / "episode"
        frame_dir = episode / "frames" / ("%06d" % row["source_index"])
        frame = json.loads((frame_dir / "frame.json").read_text())
        sensor = json.loads((episode / "metadata.json").read_text())["sensors"]["semantic_lidar"]
        lidar = np.load(frame_dir / frame["sensors"]["semantic_lidar"], allow_pickle=False)
        points, velocity, _ = prepare_scatterers(lidar, frame, sensor)
        points, velocity, _ = select_scatterers(points, velocity)
        dynamic = np.isin(points[:, 5].astype(int), (4, 10))
        if not dynamic.any():
            continue
        moving += 1
        rho, theta, loss, speed = scatterers_from_frame(points[dynamic], velocity[dynamic])
        product = Path(row["radar_product"])
        folder = product / "frames" / ("%06d" % row["radar_frame_index"])
        meta = json.loads((product / "radar_meta.json").read_text())
        with np.load(product / "radar_axes.npz") as axes, \
                np.load(folder / "folded_ra.npz") as folded, \
                np.load(folder / "unfolded_ra.npz") as dense:
            oracle = oracle_from_scatterers(
                rho, theta, loss, speed, folded["doppler_at_peak_ra_mps"],
                folded["power_peak_ra"], axes["range_m"], axes["azimuth_deg"],
                axes["velocity_mps"], float(meta["global_power_peak"]),
                window_mode="hann_3d")
            truth = oracle["doppler_oracle_ra_mps"]
            valid = oracle["doppler_oracle_valid"]
            reliable = dense["reliable_mask"]
            both = valid & reliable
            supported += int(valid.sum())
            overlap += int(both.sum())
            errors.extend(np.abs(dense["doppler_unfolded_dense_ra_mps"][both]
                                 - truth[both]).tolist())
    return {"sampled_frames": len(sample), "frames_with_dynamic_scatterers": moving,
            "supported_cells": supported, "reliable_overlap_cells": overlap,
            "median_abs_error_mps": float(np.median(errors)) if errors else None,
            "p90_abs_error_mps": float(np.quantile(errors, .9)) if errors else None,
            "interpretation": "CARLA-assisted audit only; dense candidate remains radar temporal estimate"}


def smoke_report(config, output, accepted, products):
    categories = {row["category"] for row in accepted}
    weather = {row["weather"] for row in accepted}
    expected = {job_key(job) for job in jobs(config, "smoke")}
    selected = {row["job_id"] for row in accepted}
    audit = smoke_dynamic_audit(output, products)
    checks = {"all_smoke_jobs_have_anchors": expected == selected,
              "required_categories": {"normal", "scenario2"}.issubset(categories) and
              bool(categories & {"scenario3", "scenario4", "scenario7_9", "scenario10"}),
              "multiple_weather": len(weather) >= 2,
              "radar_for_every_anchor": len(products) == len(accepted),
              "dynamic_physical_support": audit["supported_cells"] > 0,
              "dynamic_reliable_overlap": audit["reliable_overlap_cells"] > 0}
    return {"status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks, "dynamic_oracle_audit": audit}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Explicitly bypass smoke release for full mode; pilot evidence is still required")
    parser.add_argument("--smoke-name", default="smoke",
                        help="Fresh smoke directory name after a failed run; old evidence is retained")
    parser.add_argument("--output-root", type=Path,
                        help="Override manifest storage root, mainly for tests")
    args = parser.parse_args(argv)
    if args.smoke_name != Path(args.smoke_name).name or not args.smoke_name.startswith("smoke"):
        parser.error("--smoke-name must be one directory name beginning with smoke")
    manifest = args.manifest.resolve()
    config = load_manifest(manifest)
    root = (args.output_root or Path(config["runtime"]["data_root"])).resolve()
    if args.skip_smoke and args.mode != "full":
        parser.error("--skip-smoke is only valid in full mode")
    stamp = preflight(config, manifest, root, args.mode, skip_smoke=args.skip_smoke)
    output = root / (args.smoke_name if args.mode == "smoke" else "full")
    if output.exists() and not args.resume:
        raise FileExistsError("output exists; use --resume: " + str(output))
    output.mkdir(parents=True, exist_ok=True)
    accepted, failures = [], []
    try:
        if args.mode == "smoke":
            with CarlaServer(config, output):
                prepare_scenario2_for_smoke(config, output, root)
        accepted, failures = collect(config, args.mode, output)
        if args.mode == "full" and deficits(accepted, config)["total"]:
            raise RuntimeError("candidate jobs exhausted before quotas were filled")
        if args.mode == "smoke" and (failures or len(accepted) < int(config["smoke_min_anchors"])):
            raise RuntimeError("smoke failed or too few valid anchors")
        products, radar_failures = process_radar(config, output, accepted)
        failures.extend(radar_failures)
        detailed = smoke_report(config, output, accepted, products) if args.mode == "smoke" else None
        status = {"status": "PASS" if not failures and len(products) == len(accepted) else "FAIL",
                  "accepted": len(accepted),
                  "radar_products": len(products), "failures": failures,
                  "deficits": deficits(accepted, config), "signature": stamp,
                  "smoke_release_bypassed": args.skip_smoke}
        if detailed is not None:
            status["smoke_validation"] = detailed
            if detailed["status"] != "PASS":
                status["status"] = "FAIL"
        atomic_json(output / "status.json", status)
        if args.mode == "smoke" and status["status"] == "PASS":
            atomic_json(root / "qualification" / "smoke_release.json", status)
        if status["status"] != "PASS":
            raise RuntimeError("batch completed with failed jobs")
        print(json.dumps(status, indent=2))
    except Exception as exc:
        status_path = output / "status.json"
        if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "FAIL":
            atomic_json(status_path, {"status": "FAIL", "error": "%s: %s" %
                                          (type(exc).__name__, exc),
                                      "accepted": len(accepted), "failures": failures,
                                      "signature": stamp})
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR [batch]: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        sys.exit(1)
