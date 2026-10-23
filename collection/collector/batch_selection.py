"""Conservative, episode-local selection of real future anchors."""
from __future__ import annotations

import json
import math
from pathlib import Path


CATEGORIES = {
    "scenario2": {"Scenario2"},
    "scenario3": {"Scenario3"},
    "scenario4": {"Scenario4"},
    "scenario7_9": {"Scenario7", "Scenario8", "Scenario9"},
    "scenario10": {"Scenario10"},
}


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def speed(actor):
    return math.sqrt(sum(float(x) ** 2 for x in actor["velocity_world_mps"]))


def distance_xy(left, right):
    return math.hypot(float(left[0]) - float(right[0]),
                      float(left[1]) - float(right[1]))


def _scenario_windows(frames, events, category):
    """Use recorded transitions and observed motion, never a terminal flag alone."""
    selected = {item.get("index"): item for item in events
                if item.get("event") == "scenario_selected" and
                item.get("scenario_name") in CATEGORIES[category]}
    by_index = {}
    for item in events:
        if item.get("index") in selected and item.get("carla_frame") is not None:
            by_index.setdefault(item["index"], {}).setdefault(item["event"], []).append(
                int(item["carla_frame"]))
    result = []
    for index, definition in selected.items():
        history = by_index.get(index, {})
        completed = history.get("scenario_completed", [])
        if not completed:
            continue
        if category == "scenario2" and not all(history.get(name) for name in (
                "actor_restored", "scenario_activated", "following_started",
                "braking_started", "scenario_completed")):
            continue
        trigger = definition.get("trigger_position") or {}
        if "x" not in trigger or "y" not in trigger:
            continue
        end = min(completed)
        prior = [row for row in frames if row["carla_frame"] <= end and
                 distance_xy(row["ego"]["pose"]["position_m"],
                             (trigger["x"], trigger["y"])) <= 5.0]
        if not prior:
            continue
        start = prior[0]["carla_frame"]
        if category == "scenario2":
            start = max(start, min(history["actor_restored"]))
        interval = [row for row in frames if start <= row["carla_frame"] <= end]
        if not interval:
            continue
        actor_ids = set(map(int, definition.get("other_actor_ids") or []))
        physical = False
        ego_reacted = False
        motion_frames = []
        for row in interval:
            ego = row["ego"]
            nearby = [actor for actor in row.get("actors", [])
                      if actor.get("actor_id") != ego["actor_id"] and
                      actor.get("actor_type") in ("vehicle", "pedestrian") and
                      (not actor_ids or actor["actor_id"] in actor_ids) and
                      distance_xy(actor["pose"]["position_m"],
                                  ego["pose"]["position_m"]) <= 50.0]
            moving_nearby = any(speed(actor) >= .5 for actor in nearby)
            braking = float((ego.get("commanded_control") or {}).get("brake", 0)) > 0.05
            ego_reacted |= braking and bool(nearby)
            if moving_nearby:
                physical = True
                motion_frames.append(row["carla_frame"])
        if physical and ego_reacted and motion_frames:
            # Keep the whole in-view interaction, including stopped lead + ego braking.
            result.append({"start": min(motion_frames), "end": end,
                           "scenario_index": index, "actor_ids": sorted(actor_ids)})
    return result


def episode_candidates(episode, category, weather, split):
    """Return unique frames with observed 30-step Ego future and event evidence."""
    episode = Path(episode)
    paths = sorted((episode / "frames").glob("[0-9]*/frame.json"),
                   key=lambda path: int(path.parent.name))
    frames = [json.loads(path.read_text()) for path in paths]
    anchors = {row["carla_frame"]: row for row in read_jsonl(episode / "anchors.jsonl")}
    events = list(read_jsonl(episode / "events.jsonl"))
    windows = _scenario_windows(frames, events, category) if category != "normal" else []
    ego_ids = {row["carla_frame"]: str(row["ego"]["actor_id"]) for row in frames}
    candidates = []
    for frame in frames:
        fid = frame["carla_frame"]
        future = anchors.get(fid, {}).get("futures_by_actor_id", {}).get(ego_ids[fid], {})
        if len(future.get("valid", [])) != 30 or not all(future["valid"]):
            continue
        active = next((window for window in windows
                       if window["start"] <= fid <= window["end"]), None)
        if category != "normal" and active is None:
            continue
        if active is not None:
            relevant = [actor for actor in frame.get("actors", [])
                        if actor.get("actor_type") in ("vehicle", "pedestrian") and
                        (not active["actor_ids"] or actor["actor_id"] in active["actor_ids"]) and
                        distance_xy(actor["pose"]["position_m"],
                                    frame["ego"]["pose"]["position_m"]) <= 50]
            braking = float((frame["ego"].get("commanded_control") or {}).get("brake", 0)) > .05
            if not relevant or not (any(speed(actor) >= .5 for actor in relevant) or braking):
                continue
        if category == "normal" and any(item.get("event") == "scenario_completed" for item in events):
            # Normal jobs use no_scenarios.json. Reject accidental scene activation.
            continue
        candidates.append({"episode_id": frame["episode_id"], "carla_frame": fid,
                           "source_index": int(frame["index"]), "category": category,
                           "weather": weather, "split": split,
                           "scenario_index": active["scenario_index"] if active else None})
    return candidates, {"windows": windows, "full_ego_future": len(candidates),
                        "observed_frames": len(frames)}


def choose_with_quotas(candidates, accepted, category_quota, split_quota, weather_quota, joint_quota=None):
    """Select without duplicates or quota overflow; the caller persists acceptance."""
    counts = {"category": {}, "split": {}, "weather": {}}
    keys = set()
    for row in accepted:
        key = (row["episode_id"], row["carla_frame"])
        if key in keys:
            raise ValueError("duplicate accepted anchor")
        keys.add(key)
        for field in counts:
            value = row[field]
            counts[field][value] = counts[field].get(value, 0) + 1
    joint_counts = {}
    for row in accepted:
        pair = row['category'] + '/' + row['weather']
        joint_counts[pair] = joint_counts.get(pair, 0) + 1
    picked = []
    for row in candidates:
        key = (row["episode_id"], row["carla_frame"])
        if key in keys:
            continue
        if any(counts[field].get(row[field], 0) >= quota[row[field]] for field, quota
               in (("category", category_quota), ("split", split_quota),
                   ("weather", weather_quota))):
            continue
        pair = row['category'] + '/' + row['weather']
        if joint_quota is not None and joint_counts.get(pair, 0) >= joint_quota[pair]:
            continue
        joint_counts[pair] = joint_counts.get(pair, 0) + 1
        keys.add(key)
        picked.append(row)
        for field in counts:
            value = row[field]
            counts[field][value] = counts[field].get(value, 0) + 1
    return picked
