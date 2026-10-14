"""Compatibility CLI; C-Shenron owns CARLA radar visualization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from radar_viewer import save_examples


def visualize(episode, radar_output, output, count=4):
    return save_examples(episode, radar_output, output, count)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("radar_output", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--count", type=int, default=4)
    a = p.parse_args()
    print(json.dumps(visualize(a.episode, a.radar_output, a.output, a.count), indent=2))
