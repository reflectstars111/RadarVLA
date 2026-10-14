"""Finish radar after parallel collection and preserve a truthful batch status."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from batch import accepted_from_jobs, atomic_json, deficits, load_manifest


def wait_for_report(path, process_name, timeout_s=86400):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return json.loads(path.read_text())
        running = subprocess.run(["pgrep", "-f", process_name], capture_output=True).returncode == 0
        if not running:
            raise RuntimeError(process_name + " stopped without writing " + str(path))
        time.sleep(30)
    raise TimeoutError("timed out waiting for " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    config = load_manifest(manifest)
    output = Path(config["runtime"]["data_root"]).resolve() / "full"
    status_path = output / "status.json"
    try:
        collection = wait_for_report(output / "collection_remaining_status.json",
                                     "collection/collector/collect_remaining.py")
        if collection.get("status") != "QUOTAS_FILLED":
            raise RuntimeError("collection did not fill all quotas")
        first_radar = wait_for_report(output / "radar_existing_status.json",
                                      "collection/collector/radar_existing.py")
        if first_radar.get("status") != "PASS" or first_radar.get("accepted") != 9000:
            raise RuntimeError("first 9,000-frame radar pass did not finish cleanly")
        subprocess.run([sys.executable, str(Path(__file__).with_name("radar_existing.py")),
                        "--manifest", str(manifest), "--output", str(output)], check=True)
        final_radar = json.loads((output / "radar_existing_status.json").read_text())
        accepted = accepted_from_jobs(output)
        index = [json.loads(line) for line in (output / "dataset_index.jsonl").open()]
        missing = deficits(accepted, config)
        if (final_radar.get("status") != "PASS" or len(accepted) != 10000 or
                len(index) != 10000 or missing["total"] or any(
                    any(group.values()) for name, group in missing.items()
                    if name != "total")):
            raise ValueError("final radar index or collection quota is incomplete")
        report = {"status": "RAW_RADAR_COMPLETE", "accepted": len(accepted),
                  "radar_products": len(index), "deficits": missing,
                  "smoke_release_bypassed": True,
                  "training_ready": "UNTESTED",
                  "historical_failed_jobs": collection["historical_failed_jobs"]}
    except Exception as exc:
        report = {"status": "FAIL", "error": "%s: %s" % (type(exc).__name__, exc),
                  "smoke_release_bypassed": True}
        atomic_json(status_path, report)
        raise
    atomic_json(status_path, report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
