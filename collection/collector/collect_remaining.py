"""Resume CARLA collection without waiting to start or duplicating radar work."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch import atomic_json, collect, deficits, load_manifest, preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    config = load_manifest(manifest)
    root = Path(config["runtime"]["data_root"]).resolve()
    signature = preflight(config, manifest, root, "full", skip_smoke=args.skip_smoke)
    output = root / "full"
    if not output.is_dir():
        raise FileNotFoundError("existing full batch directory is required")
    accepted, failures = collect(config, "full", output)
    missing = deficits(accepted, config)
    historical_failures = sum(1 for _ in (output / "jobs").glob("*/failure.json"))
    report = {"accepted": len(accepted), "deficits": missing,
              "current_manifest_failed_jobs": len(failures),
              "historical_failed_jobs": historical_failures,
              "signature": signature,
              "status": "QUOTAS_FILLED" if missing["total"] == 0 and all(
                  not any(group.values()) for name, group in missing.items()
                  if name != "total") else "QUOTA_SHORTFALL"}
    atomic_json(output / "collection_remaining_status.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "QUOTAS_FILLED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
