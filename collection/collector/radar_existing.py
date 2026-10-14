"""Generate resumable K-Radar products for already accepted batch anchors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from batch import accepted_from_jobs, atomic_json, load_manifest, process_radar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_manifest(args.manifest.resolve())
    output = args.output.resolve()
    accepted = accepted_from_jobs(output)
    if not accepted:
        raise ValueError("no accepted anchors to process")
    products, failures = process_radar(config, output, accepted)
    report = {"status": "PASS" if not failures and len(products) == len(accepted) else "FAIL",
              "accepted": len(accepted), "radar_products": len(products),
              "failures": failures}
    atomic_json(output / "radar_existing_status.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
