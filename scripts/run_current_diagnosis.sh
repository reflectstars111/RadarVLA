#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
AUDIT_PYTHON=/home/luxinlan/.conda/envs/lxl_radarvla/bin/python
AUDIT_OUT=${1:?Usage: bash scripts/run_current_diagnosis.sh NEW_OUTPUT_DIRECTORY}
AUDIT_MANIFEST=${RADAR_VLA_AUDIT_MANIFEST:-analysis/rebalanced_50m_domain/manifest.jsonl}
AUDIT_DEVICE=${RADAR_VLA_AUDIT_DEVICE:-cuda:0}
if [ -e "$AUDIT_OUT" ]; then
  echo "Refusing to overwrite existing diagnosis directory: $AUDIT_OUT" >&2
  exit 1
fi
AUDIT_RAW_ROOT=$($AUDIT_PYTHON -c 'import json,sys;from pathlib import Path; print(Path(json.loads(next(open(sys.argv[1])))["route_archive"]).parents[3])' "$AUDIT_MANIFEST")
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/radarvla_mpl
$AUDIT_PYTHON scripts/audit_current_pipeline.py --manifest "$AUDIT_MANIFEST" --raw-root "$AUDIT_RAW_ROOT" --output-dir "$AUDIT_OUT"
$AUDIT_PYTHON scripts/diagnose_detection.py --manifest "$AUDIT_OUT/detection_manifest.jsonl" --checkpoints analysis/retrained_grounding_v2/best.pt analysis/retrained_sft_v2/epoch_002.pt analysis/retrained_sft_v2/epoch_003.pt analysis/retrained_sft_v2/epoch_005.pt --output-dir "$AUDIT_OUT/detection" --device "$AUDIT_DEVICE"
$AUDIT_PYTHON scripts/diagnose_sft_trajectory.py --manifest "$AUDIT_MANIFEST" --checkpoint analysis/retrained_sft_v2/epoch_003.pt --output "$AUDIT_OUT/trajectory_decomposition.json" --device "$AUDIT_DEVICE"
$AUDIT_PYTHON scripts/diagnose_risk_modes.py --manifest "$AUDIT_MANIFEST" --checkpoint analysis/retrained_sft_v2/epoch_003.pt --output "$AUDIT_OUT/risk_modes_fresh.json" --device "$AUDIT_DEVICE"
$AUDIT_PYTHON scripts/render_diagnosis_example.py --manifest "$AUDIT_OUT/detection_manifest.jsonl" --output-dir "$AUDIT_OUT" --device "$AUDIT_DEVICE"
$AUDIT_PYTHON scripts/summarize_current_diagnosis.py --output-dir "$AUDIT_OUT"
