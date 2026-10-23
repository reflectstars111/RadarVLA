#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${1:-/home/luxinlan/data/lxl_data/RadarVLA}
if (( $# )); then shift; fi
PYTHON=${RADAR_VLA_PYTHON:-/home/luxinlan/.conda/envs/lxl_radarvla/bin/python}
REPORTS="$DATA_ROOT/analysis/taskwise_moving_v1"
mkdir -p "$REPORTS"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$PROJECT"
exec "$PYTHON" "$PROJECT/scripts/migrate_moving_agents.py" \
  --data-root "$DATA_ROOT" --organize-existing \
  --config "$PROJECT/configs/l40s_qwen25_3b_taskwise_moving_v1.json" "$@" \
  >> "$REPORTS/migration.log" 2>&1
