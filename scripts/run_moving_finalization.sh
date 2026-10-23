#!/usr/bin/env bash
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
DATA_ROOT=${1:-/home/luxinlan/data/lxl_data/RadarVLA}
PYTHON=${RADAR_VLA_PYTHON:-/home/luxinlan/.conda/envs/lxl_radarvla/bin/python}
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$PROJECT"
exec "$PYTHON" scripts/finalize_moving_agents.py --data-root "$DATA_ROOT" \
  --config "$PROJECT/configs/l40s_qwen25_3b_taskwise_moving_v1.json" \
  >> "$DATA_ROOT/analysis/taskwise_moving_v1/migration.log" 2>&1
