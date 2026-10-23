#!/usr/bin/env bash
set -euo pipefail
project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python=/home/luxinlan/.conda/envs/lxl_radarvla/bin/python
action=${1:-status}
if (( $# )); then shift; fi
exec "$python" "$project_root/scripts/rain_fog_pipeline.py" "$action" "$@"
