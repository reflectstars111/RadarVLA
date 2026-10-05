#!/usr/bin/env bash
# Single-node training launcher. Prints commands with --dry-run; never downloads.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'HELP'
Usage: MANIFEST=/path/data.jsonl OUTPUT=/path/run [variables] bash ./launch_l40s.sh [--dry-run]

Required: MANIFEST, OUTPUT; for new SFT also QWEN_MODEL_PATH, INIT_GROUNDING.
Variables: STAGE=grounding|sft, NPROC_PER_NODE=8, BATCH_SIZE=1,
           ACCUMULATION_STEPS=4, EPOCHS=5, LR=0.0003, WORKERS=2,
           PRECISION=bfloat16, SEED=42, RESUME=0, STOP_AFTER_EPOCH,
           RADAR_VLA_ENV=/path/conda/env, RADAR_VLA_PYTHON=/path/python,
           RADAR_VLA_CONFIG=/path/config.json.
The input grid is native [2,256,107]; four history frames are a configurable initial assumption.
See DISTRIBUTED.md. This does not claim the configuration fits GPU memory.
HELP
    exit 0
fi

dry_run=0
if [[ ${1:-} == --dry-run ]]; then
    dry_run=1
elif [[ $# -gt 0 ]]; then
    printf 'Unknown argument: %s\n' "$1" >&2
    exit 2
fi
if [[ $# -gt 1 ]]; then
    printf 'Expected at most --dry-run.\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
: "${MANIFEST:?Set MANIFEST to the prepared RadarVLA JSONL manifest}"
: "${OUTPUT:?Set OUTPUT to a new run directory (or existing directory with RESUME=1)}"
STAGE=${STAGE:-grounding}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-4}
EPOCHS=${EPOCHS:-5}
WORKERS=${WORKERS:-2}
PRECISION=${PRECISION:-bfloat16}
SEED=${SEED:-42}
LR=${LR:-0.0003}
RESUME=${RESUME:-0}
RADAR_VLA_ENV=${RADAR_VLA_ENV:-"$script_dir/.conda"}
RADAR_VLA_PYTHON=${RADAR_VLA_PYTHON:-"$RADAR_VLA_ENV/bin/python"}
RADAR_VLA_CONFIG=${RADAR_VLA_CONFIG:-"$script_dir/configs/l40s_qwen25_3b.json"}

[[ $STAGE == grounding || $STAGE == sft ]] || { printf 'STAGE must be grounding or sft\n' >&2; exit 2; }
[[ $PRECISION == float32 || $PRECISION == bfloat16 ]] || { printf 'Unsupported PRECISION\n' >&2; exit 2; }
[[ $RESUME == 0 || $RESUME == 1 ]] || { printf 'RESUME must be 0 or 1\n' >&2; exit 2; }
for integer in "$NPROC_PER_NODE" "$BATCH_SIZE" "$ACCUMULATION_STEPS" "$EPOCHS"; do
    [[ $integer =~ ^[1-9][0-9]*$ ]] || { printf 'Rank/batch/accumulation/epoch values must be positive integers\n' >&2; exit 2; }
done
[[ $WORKERS =~ ^[0-9]+$ ]] || { printf 'WORKERS must be a nonnegative integer\n' >&2; exit 2; }
(( NPROC_PER_NODE <= 8 )) || { printf 'This launcher is configured for at most 8 local GPUs\n' >&2; exit 2; }

if [[ $STAGE == sft ]]; then
    : "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the local Qwen2.5-3B model directory}"
    if [[ $RESUME == 0 ]]; then
        : "${INIT_GROUNDING:?Set INIT_GROUNDING to the Stage 1 checkpoint}"
    fi
fi
if [[ $dry_run == 0 ]]; then
    [[ -x $RADAR_VLA_PYTHON ]] || { printf 'Python not executable: %s\n' "$RADAR_VLA_PYTHON" >&2; exit 2; }
    [[ -f $MANIFEST ]] || { printf 'Manifest does not exist: %s\n' "$MANIFEST" >&2; exit 2; }
    [[ -f $RADAR_VLA_CONFIG ]] || { printf 'Config does not exist: %s\n' "$RADAR_VLA_CONFIG" >&2; exit 2; }
    if [[ $STAGE == sft ]]; then
        [[ -d $QWEN_MODEL_PATH ]] || { printf 'Local model directory does not exist: %s\n' "$QWEN_MODEL_PATH" >&2; exit 2; }
        if [[ $RESUME == 0 ]]; then
            [[ -f $INIT_GROUNDING ]] || { printf 'Grounding checkpoint does not exist: %s\n' "$INIT_GROUNDING" >&2; exit 2; }
        fi
    fi
fi

# Resolve paths before switching directory so relative MANIFEST/OUTPUT arguments
# continue to refer to the caller's directory. GNU realpath -m allows new outputs.
MANIFEST=$(realpath -m -- "$MANIFEST")
OUTPUT=$(realpath -m -- "$OUTPUT")
RADAR_VLA_PYTHON=$(realpath -m -- "$RADAR_VLA_PYTHON")
RADAR_VLA_CONFIG=$(realpath -m -- "$RADAR_VLA_CONFIG")
command=("$RADAR_VLA_PYTHON" -m torch.distributed.run --standalone --nnodes=1
    "--nproc_per_node=$NPROC_PER_NODE" -m radar_vla train
    --manifest "$MANIFEST" --output "$OUTPUT" --stage "$STAGE"
    --config "$RADAR_VLA_CONFIG" --device cuda --precision "$PRECISION"
    --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --accumulation-steps "$ACCUMULATION_STEPS"
    --lr "$LR" --workers "$WORKERS" --seed "$SEED")
if [[ $STAGE == sft ]]; then
    command+=(--language-model-path "$(realpath -m -- "$QWEN_MODEL_PATH")")
    if [[ $RESUME == 0 ]]; then
        command+=(--init-grounding "$(realpath -m -- "$INIT_GROUNDING")")
    fi
fi
if [[ $RESUME == 1 ]]; then
    command+=(--resume)
fi
if [[ -n ${STOP_AFTER_EPOCH:-} ]]; then
    [[ $STOP_AFTER_EPOCH =~ ^[1-9][0-9]*$ ]] || { printf 'STOP_AFTER_EPOCH must be positive\n' >&2; exit 2; }
    command+=(--stop-after-epoch "$STOP_AFTER_EPOCH")
fi

printf 'Stage=%s ranks=%s batch/rank=%s accumulation=%s full-window batch=%s\n' \
    "$STAGE" "$NPROC_PER_NODE" "$BATCH_SIZE" "$ACCUMULATION_STEPS" \
    "$((NPROC_PER_NODE * BATCH_SIZE * ACCUMULATION_STEPS))"
printf 'cd %q && ' "$project_root"
printf '%q ' "${command[@]}"
printf '\n'
if [[ $dry_run == 1 ]]; then
    exit 0
fi
cd -- "$project_root"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "${command[@]}"
