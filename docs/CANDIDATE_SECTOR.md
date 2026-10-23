# Unified detection candidate coverage

The opt-in `model.detection_candidate_policy=supervised_sector_v1` gives training assignment and decoding one calibrated anchor domain. The default `legacy_v1` preserves old checkpoint behavior.

## Contract

- Bounds are fixed in model config/checkpoint: `detection_candidate_range_m=50`, `detection_candidate_half_fov_deg=53`. Forward computes a boolean `[B,K]` mask from calibrated latest radar pose and anchor geometry in FP32. It does not read GT boxes, risk labels, or annotation availability. Calibrated `radar_pose` is required.
- Training requires each supervised sample's declared annotation sector to match the configured bounds. Hungarian assignment operates only on eligible anchors. All GT instances, including straddling boxes and centers outside the sector, remain present and can regress an offset from an inner anchor. Too few eligible anchors fails explicitly; no targets are dropped.
- Decoding applies the identical mask before confidence sorting, candidate caps and NMS. Raw head capacity remains 3456 on `[4,2,256,107]`; the representative real inputs have 1458 eligible candidates. This is a spatial coverage restriction, not an agent count cap.
- Raw GT-matched physical metrics also use eligible candidates. Confidence/NMS detection scoring and its sector/GT matching rules are unchanged. Risk computation, radar input, normalization, Doppler masking and losses/scales are unchanged.
- Persistent `_candidate_sector` identifies the new contract and bounds. Model loading rejects cross-policy/bounds weights, even with `strict=False`. Existing legacy checkpoint state keys and forward outputs are preserved. New-contract Grounding starts from random initialization; SFT uses its matching checkpoint/config. Do not reinterpret old weights by merely changing JSON.

## Independent configs

- `configs/l40s_qwen25_3b_taskwise_moving_v1_candidate_sector.json`
- `configs/l40s_qwen25_3b_taskwise_moving_hg_v1_candidate_sector.json`
- `configs/l40s_qwen25_3b_taskwise_moving_trodnet_v1_candidate_sector.json`

All use the accepted `prepared/taskwise_moving_v1/manifest.jsonl`, SHA256 `6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e`. This task does not change labels or data. Static vehicles, pedestrians and cyclists remain background under this dataset's policy; two-sided Doppler remains offline preprocessing.

## Isolated execution

Source worktree: `/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-candidate-domain-20261021`, branch `codex/candidate-domain-20261021`.

Python: `/home/luxinlan/data/lxl_data/RadarVLA/model_runs/candidate_contract_20261021/venv/bin/python`. It has an independent editable install; do not redirect the active original/backbone environments or use PYTHONPATH.

Read-only real-data acceptance evidence: `/home/luxinlan/data/lxl_data/RadarVLA/analysis/candidate_contract_20261021/acceptance.json`. All three backbones passed empty/boundary/12-target forward/backward with no optimizer; outside-anchor direct objectness gradients are zero, and old trained checkpoints retain bitwise-equal outputs under legacy configs.

Launcher preview (does not train; select an actually available GPU for execution):

```bash
RADAR_VLA_PYTHON=/home/luxinlan/data/lxl_data/RadarVLA/model_runs/candidate_contract_20261021/venv/bin/python \
RADAR_VLA_CONFIG=configs/l40s_qwen25_3b_taskwise_moving_hg_v1_candidate_sector.json \
MANIFEST=/home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_v1/manifest.jsonl \
OUTPUT=/home/luxinlan/data/lxl_data/RadarVLA/model_runs/candidate_contract_20261021/planned/hg \
CUDA_VISIBLE_DEVICES=6 NPROC_PER_NODE=1 WORKERS=0 BATCH_SIZE=4 ACCUMULATION_STEPS=2 EPOCHS=5 LR=0.0001 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash launch_l40s.sh --dry-run
```

Formal jobs must use the external resource coordinator/guard: low CPU/I/O priority, calibrated per-worker allocator budgets and self-only memory-pressure exit. Foreign process presence alone is not a pause condition. Keep outputs separate and never overwrite the existing four optimization runs. This source change alone is not evidence of precision, recall, motion, Risk or closed-loop improvement.

## Verification and rollback

108 focused tests passed across candidate coverage, detection domain, dense head, model, pipeline, physical metrics and backbones. New regression tests cover boundary GT assignment, high-score invalid candidates before NMS/caps, annotation mismatch, empty/capacity cases, more than eight targets, calibration, observation-only forward, strict checkpoint policy and eligible physical metrics. See `tests/test_candidate_contract.py`.

Rollback means continuing to use the unchanged previous backbone worktree/environment and old configs/checkpoints. No reset, cleanup or weight transfer is necessary. The original dirty project and active training source are untouched.
