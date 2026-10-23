# RadarVLA spatiotemporal backbone adapters

## Isolation and scope

The implementation branch is `codex/spatiotemporal-backbones-20261020`, in the
independent worktree `/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-backbones-20261020`.
The original `RadarVLA` directory, its uncommitted work, active branch, and its
Conda editable installation are preserved. Existing work was snapshotted in
baseline commit `88861b238d9e1fd2902334ed6a56b8e05aef5fac` before feature changes.
The first HG implementation checkpoint is commit
`97bbc96e20174fb6a704394a6301aa9808132d40`.

Only the feature extractor and its configuration/compatibility boundary change.
Normalization, unreliable Doppler zeroing (Power retained), physical geometry,
query attention, detector heads, losses, NMS, Risk, and Planner algorithms are
unchanged. `backbone` is appended to ModelConfig, preserving the older positional
constructor arguments; its default is `cnn2d`. CNN encoder parameter names and
state dictionaries remain unchanged. New backbones start from random weights.

No formal training is started. The dry-run constructs no optimizer and performs
no optimizer updates. BatchNorm buffers are restored after backward checks.
Original checkpoint bytes are unchanged. Current two-sided Doppler preprocessing
is an **offline dataset**; these backbones do not establish an online causal
sensor pipeline. Within a supplied history window, reconstructed earlier feature
slots can contain information from later supplied slots; they are fused features,
not independent causal historical-frame encodings.

## Architecture and upstream provenance

Both upstream repositories use the MIT license, Copyright (c) 2021 Yizhou Wang.
Complete notices are retained in `src/RODNET_LICENSE.txt` and `third_party/`;
the installed package includes the notice and `BACKBONE_UPSTREAM.json`.
Exact upstream source files are preserved under `third_party/` for audit only,
never imported at runtime.

- HG: `yizhou-wang/RODNet`, commit
  `44fd61fa7b433b920469bd8455a54d5676e57fd8`,
  `rodnet/models/backbones/hg.py`.
- T-RODNet: `Zhuanglong2/T-RODNet`, commit
  `39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9`, `Nets/T_RODNet.py`.

The adapter accepts normalized `[B,T,2,R,A]`. It permutes to `[B,2,T,R,A]`,
left-pads time to at least four and a multiple of four, and tail-pads R/A to
multiples of eight. After feature reconstruction it crops padding and spatially
pools to `ceil(R/4) × ceil(A/4)`. It never interpolates raw radar values.
The default output is `[B,4,64,64,27]`; `hidden_dim != 64` adds a 1×1×1 projection.
Temporal ablation selects the latest observation **before** the adapter pads it.
HG retains official BatchNorm: training with batch=1 and a one-voxel bottleneck
is unsupported by that normalization; small-shape tests use valid bottleneck sizes.

HG retains the two stem convolutions, the unchanged official `RODEncode` and
`RODDecode`, temporal kernels of length nine, two temporal reductions, additive
skips and BatchNorm/ReLU/PReLU. Only one hourglass is instantiated. Confidence
and feedback heads are omitted.

T-RODNet retains the active stem, DAM, 64/128/256 encoder stages, two blocks per
stage, 2/4/8 attention heads, window=4, and upstream stochastic-depth schedule
0→0.1. Time reductions are in the stem and first merge; the second merge is
spatial only. Decoder projections, normalized skips, cross-attention using
encoder K/V, sinusoidal decoder features and trilinear upsampling are retained.
The final class predictor is omitted. The official stem uses GroupNorm(16,64),
verified numerically against the pinned implementation. Active merging and DAM
operations also have exact numerical parity tests.

T-RODNet-specific adaptations:

- Pass actual dimensions; align upsampling explicitly to the corresponding skip.
- Derive each actual window from `min(axis_length,4)` and disable shift on axes
  no larger than a window. Rebuild D/H/W relative-position indices into the
  maximum-window bias table; never slice a prefix of a fixed index matrix.
- Keep physical padding validity distinct from unreliable Doppler and cyclic
  boundary masks. Coarse validity means at least one real source position in
  the corresponding reduction cell (adaptive max pooling); it is not a
  Doppler reliability estimate or a convolutional visibility guarantee.
- Mask invalid keys in shifted and unshifted attention, zero invalid outputs,
  and give fully masked rows zero attention without NaN forward/backward values.
- Encoder and decoder use the same layout functions and assert window geometry
  and key-validity alignment before reusing K/V.
- Preserve the upstream decoder's additional query scaling; recalibrating that
  behavior is outside this port's scope.
- Remove inactive parameter branches and profiling-only imports. All retained
  backbone parameters participate in backward, which is checked explicitly to
  avoid unused-parameter problems in subsequent distributed training.
- No timm, thop, fvcore, einops, deformable-convolution or dataset dependencies
  are introduced. Only the existing Torch runtime is required by the adapters.

Detection consumes the last fused feature slice and retains **3456 candidates**.
Risk/detection use eight Radar query tokens. When the existing
`planner_ego_context=true`, the public Planner input contains nine tokens:
eight Radar tokens plus one Ego token. That distinction is preserved.
Dense detection and Risk receive the complete target set, without a new
eight-agent truncation. Planner's existing selection/grammar policy is unchanged.

## Data, configurations, and compatibility

All comparison arms use the same accepted manifest:

`/home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_v1/manifest.jsonl`

SHA256: `6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e`.
The acceptance is PASS; split sizes remain train=6847, val=1472, test=1522.
Dry-run examples use only train records: empty, 14-agent, SHORT and UNKNOWN.
No test inference is performed. No data or Doppler preprocessing is regenerated.

Independent configurations:

- CNN: existing `configs/l40s_qwen25_3b_taskwise_moving_v1.json` (unchanged).
- HG: `configs/l40s_qwen25_3b_taskwise_moving_hg_v1.json`.
- T-RODNet: `configs/l40s_qwen25_3b_taskwise_moving_trodnet_v1.json`.

Structural options in the new configs are explicit and validated. The current
port fixes the upstream-style architecture rather than silently accepting
unimplemented window/channel/depth settings. A persistent backbone marker exists
only in new HG/T-RODNet state dictionaries; old CNN checkpoints need no marker.
Cross-backbone loading is rejected even with `strict=False`. Grounding, SFT,
evaluation and diagnosis must use the matching configuration and checkpoint.
Legacy CNN Grounder outputs are bitwise identical on the original epoch-5 SFT
checkpoint in a real input regression; this does not load or benchmark the full
3B language model. Native HF/LoRA joint backward is checked using local tiny Qwen.

This manifest labels only vehicles with current horizontal speed >=0.1m/s.
Stationary vehicles, pedestrians and cyclists are background. Risk/Mode labels
were recomputed for this target set. Results must not be compared as unchanged
labels against `taskwise_local_v1`, or interpreted as stationary-obstacle safety.

## Isolated environment and executable dry-runs

The environment is a system-site-packages venv sharing existing dependencies,
with its **own** editable package pointing to this worktree. No shared Conda
package is replaced, upgraded or redirected.

```bash
W=/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-backbones-20261020
P=/home/luxinlan/data/lxl_data/RadarVLA/model_runs/backbone_adaptation_20261020/venv/bin/python
A=/home/luxinlan/data/lxl_data/RadarVLA/analysis/backbone_adaptation_20261020
M=/home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_v1/manifest.jsonl
cd "$W"
"$P" -c 'import radar_vla; print(radar_vla.__file__)'

# Tests perform backward checks, with no optimizer steps.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$P" -m pytest -q tests/test_backbones.py tests/test_backbone_windows.py

# Print launch commands only; these commands never start training.
"$P" scripts/radarvla_cli.py --project-root "$W" --paths "$A/hg_paths.json" \
  train-grounding --manifest "$M" \
  --output /home/luxinlan/data/lxl_data/RadarVLA/model_runs/HG_CONTROL_NOT_STARTED --dry-run
"$P" scripts/radarvla_cli.py --project-root "$W" --paths "$A/trodnet_paths.json" \
  train-grounding --manifest "$M" \
  --output /home/luxinlan/data/lxl_data/RadarVLA/model_runs/TRODNET_CONTROL_NOT_STARTED --dry-run
```

Use the explicit isolated Python and paths files. The old default local paths
configuration is intentionally preserved from the source snapshot and points at
the original environment. Qwen weights are referenced by absolute path to the
existing original installation, not copied into the worktree.

The benchmark command, for an independently chosen spare GPU, is:

```bash
# Replace UUID with a currently spare GPU; use a new output directory each run.
CUDA_VISIBLE_DEVICES=GPU-UUID OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  nice -n 19 ionice -c 3 "$P" scripts/benchmark_backbones.py \
  --manifest "$M" --config configs/l40s_qwen25_3b_taskwise_moving_trodnet_v1.json \
  --output "$A/NEW_UNIQUE_DRYRUN" --backbones cnn2d rodnet_hg trodnet
```

The completed launch selected a GPU with >=32GiB free and <=10% utilization.
Benchmark allocation is capped at 40% of device memory and the process exits
itself if free VRAM falls below 8GiB. CPU threads=1, nice=19, ionice=idle. The
benchmark checks the accepted manifest hash, records source/script/config hashes,
uses BF16, batch=1, one warmup and three timed forwards per case, and synchronizes
CUDA around timing. It restores all parameters/buffers and executes no optimizer.
Do not overlap a new benchmark with another run on the same selected GPU.

## Resource acceptance and future experimental protocol

Machine measurements and all raw case timings are in
`analysis/backbone_adaptation_20261020/final_dryrun/results.json`; the human report
is in that analysis directory's `REPORT.md`. These are untrained engineering
checks, not precision/recall measurements. Both new backbones fit the measured
Grounder dry-run resource envelope. Adam states, larger batches and HF 3B joint
training require their own subsequently authorized measurements.

Future training comparisons must hold manifest/hash, episode split, seed set,
effective batch, optimizer-step budget and evaluation thresholds fixed. Report
GPU hours as well as quality; compare motion-agent recall **with background false
positives**, radial/2D velocity errors and future ADE, five independently masked
Risk metrics, calibration and Pcol AP. Report errors on a common matched-object
subset as well as each model's own matches. Evaluate Planner mode and rollout
only after a matching SFT experiment; detection improvement alone is insufficient.
No formal training or scheduler for future training is created by this task.

## Review and rollback

Return to `/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA` and its original
Conda environment to use the unchanged original project. No source rollback or
reinstallation of that environment is needed.

Review only feature commits after the snapshot baseline:

```bash
git diff 88861b238d9e1fd2902334ed6a56b8e05aef5fac..HEAD
git log --oneline 88861b238d9e1fd2902334ed6a56b8e05aef5fac..HEAD
```

Do not reset/clean the original dirty worktree. Do not treat the snapshot commit
as a new feature to merge. Any later integration should apply only the feature
commits into a prepared clean branch. Source snapshots, the feature patch and
Git bundle are exported outside the repository for review/recovery.
