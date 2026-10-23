# Controlled 20-epoch Grounding verification

Preserve candidate-sector eligibility, real batch4/accumulation2, seed42, BF16, moving-v1 and all native losses. Compare HG/T-RODNet crossed with constant LR and flat-five-then-cosine LR. All arms start from random initialization and run20 epochs; never rewrite the provenance of an existing five-epoch checkpoint to resume it.

`run_training` and `python -m radar_vla train` support:

- `lr_schedule=constant` / `--lr-schedule constant` (default): unchanged constant optimization behavior.
- `lr_schedule=flat_cosine` / `--lr-schedule flat_cosine`: hold base LR through `lr_decay_start_epoch` (default5), then cosine decay to `base_lr*lr_min_ratio` (default0.1) at the last planned epoch.
- Launcher env: `LR_SCHEDULE`, `LR_DECAY_START_EPOCH`, `LR_MIN_RATIO`.

For20 epochs/base1e-4, cosine LR at epochs5/10/15/20 is1e-4/7.75e-5/3.25e-5/1e-5. The optimizer LR is set at the start of each epoch. Schedule parameters and the total planned budget are immutable protocol fields; history/status include actual LR. Resume derives the next LR analytically from restored epoch and protocol. No Torch scheduler step-order ambiguity or unsaved scheduler state is introduced. Changed plans are rejected by the normal resume fingerprint check.

The Grounding constant and cosine arms share the same first-five-epoch LR/data/initialization, with gt_risk_mix=0. Native regression tests verify their common prefix and uninterrupted-versus-resumed weights exactly on CPU. Real GPU prefix parity against the retained prior five-epoch checkpoint is recorded after completion; it must be inspected before claiming exact reproduction.

Primary comparisons are fixed epoch5 and20. Epoch10/15 and validation-selected best.pt/best_detection.pt are secondary; do not report only a favorable intermediate epoch. External validation observers record per-epoch critical-object recall and decoded-match motion using the same native forward outputs; they never change predictions or optimizer updates. Motion must include matched counts/common-object comparisons. All native five-dimensional Risk metrics are retained. No test data, LLM training or closed-loop safety claims.

Isolated worktree: `/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-long-training-20261021`, branch `codex/long-training-20261021`, base `3b90c785be32adb5350f1be959a80036d2be6107`. The main project remains `/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA` and is untouched. Use the independent editable Python at `/home/luxinlan/data/lxl_data/RadarVLA/model_runs/long_training_20261021/venv/bin/python`; never redirect active environments or use PYTHONPATH.

Data root: `/home/luxinlan/data/lxl_data/RadarVLA`. Manifest: `prepared/taskwise_moving_v1/manifest.jsonl` (SHA2566f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e). Reports/protocol/logs/per-epoch audits: `analysis/long_training_20261021`. Checkpoints: `model_runs/long_training_20261021`. Use the external coordinator: one own worker per spare GPU, HG4GiB/T6GiB allocator caps,2GiB headroom, self-only low-memory recovery, CPU1/workers0/nice19/idle I/O. Other users' process presence is not itself a runtime pause condition.

Rollback uses the prior candidate worktree and its unchanged environment/checkpoints. New source only changes LR controls/recording; model, candidate rules and label definitions are unchanged. The data's moving-only and offline two-sided Doppler limitations still apply.
