# KRS-v2 single-variable weight controls

Base: ecfb71ce53f1aba12b54722d5243cc34a43f73fe. Native default configurations keep their original resolved protocol. Optional top-level loss_weights supports finite nonnegative objective coefficients; physical normalization scales cannot be overridden.

Arm A: configs/l40s_qwen25_3b_taskwise_moving_trodnet_krsv2_dmin025.json changes only distance=0.25.
Arm B: configs/l40s_qwen25_3b_taskwise_moving_trodnet_krsv2_areq3.json changes only brake=3.0.

All other model/data parameters match the baseline. The Grounder stores objective weights as runtime metadata, with no added parameters or changed predictions. Config weights are pinned in checkpoint protocol, checked by native resume comparison and restored by load_models for consistent validation/evaluation loss. Old configurations retain default weights.

Grounding workers use seed42 fresh initialization,20epochs,batch4,accumulation2,LR2e-4,flat5epochs then update-cosine floor0.1,BF16. No checkpoint continuation between arms; no optimizer updates in real-sample preflight. The real test split is not evaluated for selection.

Weighted instrumentation must scale each Risk term before computing feature gradient norms; detection loss is total minus weighted Risk. Record raw and weighted component values separately. Do not compare differently weighted val total losses as if they were the same objective.

Native config/reload/resume and unit-weight equivalence tests: tests/test_loss_weights.py. External observer equivalence and coefficient-gradient scaling tests are retained in data analysis/krsv2_weight_controls_20261023/.

Source worktree,editable environment,worker runtime files,configuration fingerprints and output directories are isolated. Source of the completed baseline and the main checkout are not changed. A and B may run concurrently on separate available GPUs; each worker has its own6GiB allocator cap,CPUthreads1,dataworkers0,nice19,ionice idle and exits only itself if freeVRAM<8GiB. Other-user process presence alone does not pause a worker.

All epoch checkpoints are retained. Primary comparison: fixedepoch20; fixedepoch12/16 are secondary views. Compare AP/Brier,critical recall,false alarms,danger dmin MAE,positive-demand and zero-demand braking error,and motion on common matched IDs. Single seed and rare positive episodes limit conclusions; no automatic winner and no automatic100epoch training.
