# RadarVLA current runbook

## Rain/fog 10k batch (2026-10-22)

Use `bash scripts/collect_rain_fog_10k.sh check|status|smoke|run|resume`.
The batch uses moving full-BBox v2 labels and explicit paths inside the existing
full/prepared/analysis categories. `run` requires matching PASS smoke; no training
is started. See [rain/fog collection](docs/collection/RAIN_FOG_COLLECTION.md).


## Current Agent policy and dataset locations (2026-10-19)

The current policy is `moving_vehicle_01_background_v1`: current-frame **world
horizontal speed >= 0.1 m/s**, original **50 m / +/-53 degree BBox-intersection**
domain. Pedestrians, bicycles and stationary vehicles are **background**, with
normal objectness penalties and FP accounting, without ignore regions. Motorcycles
retain the existing vehicle classification. Ego motion/Doppler does not determine
world motion membership. Full world snapshots, unfiltered `observed_agents` and
real futures remain available. A moving actor's future is retained if it later stops.

Detection and Local Risk use **all** eligible vehicles without a nearest-distance
or eight-agent cutoff. Planner behavior is unchanged: nearest eight selected,
SHORT describes at most two critical agents. Excluded objects do not contribute
to this version's motion-only Risk. Unknown annotations remain unknown.

Training data lives in `<data_root>/prepared/<version>/`; statistics and diagnostics
live in `<data_root>/analysis/<version>/`. The existing baseline moves to
`prepared/taskwise_local_v1/`, with compatibility links at its old analysis paths.
The new release is `prepared/taskwise_moving_v1/manifest.jsonl`; its reports,
`migration.log` and `status.json` live in `analysis/taskwise_moving_v1/`.
Use the new release only after status is `COMPLETE` with matching validation.

Start or resume the independent migration/acceptance job from the project root:

```bash
screen -dmS radarvla_moving_v1 bash scripts/run_moving_migration.sh
tail -f /home/luxinlan/data/lxl_data/RadarVLA/analysis/taskwise_moving_v1/migration.log
cat /home/luxinlan/data/lxl_data/RadarVLA/analysis/taskwise_moving_v1/status.json
```

The job starts no training or CARLA/radar generation. It locks its output, journals
committed rows, rejects changed source/code/config on resume, checks raw membership,
recomputes Risk/Mode, checks representative real gradients and runs full sensor
validation. Repeat after failure to resume; inspect its failure first. Changed
source/code requires a new output version; do not delete partial data to resume.

If v1 acceptance failed because `Metadata` lacked `supervision`, the script
entrypoint now supplies that Dataset flag while preserving the original migration
provenance and hash checks. Restart only acceptance, without regenerating labels:

```bash
screen -dmS radarvla_moving_verify bash scripts/run_moving_migration.sh /home/luxinlan/data/lxl_data/RadarVLA --verify-only
```

`verification_adapter.json` records the compatibility fix separately. Acceptance
repeats raw-evidence/Risk checks, real-batch gradients and full sensor validation.

If `acceptance.json` is already PASS and the final command failed because
`--full` and `--manifest` are mutually exclusive, resume only sensor validation:

```bash
screen -dmS radarvla_moving_finalize bash scripts/run_moving_finalization.sh
```

This entrypoint checks acceptance, manifest, source, configuration and original
implementation hashes before calling `validate --manifest`. Status includes
`phase=sensor_validation`; `COMPLETE` requires a matching full `validation.json`.

For experiments explicitly select the dataset, matching config and new output:

```bash
DATA=/home/luxinlan/data/lxl_data/RadarVLA
bash scripts/radarvla.sh train-grounding --gpus 1,3 \
  --manifest "$DATA/prepared/taskwise_moving_v1/manifest.jsonl" \
  --config configs/l40s_qwen25_3b_taskwise_moving_v1.json \
  --output "$DATA/model_runs/taskwise_moving_v1/grounding" --dry-run
bash scripts/radarvla.sh train-sft --gpus 1,3 \
  --manifest "$DATA/prepared/taskwise_moving_v1/manifest.jsonl" \
  --config configs/l40s_qwen25_3b_taskwise_moving_v1.json \
  --init-grounding "$DATA/model_runs/taskwise_moving_v1/grounding/best.pt" \
  --output "$DATA/model_runs/taskwise_moving_v1/sft" --dry-run
```

Remove `--dry-run` to intentionally launch training. Regenerate experiment plans
for new manifests because they bind a data fingerprint. Legacy default discovery
searches `prepared/full_v3_*`; named releases require explicit `--manifest`.
New `prepare` commands use the motion/taskwise policy and its matching config;
historical labels require explicit `--legacy-labels`. Legacy preparation and
training notes below apply to their original versions; motion Risk and background
rules above supersede historical full-world Risk/ignore policies.

Run these commands from any working directory. Set `PROJECT` once; the script resolves the project root.

```bash
PROJECT=/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA
bash "$PROJECT/scripts/radarvla.sh" doctor
bash "$PROJECT/scripts/radarvla.sh" status
```

## Parameters and storage

| Change | File |
| --- | --- |
| Conda Python, raw/prepared data root, Qwen weights, model output root | `configs/local_paths.json` |
| Route XML, official scenario annotations, town, weather, seed, quotas, CARLA and radar runtime | `collection/configs/collection_10k.yaml` |
| RadarVLA input, planner and training model settings | `configs/l40s_qwen25_3b.json` |
| Stage batch size, GPU ranks, epochs, workers | `launch_l40s.sh` environment variables |

The local Python is `/home/luxinlan/.conda/envs/lxl_radarvla/bin/python`. Editable installation after source changes:

```bash
cd "$PROJECT"
/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
```

`doctor` checks the installed package path, configs, Qwen files and data root. `status` reads the current batch completion marker, accepted count and final radar index count without starting a process. A missing `status.json` or `dataset_index.jsonl` means full data preparation must wait.

## Collection and raw products

The batch collector launches and stops its own CARLA service. **Do not run a separate UE4 shell script.** These commands start new work only when invoked; project reorganization does not restart the existing batch.

```bash
bash "$PROJECT/scripts/radarvla.sh" collect-smoke
bash "$PROJECT/scripts/radarvla.sh" collect-full --resume
```

The collection entry processes each episode through CARLA capture, anchor selection, and C-Shenron K-Radar DRA/RA plus dense Doppler generation. Inspect `<data_root>/full/status.json`, `<data_root>/full/dataset_index.jsonl`, and `<data_root>/full/jobs/<job_id>/collect.log` or `radar/<window>/radar.log` on failure. The raw collection contract is [DATA_COLLECTION_SPEC.md](docs/collection/DATA_COLLECTION_SPEC.md); collector details are in [RAW_COLLECTION_RUNBOOK.md](docs/collection/RAW_COLLECTION_RUNBOOK.md).

For the current batch, `collection_remaining_status.json` already reports filled quotas. If manually invoked, `collect-full --resume` continues only its unfinished radar/final-index stage; it does not recollect those episodes. A brand-new full batch still requires the batch collector's real Scenario2 pilot certificate. Do not launch a second resume process while an existing radar process is active. `doctor` lists missing town maps and route/scenario files before any new collection starts.

## Training preparation and validation

```bash
bash "$PROJECT/scripts/radarvla.sh" prepare --sample 64
bash "$PROJECT/scripts/radarvla.sh" validate --sample
bash "$PROJECT/scripts/radarvla.sh" prepare --full
bash "$PROJECT/scripts/radarvla.sh" validate --full
```

If an earlier full preparation has finished and the preparation code has changed, reuse its compatible successful records and retry its exclusions with:

```bash
bash "$PROJECT/scripts/radarvla.sh" prepare --full --legacy-labels --reuse-previous
bash "$PROJECT/scripts/radarvla.sh" validate --full
```

`--reuse-previous` requires the previous `report.json` to say `PREPARED`. It writes an independent, versioned `prepared/full_v3_*/manifest.jsonl` and copies its radar cache; it never edits the earlier result. Previously excluded anchors are rebuilt from the raw episode. An incompatible prior record is also rebuilt. The new `report.json` lists `reused_rows`, `fresh_rows`, `excluded_rows`, and `reuse_source`. A repeated command resumes the same new output. Do not combine this flag with `--sample`.

The `validate --full` and default training commands select the newest completed full manifest automatically, so this new manifest needs **no downstream code or path edits**. Run `validate --full` again after preparing because validation is tied to the exact manifest hash. Use training `--manifest /absolute/path/to/manifest.jsonl` only if you intentionally want a different version. Before training, confirm `status` reports the expected `full_manifest`.

`prepare --sample` uses only completed radar windows and selects across split, category, and weather where available. It is independent of the final batch marker. `prepare --full` requires `full/status.json` with `RAW_RADAR_COMPLETE` and exactly 10,000 rows in `full/dataset_index.jsonl`; it stops with the missing path/status otherwise. Neither mode modifies `full/jobs/`. A passing `validate` command writes `validation.json` with the exact manifest SHA-256. Real training requires the matching full validation report; dry-run does not.

Outputs are versioned `prepared/sample_v3_*/` and `prepared/full_v3_*/` with `manifest.jsonl`, `exclusions.jsonl`, `report.json`, cached per-frame RA arrays, and per-anchor Doppler reliability masks. A selected raw anchor can be excluded for missing four causal radar frames, incomplete real ego future, unsourced road geometry/width, unresolved Risk teachers, or frame mismatch. All currently visible agents remain in Grounder detection supervision and all relevant world actors remain in scene Risk; `planner_agent_ids` selects the nearest eight visible agents for the structured planner output. A hazardous sample whose responsible actor is outside those eight is excluded rather than relabeled safe. The report's `prepared_rows` is the training candidate count; **10,000 raw anchors are not 10,000 validated training samples**. `validate` loads every record and sensor array and checks split isolation. Inspect exclusions and report before training. The v3 manifest uses left-positive ego coordinates, with a separate ordered road segment and scalar width for each route-connected road/section. Numeric road IDs remain provenance, not prediction targets.

Preparation writes a per-candidate `progress.json` journal plus partial JSONL files. Repeating the same `prepare` command resumes an interrupted matching source/code version; repeating it after completion returns the existing report. Source/code changes create a new versioned directory. Never remove a partial directory just to resume it.
Full preparation reads only the selected future anchors from each episode and keeps at most eight episode product indexes in memory. Its split, weather, category, and coverage counts are accumulated from the manifest stream, including when resuming; it does not load the entire prepared manifest into memory during writing.

The raw episode includes ego, agent snapshots, official route/RoadOption and instruction, real 30-frame futures/masks, RGB and semantic LiDAR. Radar products include power, folded and dense unfolded Doppler, ambiguity/source and reliability. The v3 loader consumes four causal power/dense Doppler frames and the reliable mask; it retains raw sensor paths and provenance in the manifest. Unreliable dense estimates remain observations, not velocity GT.

## Training commands

First inspect commands without launching training:

```bash
nvidia-smi
bash "$PROJECT/scripts/radarvla.sh" train-grounding --gpus 1,3 --dry-run
bash "$PROJECT/scripts/radarvla.sh" train-sft --gpus 1,3 --dry-run
```

`--gpus` accepts physical GPU indices separated by commas. For example, `--gpus 1,3` sets `CUDA_VISIBLE_DEVICES=1,3` and starts two DDP processes; local ranks 0 and 1 map to those cards. Pick cards with enough free memory using `nvidia-smi`; the launcher does not reserve GPUs or infer a safe memory threshold. If `--gpus` is omitted, an existing `CUDA_VISIBLE_DEVICES` selection is respected and determines the process count. With neither set, the legacy default is eight GPUs. Training and per-epoch validation show rank-0 `tqdm` bars with batch progress and ETA; other ranks stay quiet.

After `validate --full` succeeds, run Grounding, then SFT with its best Grounding checkpoint. Actual training is never launched automatically:

```bash
bash "$PROJECT/scripts/radarvla.sh" train-grounding --gpus 1,3
bash "$PROJECT/scripts/radarvla.sh" train-sft --gpus 1,3
```

For an interrupted stage, use the same command with `--resume`. Keep the **same number of selected GPUs** and training settings when resuming; GPU indices may change if the count remains the same. A different rank count changes the training protocol and requires a new run directory. Optional `--manifest`, `--output`, and `--init-grounding` select explicit paths. The launcher is `launch_l40s.sh`; `BATCH_SIZE`, `ACCUMULATION_STEPS`, `EPOCHS`, and `WORKERS` can be exported before invoking `radarvla.sh`. Direct use of `launch_l40s.sh` also accepts `CUDA_VISIBLE_DEVICES`; an explicit `NPROC_PER_NODE` must match its count. Model runs go under `<run_root>/grounding` and `<run_root>/sft` unless overridden. See [DISTRIBUTED.md](DISTRIBUTED.md).

The segmented road grammar changes the physical tokenizer vocabulary and LONG output structure; checkpoints trained on the old single-road grammar are incompatible. The raw v5 pipeline and Risk formula are unchanged. Unknown risk, agent future, or road width must be handled via explicit exclusion or a documented supervision policy; a passing raw/radar status alone is not an SFT release.

**SFT coverage rule:** v5 records full world actor snapshots at every frame. The adapter checks all 30 following snapshots and each current actor's real future; a new or missing actor makes later coverage invalid. It also includes every actor that could affect the existing 80 m Risk distance cap, using a conservative motion bound. This produces `tracking_coverage_valid` without changing the Risk formula or inventing negative labels. The real `train-sft` command stops before loading Qwen if the prepared manifest has no known risk teacher; `--dry-run` still prints the intended command. Check the sample's valid Risk masks before treating the dataset as SFT ready.
