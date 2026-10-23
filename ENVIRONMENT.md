# RadarVLA environment

The project root is `RadarVLA/`; source is in `src/` and imports as `radar_vla`.
This machine uses the existing Conda environment
`/home/luxinlan/.conda/envs/lxl_radarvla`. There is no project-local `.conda`.
Collection uses separate CARLA and C-Shenron interpreters specified by
`collection/configs/collection_10k.yaml`.

After changing model source or packaging, reinstall the local package without
downloading dependencies:

```bash
cd /home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA
/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pip install \
  --no-index --no-deps --no-build-isolation -e .
bash scripts/radarvla.sh doctor
```

`doctor` checks that `python -I` imports `radar_vla` from `RadarVLA/src/`,
which catches a stale editable install pointing at the former physical
`radar_vla/` directory. Machine paths live in `configs/local_paths.json`.
The command reference is [RUNBOOK.md](RUNBOOK.md).

## Local data storage

The unified data directory is `/home/luxinlan/data/lxl_data/RadarVLA`.
It contains `full/`, `prepared/`, `analysis/`, `model_runs/`, `weights/`,
`visualizations/` and a copy of historical `output_json/` metrics.
The project directories `analysis/` and `collection/weights/` are compatibility
symlinks into this data directory. Tracked historical JSON metrics remain in Git.
The current Local Risk manifest is
`/home/luxinlan/data/lxl_data/RadarVLA/analysis/local_agent_risk_v1/manifest.jsonl`,
paired with `configs/l40s_qwen25_3b_localrisk_v1.json`. The Task-1 baseline remains
in `analysis/grounding_agent50_v1/`. See [Local Risk migration](docs/LOCAL_RISK_MIGRATION.md).
See `STORAGE_LAYOUT.md` and `storage_relocation.json` in the data directory for
the directory map and verified file hashes. Existing `/storage/data/...` paths
resolve to the same storage; manifest contents and their hashes are preserved.
