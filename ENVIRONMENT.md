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
