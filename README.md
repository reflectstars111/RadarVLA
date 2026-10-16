# RadarVLA

`RadarVLA/` is the project and Git root. Model source lives in `src/`, installed as the Python package `radar_vla`. Collection code, configuration, documentation, tests, and entry scripts remain at the project root.

**Start here:** [RUNBOOK.md](RUNBOOK.md) gives current commands, parameters, inputs, outputs, and failure checks. Run `bash scripts/radarvla.sh doctor` from any directory to check the installation.

Current training-quality findings, implemented fixes, checkpoint selection and retraining requirements: [2026-10-16 review](docs/TRAINING_QUALITY_REVIEW_20261016.md).

| Path | Purpose |
| --- | --- |
| `src/` | Model, loader, and collection adapter; import as `radar_vla` |
| `collection/collector/` | CARLA episode and C-Shenron radar batch entry |
| `collection/configs/collection_10k.yaml` | Routes, scenarios, weather, quotas, runtime |
| `configs/l40s_qwen25_3b.json` | Model and planner settings |
| `configs/local_paths.json` | This machine's Python, data, weights, and run paths |
| `scripts/radarvla.sh` | Single operational command entry |
| `docs/collection/` | Raw collection contract, runbook, and reports |
| `tests/`, `collection/tests/` | Model and collection checks |

The current batch stores raw episodes and radar products under `/home/luxinlan/data/lxl_data/RadarVLA/full/`. Training preparation writes a separate versioned `prepared/` directory and does not alter `full/jobs/`.

See [DATA_FORMAT.md](DATA_FORMAT.md) for the research format, [ENVIRONMENT.md](ENVIRONMENT.md) for environment setup, and [DISTRIBUTED.md](DISTRIBUTED.md) for distributed training details. A completed raw/radar index is a prerequisite for full preparation; it does not by itself mean the model dataset has passed validation.
