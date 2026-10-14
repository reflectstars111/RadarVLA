# K-Radar sequence 1: dense Doppler unfolding A

This directory contains one 50-frame, native RA `[Power, dense Doppler]` diagnostic for radar stems 00281–00330. Each frame is `[2, 256, 107]`; all cells retain finite velocity candidates. Power is copied exactly from the established visualization cache. The velocity channel is a pose-assisted fold-order hypothesis, not an oracle or validated VLA input.

## Selected settings

- Compare matching lags 1, 2, and 3 on the same interior frames; use first-half frames 3–24 for selection and frames 25–46 only for reporting. Sequence edges use the nearest available lag.
- Selected lag: **2 frames** on each available side. Selected reliable-QC score margin: **0.45**. The dense A tensor is byte-for-byte identical to the accepted prior version; this threshold changes QC only.
- Selection requires at least 500 stable static pseudo-reference anchors, at least 90% agreement on them, and no more than a 2 percentage point loss of dense pseudo-reference agreement against lag 2. Among eligible choices, maximize reliable coverage at −48 dB, then prefer shorter lag.
- Strong-power cuts of −48, −40, and −35 dB are reporting strata, not gates on A. Weak cells remain in A. Matching power is clipped to per-frame 0.1–99.9% `log1p` quantiles; cells are not removed.
- On the common selection frames, train stable-anchor agreement is **90.34%** and held-out descriptive agreement is **91.25%**. Across all 50 frames, reliable QC covers **5.84%** of cells; −48 dB strong-cell reliable coverage is **13.17%**.

The static pseudo-reference and pose-consensus QC share odometry and do not independently establish fold-order accuracy. Dynamic targets and weak cells have no independent true fold order. LiDAR label timestamps proxy radar times. The QC mask is not part of the two-channel A input.

## Files and reproduction

- `build_dense_abstain_unfolding.py`, `dense_abstain_support.py`, and `static_anchor_candidates.csv`: A generation and its required support. The source K-Radar tensors and external visualization cache are read-only inputs.
- `select_dense_abstain_params.py`: first-half-only parameter comparison. The full sweep and chosen parameters are recorded in `diagnostics/dense_abstain_unfolding_v1/parameter_selection.json`.
- `diagnostics/dense_abstain_unfolding_v1/version_a_50x2x256x107.npy`: dense A tensor. `version_a/` contains per-frame copies; `version_b_qc/` holds separate reliability diagnostics.
- `visualize_dense_abstain_unfolding.ipynb`, `visualize_dense_abstain_unfolding.py`, and `ra_frames/` under the diagnostic directory: native RA visualization with fixed 50-frame Power and velocity scales.
- `summary.json`, `results.json`, and the diagnostic README contain counts, provenance, and limitations. `cleanup_manifest.json` records the removed legacy files.

From this directory, reproduce the selected product with:

```bash
python3 build_dense_abstain_unfolding.py --match-lag 2 --margin-threshold 0.45
python3 summarize_dense_abstain_unfolding.py
python3 visualize_dense_abstain_unfolding.py
python3 -m pytest -q test_build_dense_abstain_unfolding.py
```

The generator requires the same original 50-frame cache and source K-Radar files referenced by `dense_abstain_support.py`.
