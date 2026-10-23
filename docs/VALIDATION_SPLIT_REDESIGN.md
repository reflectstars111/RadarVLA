# Episode-level validation redesign (2026-10-15)

The previous prepared manifest held 8,835 frames from 64 episodes: 6,961/942/932
frames and 48/6/10 episodes in train/val/test. Its validation set contained
only `normal` Town03 scenes and `turn right` instructions. Seven 3-second
collision-positive frames came from one validation episode.

The new split pools all 64 prepared episodes and assigns whole episodes with
seed 42. The generator in `scripts/rebalance_manifest.py` targets 70/15/15
frames while balancing category, weather, instruction, and the number of
episodes containing a 3-second collision positive. It keeps the only
substantial `scenario4` episode in train. Its input is the completed
`full_v3_44141be5af70/manifest.jsonl`; the original is unchanged.

| Split | Frames | Episodes | Positive episodes | Positive frames |
| --- | ---: | ---: | ---: | ---: |
| Train | 6,166 | 45 | 16 | 486 |
| Validation | 1,337 | 10 | 4 | 56 |
| Test | 1,332 | 9 | 3 | 57 |

The exact episode IDs and category/weather/instruction counts are in
`analysis/rebalanced_v4/split_report.json`. The new manifest is next to that
report. Its sensor paths point to the existing immutable prepared arrays;
the source files are not copied. Validation includes `normal`, `scenario2`,
`scenario3`, `scenario7_9`, and `scenario10`. `scenario4` cannot be both trained
and independently validated with the current collection. `scenario7_9` has
only two substantial episodes and is absent from the new test set. These
coverage gaps require additional independent collection rather than frame
reshuffling.

The old model checkpoints are fingerprinted to the old manifest. New Grounding
and SFT runs must use new output directories and must be assessed on this new
validation split. Previously viewed test results are historical diagnostics,
not a pristine final holdout. Detection thresholds and checkpoint choices
must use validation only.

Old-checkpoint diagnostics are provided by `scripts/diagnose_detection.py`
and `scripts/diagnose_sft_trajectory.py`. They use the original validation
split to locate failures without changing model weights. The former caches
raw candidates and sweeps objectness thresholds; the latter compares a
constant-velocity baseline, oracle token codec, teacher-forced physical tokens,
and free generation on fixed spaced and collision-positive frames.
