# Backbone optimization progress

Grounding only; five epochs per arm, seed42, effective batch8, fixed epoch5 primary comparison. All compared arms use LR=1e-4 and real batch4/accumulation2. New arms unify eligible anchors; baselines use legacy coverage. No test-set evaluation or Planner training.

| Arm | Epoch | Train loss | Val loss | Recall | FP | Precision | Pcol3 AP | dmin MAE | areq MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hg_sector_lr1e4_b4 | 1 | 1.9047 | 2.4103 | 0.2799 | 496 | 0.5944 | 0.0855 | 14.4427 | 0.2009 |
| hg_sector_lr1e4_b4 | 2 | 1.1935 | 1.7865 | 0.3635 | 598 | 0.6122 | 0.1561 | 8.3354 | 0.0701 |
| hg_sector_lr1e4_b4 | 3 | 0.8387 | 1.6416 | 0.3543 | 337 | 0.7319 | 0.1474 | 8.7753 | 0.0701 |
| hg_sector_lr1e4_b4 | 4 | 0.6884 | 1.9025 | 0.3489 | 473 | 0.6570 | 0.1223 | 8.8648 | 0.0827 |
| hg_sector_lr1e4_b4 | 5 | 0.5732 | 1.4886 | 0.3612 | 229 | 0.8038 | 0.1508 | 6.4477 | 0.0716 |
| trodnet_sector_lr1e4_b4 | 1 | 1.9780 | 2.2520 | 0.2522 | 245 | 0.7278 | 0.1054 | 13.8418 | 0.0981 |
| trodnet_sector_lr1e4_b4 | 2 | 1.3766 | 1.9018 | 0.1663 | 77 | 0.8487 | 0.0773 | 12.3275 | 0.0577 |
| trodnet_sector_lr1e4_b4 | 3 | 1.0880 | 1.8568 | 0.2507 | 158 | 0.8047 | 0.0652 | 11.5100 | 0.0953 |
| trodnet_sector_lr1e4_b4 | 4 | 0.8052 | 1.7879 | 0.3566 | 237 | 0.7962 | 0.1309 | 10.3392 | 0.0768 |
| trodnet_sector_lr1e4_b4 | 5 | 0.6698 | 1.5563 | 0.3881 | 405 | 0.7134 | 0.1156 | 7.7282 | 0.0625 |
| legacy_hg_lr1e4_b4 | 1 | 1.8907 | 2.4962 | 0.3150 | 717 | 0.5329 | 0.0814 | 14.2475 | 0.1855 |
| legacy_hg_lr1e4_b4 | 2 | 1.2868 | 1.6808 | 0.3007 | 253 | 0.7553 | 0.1075 | 10.0853 | 0.0907 |
| legacy_hg_lr1e4_b4 | 3 | 0.8782 | 1.6282 | 0.3835 | 699 | 0.5876 | 0.1311 | 8.7095 | 0.0807 |
| legacy_hg_lr1e4_b4 | 4 | 0.6783 | 1.7999 | 0.3824 | 452 | 0.6872 | 0.1611 | 10.3199 | 0.0820 |
| legacy_hg_lr1e4_b4 | 5 | 0.5628 | 1.6415 | 0.3997 | 385 | 0.7294 | 0.1482 | 8.3755 | 0.0737 |
| legacy_trodnet_lr1e4_b4 | 1 | 1.9640 | 2.0538 | 0.3254 | 541 | 0.6097 | 0.1147 | 13.8533 | 0.1001 |
| legacy_trodnet_lr1e4_b4 | 2 | 1.3450 | 1.8595 | 0.1852 | 220 | 0.6862 | 0.0924 | 12.2143 | 0.0610 |
| legacy_trodnet_lr1e4_b4 | 3 | 1.1303 | 1.7396 | 0.3700 | 1212 | 0.4422 | 0.1019 | 11.1048 | 0.1085 |
| legacy_trodnet_lr1e4_b4 | 4 | 0.8961 | 1.8780 | 0.3419 | 559 | 0.6137 | 0.1324 | 11.4549 | 0.1060 |
| legacy_trodnet_lr1e4_b4 | 5 | 0.6955 | 1.7244 | 0.3654 | 474 | 0.6669 | 0.1122 | 10.5804 | 0.0526 |

Common matched set currently includes 596 objects across: hg_sector_lr1e4_b4, trodnet_sector_lr1e4_b4, legacy_hg_lr1e4_b4, legacy_trodnet_lr1e4_b4.

| Arm | Critical detected/total | Common speed error m/s | Common radial error m/s | Common future ADE m |
| --- | ---: | ---: | ---: | ---: |
| hg_sector_lr1e4_b4 | 5/35 | 1.1778 | 1.1424 | 2.4236 |
| trodnet_sector_lr1e4_b4 | 23/35 | 1.1420 | 1.3287 | 2.2194 |
| legacy_hg_lr1e4_b4 | 20/35 | 1.0611 | 1.0609 | 2.2085 |
| legacy_trodnet_lr1e4_b4 | 8/35 | 1.1506 | 1.2380 | 2.1469 |

Batch4, seed, LR, data, budget and scoring rules are fixed. Raw physical matching now uses eligible anchors; use decoded common-object motion errors for paired comparison. Moving-v1 treats stationary vehicles, pedestrians and cyclists as background. Compare detection, motion and Risk together. Common-match sets are provisional until all audits finish. One seed is exploratory evidence. Epoch wall time is affected by shared storage and resource yielding.
