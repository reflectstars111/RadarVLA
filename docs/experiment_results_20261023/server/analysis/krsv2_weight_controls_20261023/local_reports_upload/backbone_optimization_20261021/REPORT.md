# Backbone optimization progress

Grounding only; five epochs per arm, seed42, effective batch8, fixed epoch5 primary comparison. New runs use LR=1e-4; baseline LR=3e-4. No test-set evaluation or Planner training.

| Arm | Epoch | Train loss | Val loss | Recall | FP | Precision | Pcol3 AP | dmin MAE | areq MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hg_lr1e4_b1 | 1 | 1.5437 | 1.7729 | 0.2672 | 192 | 0.7833 | 0.0887 | 13.8845 | 0.1033 |
| hg_lr1e4_b1 | 2 | 1.0515 | 1.7587 | 0.3231 | 405 | 0.6744 | 0.0664 | 12.4281 | 0.0617 |
| hg_lr1e4_b1 | 3 | 0.7946 | 2.0647 | 0.3034 | 126 | 0.8621 | 0.0139 | 17.7360 | 0.2334 |
| hg_lr1e4_b1 | 4 | 0.6214 | 2.1917 | 0.3096 | 275 | 0.7451 | 0.0160 | 16.6848 | 0.1648 |
| hg_lr1e4_b1 | 5 | 0.4988 | 2.0010 | 0.2876 | 225 | 0.7685 | 0.0175 | 15.2261 | 0.0597 |
| hg_lr1e4_b4 | 1 | 1.8907 | 2.4962 | 0.3150 | 717 | 0.5329 | 0.0814 | 14.2475 | 0.1855 |
| hg_lr1e4_b4 | 2 | 1.2868 | 1.6808 | 0.3007 | 253 | 0.7553 | 0.1075 | 10.0853 | 0.0907 |
| hg_lr1e4_b4 | 3 | 0.8782 | 1.6282 | 0.3835 | 699 | 0.5876 | 0.1311 | 8.7095 | 0.0807 |
| hg_lr1e4_b4 | 4 | 0.6783 | 1.7999 | 0.3824 | 452 | 0.6872 | 0.1611 | 10.3199 | 0.0820 |
| hg_lr1e4_b4 | 5 | 0.5628 | 1.6415 | 0.3997 | 385 | 0.7294 | 0.1482 | 8.3755 | 0.0737 |
| trodnet_lr1e4_b1 | 1 | 1.6418 | 1.7375 | 0.2919 | 472 | 0.6163 | 0.1007 | 13.8387 | 0.0983 |
| trodnet_lr1e4_b1 | 2 | 1.1076 | 1.5268 | 0.2564 | 450 | 0.5968 | 0.0882 | 11.2853 | 0.0634 |
| trodnet_lr1e4_b1 | 3 | 0.8940 | 1.6041 | 0.2830 | 976 | 0.4296 | 0.1375 | 11.3040 | 0.0963 |
| trodnet_lr1e4_b1 | 4 | 0.7225 | 1.5276 | 0.3477 | 450 | 0.6674 | 0.1568 | 10.1106 | 0.0939 |
| trodnet_lr1e4_b1 | 5 | 0.5832 | 1.2923 | 0.3939 | 928 | 0.5243 | 0.1537 | 7.3265 | 0.0535 |
| trodnet_lr1e4_b4 | 1 | 1.9640 | 2.0538 | 0.3254 | 541 | 0.6097 | 0.1147 | 13.8533 | 0.1001 |
| trodnet_lr1e4_b4 | 2 | 1.3450 | 1.8595 | 0.1852 | 220 | 0.6862 | 0.0924 | 12.2143 | 0.0610 |
| trodnet_lr1e4_b4 | 3 | 1.1303 | 1.7396 | 0.3700 | 1212 | 0.4422 | 0.1019 | 11.1048 | 0.1085 |
| trodnet_lr1e4_b4 | 4 | 0.8961 | 1.8780 | 0.3419 | 559 | 0.6137 | 0.1324 | 11.4549 | 0.1060 |
| trodnet_lr1e4_b4 | 5 | 0.6955 | 1.7244 | 0.3654 | 474 | 0.6669 | 0.1122 | 10.5804 | 0.0526 |
| baseline_cnn2d | 1 | 1.8122 | 2.0770 | 0.1205 | 138 | 0.6940 | 0.1244 | 12.7587 | 0.0837 |
| baseline_cnn2d | 2 | 1.3665 | 1.9439 | 0.1906 | 332 | 0.5985 | 0.1148 | 12.2431 | 0.0890 |
| baseline_cnn2d | 3 | 1.2036 | 1.9345 | 0.2476 | 501 | 0.5621 | 0.1056 | 12.7907 | 0.0940 |
| baseline_cnn2d | 4 | 1.0760 | 2.0503 | 0.2545 | 587 | 0.5296 | 0.1023 | 13.6980 | 0.1193 |
| baseline_cnn2d | 5 | 0.9705 | 1.9793 | 0.2095 | 534 | 0.5046 | 0.1033 | 12.3172 | 0.0906 |
| baseline_rodnet_hg | 1 | 1.5832 | 1.8529 | 0.1194 | 49 | 0.8635 | 0.0955 | 13.1242 | 0.0786 |
| baseline_rodnet_hg | 2 | 0.9956 | 1.3839 | 0.2002 | 109 | 0.8267 | 0.1150 | 8.0970 | 0.1043 |
| baseline_rodnet_hg | 3 | 0.6777 | 1.7107 | 0.3966 | 574 | 0.6421 | 0.1288 | 11.6997 | 0.1124 |
| baseline_rodnet_hg | 4 | 0.5378 | 1.6879 | 0.2873 | 327 | 0.6952 | 0.1449 | 11.5018 | 0.1249 |
| baseline_rodnet_hg | 5 | 0.4530 | 1.9221 | 0.2345 | 176 | 0.7758 | 0.1505 | 11.2650 | 0.0867 |
| baseline_trodnet | 1 | 1.6072 | 1.7064 | 0.2079 | 190 | 0.7397 | 0.0781 | 12.7880 | 0.0650 |
| baseline_trodnet | 2 | 1.1111 | 1.5077 | 0.1625 | 183 | 0.6975 | 0.1040 | 9.4424 | 0.0806 |
| baseline_trodnet | 3 | 0.8561 | 1.2921 | 0.3978 | 1146 | 0.4741 | 0.1821 | 8.2965 | 0.0940 |
| baseline_trodnet | 4 | 0.7076 | 1.3119 | 0.3061 | 1527 | 0.3424 | 0.1562 | 8.0879 | 0.0707 |
| baseline_trodnet | 5 | 0.6252 | 1.5770 | 0.2580 | 1771 | 0.2745 | 0.3420 | 9.4048 | 0.0716 |

Common matched set currently includes 155 objects across: hg_lr1e4_b1, hg_lr1e4_b4, trodnet_lr1e4_b1, trodnet_lr1e4_b4, baseline_cnn2d, baseline_rodnet_hg, baseline_trodnet.

| Arm | Critical detected/total | Common speed error m/s | Common radial error m/s | Common future ADE m |
| --- | ---: | ---: | ---: | ---: |
| hg_lr1e4_b1 | 0/35 | 1.1307 | 1.7206 | 1.9924 |
| hg_lr1e4_b4 | 20/35 | 0.8402 | 1.0423 | 1.8188 |
| trodnet_lr1e4_b1 | 22/35 | 1.0824 | 1.4075 | 2.0996 |
| trodnet_lr1e4_b4 | 8/35 | 1.0082 | 1.3025 | 1.7625 |
| baseline_cnn2d | 5/35 | 1.4255 | 1.5961 | 2.5995 |
| baseline_rodnet_hg | 8/35 | 1.1766 | 1.2967 | 2.4444 |
| baseline_trodnet | 20/35 | 1.2222 | 1.4128 | 2.2791 |

Real batch4 changes BatchNorm statistics and native masked-loss normalization; it is not a pure BN ablation. Moving-v1 treats stationary vehicles, pedestrians and cyclists as background. Compare detection, motion and Risk together. Common-match sets are provisional until all audits finish. One seed is exploratory evidence. Epoch wall time is affected by shared storage and resource yielding.

FP threshold diagnostic: see trodnet_fp_audit/fp_diagnostic.json; thresholds do not alter primary evaluation.
