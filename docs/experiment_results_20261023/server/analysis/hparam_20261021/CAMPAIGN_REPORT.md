# RadarVLA hyperparameter campaign

Stage: **complete**. Updated Beijing UTC+8: 2026-10-22 17:19:42.

Grounder-only; fixed moving-v1, candidate sector, original backbone precision path. No bitwise CUDA reproducibility claim. New update-based LR schedules. User deferred seeds43/44; seed42 only, conditional100 epochs; no repeatability claim. Test is locked until final selection.

| Job | Kind | Seed | Budget | Batch/accum | LR/schedule | State | Epoch | AP3 | F1 |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: |
| train_17350fc061f691 | train | 42 | 20 | 4/2 | 5e-05/flat_cosine | complete | 20 | 0.38018701321561377 | 0.5702247191011236 |
| train_a43ed36db270e5 | train | 42 | 20 | 4/2 | 0.0001/flat_cosine | complete | 20 | 0.23054546734036885 | 0.5994094488188977 |
| train_b6a4817d9da127 | train | 42 | 20 | 4/2 | 0.0002/flat_cosine | complete | 20 | 0.6540558172933186 | 0.5897560975609756 |
| train_c4f92d3bd0fafa | train | 42 | 20 | 4/2 | 0.0001/cosine | complete | 20 | 0.37852779376289886 | 0.6031042128603105 |
| train_069cc92e1fcc1e | train | 42 | 20 | 4/4 | 0.0002/flat_cosine | complete | 20 | 0.4993691256846244 | 0.5461699895068206 |
| train_5f6f4ed9be64a1 | train | 42 | 20 | 8/2 | 0.0002/flat_cosine | complete | 20 | 0.5662643140857846 | 0.5705489994869164 |
| train_849e0c15a4deac | train | 42 | 20 | 16/1 | 0.0002/flat_cosine | complete | 20 | 0.6483237879555596 | 0.5621149897330595 |
| train_129ef481ed9c66 | train | 42 | 20 | 8/1 | 0.0002/flat_cosine | complete | 20 | 0.609094619196477 | 0.5837974683544304 |
| train_05b714a3df76de | train | 42 | 20 | 8/4 | 0.0002/flat_cosine | complete | 20 | 0.2458863856527631 | 0.5597129677088672 |
| train_328dae06859b65 | train | 42 | 40 | 4/2 | 0.0002/flat_cosine | complete | 40 | 0.6737597464896261 | 0.5822537276949401 |
| train_c80e42275688aa | train | 42 | 40 | 4/2 | 0.0001/flat_cosine | complete | 40 | 0.34050380103133737 | 0.5847225678029361 |
| test_9ca1bf91b26913 | test | 42 | - | -/- | -/- | complete | - | - | - |
| val_pr_cf476a95694337 | val_pr | 42 | - | -/- | -/- | complete | - | - | - |

Decisions, common matched-object cohorts and full five-dimensional Risk are retained in campaign_state.json. MULTISEED_COMPARISON.json records40vs20 gating. FINAL_SELECTION_LOCK.json precedes all test evaluation. FINAL_TEST.json is produced only when the entire selected three-seed test evaluation completes. No training/precision optimization benefit is assumed in advance.

Failures: {}
