# RadarVLA 系统超参优化进度与结果


用户最新调整：seed43/44验证暂缓；保留seed42训练，按单seed结果进行后续条件筛选及最终test。不得据此宣称跨seed稳定性。

更新时间：2026-10-22 17:20:22 北京时间。当前阶段：**全部完成**。

本报告自动更新。队列按计划推进；100轮为条件分支。Grounder检测、运动和五维Risk联合训练；未训练Planner/SFT，未做闭环驾驶测试。CUDA反向非确定性已记录，单seed筛选不作最终结论。

隔离源码提交：`c6c1d43a4ea9bd19aa3fe3ab25c0c88c4f8fa3a1`；主源码不合并、不覆盖。

| 实验 | 真实batch/累积 | LR/调度 | seed | 计划轮次 | 状态 | 已完成轮次 | GPU |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| train_17350fc061f691 | 4/2 | 5e-05/flat_cosine | 42 | 20 | complete | 20 | 7 |
| train_a43ed36db270e5 | 4/2 | 0.0001/flat_cosine | 42 | 20 | complete | 20 | 3 |
| train_b6a4817d9da127 | 4/2 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 4 |
| train_c4f92d3bd0fafa | 4/2 | 0.0001/cosine | 42 | 20 | complete | 20 | 5 |
| train_069cc92e1fcc1e | 4/4 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 3 |
| train_5f6f4ed9be64a1 | 8/2 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 5 |
| train_849e0c15a4deac | 16/1 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 7 |
| train_129ef481ed9c66 | 8/1 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 1 |
| train_05b714a3df76de | 8/4 | 0.0002/flat_cosine | 42 | 20 | complete | 20 | 3 |
| train_328dae06859b65 | 4/2 | 0.0002/flat_cosine | 42 | 40 | complete | 40 | 1 |
| train_c80e42275688aa | 4/2 | 0.0001/flat_cosine | 42 | 40 | complete | 40 | 5 |
| test_9ca1bf91b26913 | -/- | -/test | 42 | - | complete | - | 5 |
| val_pr_cf476a95694337 | -/- | -/val_pr | 42 | - | complete | - | 0 |

## 已锁定配方

计划长度：40轮。配方：`{"lr": 0.0002, "schedule": "flat_cosine", "batch": 4, "accumulation": 2}`。仅seed42；seed43/44暂缓。

选择使用验证集Pareto代表与预先声明的Risk优先顺序；检测、运动等取舍保留，不能解释为所有指标全面最好。

## 最终test单seed42结果

| 指标 | mean | stdev | 各seed原值 |
| --- | ---: | ---: | --- |
| f1 | 0.747832 | - | 0.747832 |
| critical | 0.491228 | - | 0.491228 |
| dmin | 2.907920 | - | 2.907920 |
| areq | 0.067664 | - | 0.067664 |
| velocity | 0.965938 | - | 0.965938 |
| radial | 0.986326 | - | 0.986326 |
| ade | 1.962019 | - | 1.962019 |
| ap1 | 0.219740 | - | 0.219740 |
| brier1 | 0.016992 | - | 0.016992 |
| ap2 | 0.351105 | - | 0.351105 |
| brier2 | 0.021865 | - | 0.021865 |
| ap3 | 0.353648 | - | 0.353648 |
| brier3 | 0.029570 | - | 0.029570 |

test仅用于已锁定配方的最终评估，不回流超参选择；未根据test改阈值。运动test均值分别条件于各模型检出集合，不能忽略匹配数量而比较为同一目标集合。

最终checkpoint：
- seed42：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/hparam_20261021/campaign/train_328dae06859b65/attempt_01/last.pt`，SHA256 `2a5154e763aa6724c0e50d683ac76fc9ad0ed317b1233e4a3b4b4324afefbdd6`。

## 主源码保护核验

{"changed_files": ["collection/collector/agent_policy.py", "src/moving_agents.py", "src/moving_migration.py", "RUNBOOK.md", "collection/collector/batch.py", "collection/collector/batch_selection.py", "collection/collector/episode.py", "collection/collector/recorder.py", "docs/collection/DATA_COLLECTION_SPEC.md", "scripts/radarvla_cli.py", "src/collection_adapter.py", "src/data.py", "src/pipeline.py", "tests/test_v3_collection_adapter.py"], "status_equal": false, "protected_file_count": 225}

完整协议和分支条件见EXECUTION_RULES.md；实时状态见campaign_state.json；逐轮原始验证、匹配目标ID、五维Risk及运动数据保存在runs/对应实验目录。在stage=complete且FINAL_TEST.json存在前，不能将队列启动称为全流程完成。
