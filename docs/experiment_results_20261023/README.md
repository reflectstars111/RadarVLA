# 各阶段实验结果归档

快照时间（北京时间）：2026-10-23T10:28:37.005381+08:00

本目录保存服务器 analysis/ 与 model_runs/ 中的实验文本结果，包括报告、配置、运行状态、逐轮指标、逐样本预测和梯度记录。

大于等于 128 KiB 的文本采用 gzip 无损压缩；INVENTORY.json 记录原文件与压缩文件的 SHA-256、大小和来源。

**这是历史快照，不表示所有阶段都成功完成。** 中断、失败、进行中的状态原样保留；旧报告中的计划可能已被后续决定取代（例如暂缓三 seed、KRS 窗口改为 0.5/1/2 秒）。

未包含原始数据、数据集 manifest、环境、模型权重、终端日志和二进制图片；具体目录及文件排除见清单。

| 来源 | 阶段 | 文件数 | 压缩后 MiB | 报告入口 |
|---|---|---:|---:|---|
| analysis | [backbone_adaptation_20261020](server/analysis/backbone_adaptation_20261020/) | 36 | 0.40 | [报告](server/analysis/backbone_adaptation_20261020/REPORT.md) |
| analysis | [backbone_optimization_20261021](server/analysis/backbone_optimization_20261021/) | 43 | 0.57 | [报告](server/analysis/backbone_optimization_20261021/REPORT.md) |
| analysis | [backbone_training_20261020](server/analysis/backbone_training_20261020/) | 25 | 0.36 | [报告](server/analysis/backbone_training_20261020/REPORT.md) |
| analysis | [boundary_audit_20261021](server/analysis/boundary_audit_20261021/) | 14 | 0.78 | [报告](server/analysis/boundary_audit_20261021/REPORT.md) |
| analysis | [candidate_contract_20261021](server/analysis/candidate_contract_20261021/) | 7 | 0.05 | [报告](server/analysis/candidate_contract_20261021/REPORT.md) |
| analysis | [candidate_training_20261021](server/analysis/candidate_training_20261021/) | 15 | 0.29 | [报告](server/analysis/candidate_training_20261021/REPORT.md) |
| analysis | [current_pipeline_audit_20261016](server/analysis/current_pipeline_audit_20261016/) | 27 | 0.90 | [报告](server/analysis/current_pipeline_audit_20261016/REPORT.md) |
| analysis | [full_val_diagnosis_20261018](server/analysis/full_val_diagnosis_20261018/) | 63 | 8.52 | [报告](server/analysis/full_val_diagnosis_20261018/REPORT.md) |
| analysis | [grounding_agent50_v1](server/analysis/grounding_agent50_v1/) | 15 | 3.15 | [报告](server/analysis/grounding_agent50_v1/REPORT.md) |
| analysis | [hparam_20261021](server/analysis/hparam_20261021/) | 651 | 27.86 | [报告](server/analysis/hparam_20261021/REPORT.md) |
| analysis | [krsv2_baseline_20261022](server/analysis/krsv2_baseline_20261022/) | 77 | 13.43 | [报告](server/analysis/krsv2_baseline_20261022/REPORT.md) |
| analysis | [krsv2_weight_audit_20261022](server/analysis/krsv2_weight_audit_20261022/) | 5 | 0.24 | [报告](server/analysis/krsv2_weight_audit_20261022/REPORT.md) |
| analysis | [krsv2_weight_controls_20261023](server/analysis/krsv2_weight_controls_20261023/) | 201 | 29.44 | [报告](server/analysis/krsv2_weight_controls_20261023/local_reports_upload/oracle_risk_distillation_20261019/REPORT.md) |
| analysis | [local_agent_risk_v1](server/analysis/local_agent_risk_v1/) | 14 | 1.90 | [报告](server/analysis/local_agent_risk_v1/REPORT.md) |
| analysis | [long_training_20261021](server/analysis/long_training_20261021/) | 195 | 8.55 | [报告](server/analysis/long_training_20261021/REPORT.md) |
| analysis | [oracle_risk_distillation_20261019](server/analysis/oracle_risk_distillation_20261019/) | 28 | 0.29 | [报告](server/analysis/oracle_risk_distillation_20261019/REPORT.md) |
| analysis | [oracle_risk_pcol3_teacher_20261019](server/analysis/oracle_risk_pcol3_teacher_20261019/) | 11 | 0.10 | [报告](server/analysis/oracle_risk_pcol3_teacher_20261019/REPORT.md) |
| analysis | [rain_fog_10k_v1](server/analysis/rain_fog_10k_v1/) | 57 | 0.28 | 见目录中的指标与状态 |
| analysis | [range_control_20261019](server/analysis/range_control_20261019/) | 21 | 3.09 | [报告](server/analysis/range_control_20261019/REPORT.md) |
| analysis | [rebalanced_50m_domain](server/analysis/rebalanced_50m_domain/) | 2 | 0.00 | 见目录中的指标与状态 |
| analysis | [rebalanced_detection](server/analysis/rebalanced_detection/) | 6 | 0.68 | 见目录中的指标与状态 |
| analysis | [rebalanced_v4](server/analysis/rebalanced_v4/) | 3 | 0.01 | 见目录中的指标与状态 |
| analysis | [rebalanced_validation](server/analysis/rebalanced_validation/) | 8 | 0.19 | 见目录中的指标与状态 |
| analysis | [retrained_grounding](server/analysis/retrained_grounding/) | 2 | 1.15 | 见目录中的指标与状态 |
| analysis | [retrained_grounding_v2](server/analysis/retrained_grounding_v2/) | 3 | 1.17 | 见目录中的指标与状态 |
| analysis | [retrained_sft_v2](server/analysis/retrained_sft_v2/) | 3 | 1.17 | 见目录中的指标与状态 |
| analysis | [risk_core_diagnosis_20261022](server/analysis/risk_core_diagnosis_20261022/) | 22 | 0.59 | [报告](server/analysis/risk_core_diagnosis_20261022/REPORT.md) |
| analysis | [risk_horizons_20261022](server/analysis/risk_horizons_20261022/) | 11 | 0.10 | [报告](server/analysis/risk_horizons_20261022/REPORT.md) |
| analysis | [risk_input_followup_20261018](server/analysis/risk_input_followup_20261018/) | 69 | 4.07 | [报告](server/analysis/risk_input_followup_20261018/REPORT.md) |
| analysis | [risk_layer_diagnosis_20261019](server/analysis/risk_layer_diagnosis_20261019/) | 51 | 1.17 | [报告](server/analysis/risk_layer_diagnosis_20261019/REPORT.md) |
| analysis | [taskwise_local_v1](server/analysis/taskwise_local_v1/) | 12 | 0.56 | [报告](server/analysis/taskwise_local_v1/REPORT.md) |
| analysis | [taskwise_local_v1_failed_initial_serialization](server/analysis/taskwise_local_v1_failed_initial_serialization/) | 3 | 0.00 | 见目录中的指标与状态 |
| analysis | [taskwise_moving_v1](server/analysis/taskwise_moving_v1/) | 6 | 0.05 | [报告](server/analysis/taskwise_moving_v1/REPORT.md) |
| model_runs | backbone_adaptation_20261020 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | [backbone_optimization_20261021](server/model_runs/backbone_optimization_20261021/) | 12 | 5.28 | 见目录中的指标与状态 |
| model_runs | [backbone_training_20261020](server/model_runs/backbone_training_20261020/) | 9 | 3.96 | 见目录中的指标与状态 |
| model_runs | candidate_contract_20261021 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | [candidate_training_20261021](server/model_runs/candidate_training_20261021/) | 6 | 2.64 | 见目录中的指标与状态 |
| model_runs | [hparam_20261021](server/model_runs/hparam_20261021/) | 34 | 15.73 | 见目录中的指标与状态 |
| model_runs | [krsv2_baseline_20261022](server/model_runs/krsv2_baseline_20261022/) | 3 | 1.42 | 见目录中的指标与状态 |
| model_runs | [krsv2_weight_controls_20261023](server/model_runs/krsv2_weight_controls_20261023/) | 8 | 2.86 | 见目录中的指标与状态 |
| model_runs | [long_training_20261021](server/model_runs/long_training_20261021/) | 12 | 5.49 | 见目录中的指标与状态 |
| model_runs | oracle_risk_distillation_20261019 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | oracle_risk_pcol3_teacher_20261019 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | [original_first1](server/model_runs/original_first1/) | 8 | 2.35 | 见目录中的指标与状态 |
| model_runs | range_control_20261019 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | [risk_controls_20261019](server/model_runs/risk_controls_20261019/) | 12 | 3.94 | 见目录中的指标与状态 |
| model_runs | risk_horizons_20261022 | 0 | 0.00 | 见目录中的指标与状态 |
| model_runs | [risk_layer_diagnosis_20261019](server/model_runs/risk_layer_diagnosis_20261019/) | 1 | 2.32 | 见目录中的指标与状态 |
| model_runs | [taskwise_local_v1_20261017_003057](server/model_runs/taskwise_local_v1_20261017_003057/) | 7 | 2.66 | 见目录中的指标与状态 |

## 读取压缩结果

```python
import gzip, json
with gzip.open('path/to/result.json.gz', 'rt', encoding='utf-8') as f:
    result = json.load(f)
```

JSONL 文件请逐行读取并对每行调用 json.loads。恢复后的字节 SHA-256 应与清单中的 sha256 一致。

## ????

???? Risk ??? 1/2/3 ??risk_horizons_20261022 ??????KRS-v2 ? 0.5/1/2 ??d_min ? a_req ? 2 ??????????????????????????????????Agent ??? 3 ? ADE/FDE ??? Risk ?????

????????? `server/analysis/krsv2_weight_controls_20261023/local_reports_upload/`????????????????????????????????????????????????
