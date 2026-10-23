# RadarVLA 完整验证集诊断包

日期：2026-10-18。先读 REPORT.md；完整1472帧用于检测/Risk，合法1461帧用于规划及相关诊断。数据来自现有CARLA/C-Shenron仿真采集。

- summary.json：全量/分组统计；samples.csv：精简逐帧表；samples.jsonl：原始完整规划、风险、检测、物理与耗时结果。
- full_grounding_summary.json 及 *_grounding_samples.jsonl：完整1472帧检测/Risk；grounding_best_detection_samples.jsonl 为 Stage1 对照。
- generated_semantics_*：生成他车/道路检查；doppler_consistency_*：canonical雷达关联的物理一致性。
- protocol.json / provenance/：模型、数据、源码、脚本摘要、分片和移卡记录；environment.json：实际软件环境。
- training_*_metrics.jsonl 与 source_project_spec.md：既有训练记录/项目审计，不是本轮重训结果。
- ARTIFACTS_SHA256.json：此包文件摘要。

服务器结果根目录：/home/luxinlan/data/lxl_data/RadarVLA/analysis/full_val_diagnosis_20261018
项目源码：/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA
环境：/home/luxinlan/.conda/envs/lxl_radarvla/bin/python

评估脚本在本目录保存；需要服务器上的当前项目包、权重及数据才能重跑神经推理。汇总入口：

```bash
cd /home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/luxinlan/.conda/envs/lxl_radarvla/bin/python \
  /home/luxinlan/data/lxl_data/RadarVLA/analysis/full_val_diagnosis_20261018/summarize_full_validation.py \
  --directory /home/luxinlan/data/lxl_data/RadarVLA/analysis/full_val_diagnosis_20261018/e5_complete_validation
```

本轮未训练、未修改既有模型源码/权重、未评价test，也未建立闭环/独占时延结论。主候选固定为SFT epoch5 best_planning.pt；其他SFT候选未做完整自由生成对比。
