# 验证记录

新专项测试在旧实现上先因`ModuleNotFoundError: radar_vla.risk_distillation`失败，随后实现共享风险头入口及蒸馏模块。

```bash
python -m pytest -q tests/test_risk_mode_distillation.py tests/test_radar_vla_model.py \
  tests/test_radar_vla_hf.py tests/test_risk_training_controls.py tests/test_radar_vla_pipeline.py
```

加入真实本地微型Qwen反传检查后：**59 passed in 20.67s**。随后专项整理检查：**9 passed in 1.90s**（为上述子集，不相加）。

- 完整Grounder输出与缓存特征重跑风险头逐位相同。
- KL方向、缺失标签mask、教师stop-gradient及空有效批次正确。
- 真实优化步骤仅更新风险query/attention/MLP；编码器、Planner参数逐位不变。
- 蒸馏损失独立反传到风险query、attention与MLP。
- 真实微型Qwen含LoRA及gradient checkpointing，冻结后仍将梯度传回风险头，Planner权重及缓冲区不变。
- 原模型、原HF路径、训练控制和pipeline回归通过。
- 相关Python脚本编译检查通过；无模型下载。

训练/评价的真实数据一致性、冻结完整摘要和验证集分母在最终报告中单列。模式评价不能代替完整自由轨迹评价。

补充检查：LF换行规范化后原风险蒸馏专项 **9 passed in 2.55s**；新增Pcol3教师专项与原专项合跑 **11 passed in 2.77s**。这些测试集有重叠，不累加为总数。新增测试验证仅替换有效Pcol3、未知制动量仍允许模式蒸馏、缺失Pcol3时不产生KL梯度，以及教师梯度隔离。运行脚本为 `scripts/refine_risk_pcol3_teacher.py`，固定3epoch的设置继承主实验。

最终产物另在CPU逐张量验证：新检查点的Planner导出状态与原文件完全相同；Grounder仅风险参数发生变化。原检查点、manifest、神经网络源码摘要保持不变；验证集样本顺序和标签与主对照逐条一致。完整检查结果以各报告目录的 `completion_checks.json` 为准。
