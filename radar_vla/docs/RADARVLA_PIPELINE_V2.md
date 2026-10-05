# RadarVLA v2 实现与验证记录

日期：2026-10-06（Asia/Shanghai）。本记录替代 v1 的当前实现说明；[v1 记录](RADARVLA_PIPELINE_V1.md) 保留为历史证据。需求依据：[data_record.md](data_record.md)、[RadarVLA_risk_adaptive_plan.md](RadarVLA_risk_adaptive_plan.md)。

## 文档要求与实现映射

| 要求 | 实现文件 | 核验内容 |
|---|---|---|
| 完整 frame_t、camera/LiDAR/radar 资产 | records.py | 实际解码、尺寸/内外参/时间戳校验；元信息和实际资产路径保留 |
| SE(3) 与运动补偿 | coordinates.py、radar_processing.py | 点/向量分开变换、安装杆臂速度、异步实际位姿、因果时窗 |
| agents bbox3d/ID/future；map 全分支 | records.py、data.py | 当前 ego 投影、稳定 ID、未来入场、中心线和两侧边界、traffic elements |
| 联合 RA encoder + temporal queries | model.py | 原生双通道、历史几何与时间编码、Doppler 有效性、完整 signed ego state |
| 反事实风险标签 | geometry.py | CV/CTRV、0.05s 网格、未来覆盖掩膜、最小可行制动完整网格搜索、截尾标签 |
| physical + KRS Stage 1 | model.py、losses.py | 风险 focal/Huber/单调性、对象匹配、实测 Doppler、全时域运动学 |
| 单一 Qwen + continuous risk | hf_planner.py | 本地 Qwen 原生 tokenizer、连续 prefix、LoRA、BF16、gradient checkpointing |
| LONG / SHORT 结构化生成 | planner.py | 实际道路、风险关键目标、变长对象语法、同一模型生成模式 |
| 物理量化与连续 Gaussian 教师 | tokenizer.py | signed-log、未量化 GT 高斯分布、显式范围/预算错误 |
| 轨迹控制点与物理损失 | curves.py、planner.py | 自然三次样条、ego 原点、真实端点拟合、曲线轨迹/道路/Doppler/平滑损失 |
| 离线物理行动评价 | evaluation.py、metrics.py | 沿原样条稠密采样的框碰撞、未来入场/缺失覆盖、TTC、制动、jerk、路线与误差 |
| 指令遵循 | instruction_metrics.py | 明确速度/停车/目标区域/车体朝向约束；缺标签/时域时为 unknown |
| Q1–Q5 可执行协议 | experiments.py、cohort.py | 配对消融、固定预算/数据指纹、oracle 共同比较范围、Doppler 干预 |
| 训练/恢复/无标注推理 | pipeline.py、__main__.py | 两阶段、检查点/优化器/RNG、严格监督预检、观测预测 |
| 独立部署 | pyproject.toml、environment.yml、launch_l40s.sh | pip 可安装包、独立 Conda、8-rank 启动、手动监测 |

具体路径以 radar_vla/ 为相对根。用户数据字段及单位规范见 [DATA_FORMAT.md](../DATA_FORMAT.md)。

## 已去除的旧实现缺口

- 不再使用固定 ROAD_UNKNOWN 替代道路目标；真实道路控制点和宽度进入监督与损失。
- SHORT 依据反事实危险目标筛选，使用物理时间短时域；不再固定选择最近两个作为关键目标。
- 实测 Radar Doppler 与同权重 LOS 投影配套监督；不把标签推导速度冒充实测回波。
- Future trajectory 以样条控制点训练、解码和评价；ego 起点固定原点，不允许独立预测出偏移起点。
- Gaussian 标签中心保留连续真值；不再以量化后中心替代。
- 摄像头、点云、标定、位姿和 3D 框具有实际导入处理；不是仅保留未使用字典键。
- 预测不要求目标、未来轨迹和地图；观测模式不从未来补标签。
- 当前空场景不直接推出未来安全；未来入场目标与 tracking 覆盖影响标签及评价。
- 指令、物理范围、对象容量超过配置时报错，不静默截断或裁剪。

## 实际验证

完整运行产物：

```text
/ssd1/data/RadarVLA/runs/pipeline_full_20261006_v2/
  raw/frames.jsonl                 # 15 帧完整原始记录及实际 RGB/点云/Radar 文件
  data/manifest.jsonl              # 6 个有效 4 帧历史窗口
  data/preparation_report.json     # 9 个历史不足窗口的明确排除记录
  labeled.jsonl                    # 整体/逐对象风险缓存
  cohort.jsonl                     # train/val/test 共同监督协议
  cohort.jsonl.cohort.json          # 完整筛选报告
  grounding/{best,last}.pt
  sft/{best,last}.pt
  test_metrics.json
  predictions.jsonl
  observations.jsonl              # 不含 agents/map/未来 GT 的原始预测输入
  observations/manifest.jsonl
  unannotated_predictions.jsonl    # 2 条无标注推理结果
  smoke_summary.json
  radar_tests.xml
```

以上完成两阶段各一轮、显式测试、标注输入预测和纯观测预测；Radar 原生尺寸 `[2,256,107]`，4 帧历史，语言后端明确选择 tiny 供 CPU 集成验证。其高误差和不稳定动作符合随机小模型的性质，不将 smoke loss/格式合法性写成驾驶质量证据。

全仓库在原项目环境回归：**303 passed、2 skipped、6 warnings**。两个跳过是原医学测试需要的本地数据根目录不可用；6 个警告来自既有 DeepSpeed/Pydantic 兼容层。医学代码和训练进程没有因此更改。

独立 RadarVLA 环境专项 **127 passed，0 skipped**，耗时 21.58 秒，最终结果见 radar_tests.xml；覆盖真实小型 Qwen backbone、LoRA 梯度、adapter 保存加载、BF16、双进程 CPU DDP 与梯度累积，以及完整帧到无标注预测。独立环境 `pip check` 无冲突，模块编译通过，8 卡启动脚本语法及 dry-run 通过。Q4 的 -2/-5/-8m/s 输入干预也已实际执行，结果保存为 doppler_sweep.json。可安装 wheel 已构建于：

```text
/ssd1/data/RadarVLA/dist/radar_vla-0.2.0-py3-none-any.whl
```

源码配置与 checkpoint 均记录内容指纹。v2 的词表、姿态编码及未来速度头不同于 v1，不接受旧 checkpoint 续训。

## 实验解释与未验证范围

- 8×L40S / 每卡 44GB / 完整 Qwen2.5-3B 是目标部署配置，当前无真实数据和完整本地权重，不能声称已经在目标硬件验证显存、吞吐或效果。
- 必需的离线监督主线已实现。方案第 13 节将闭环 RFT 明确标为“可选”，并说明不是基础方案成立的必要条件；该增强及 CARLA 联机闭环尚未实现，当前评价是离线评价。
- 规划/风险采用明确的平面车辆模型。数据接入会拒绝超出此假设的 ego 姿态与竖直运动；传感器外参仍支持三维。
- Q3 oracle 需要同一组五维风险标签均已知的样本，不可行制动/缺失未来样本会被整个比较共同排除。排除原因与比例必须随实验结果报告，不能把子集结论推广为全场景安全。
- without_ego 移除显式运动状态条件，仍使用共同坐标标定与位姿补偿；risk_none 保留共同风险辅助训练，只移除 LLM Risk Token。
- Q4 是固定 Power/指令/其他输入的 Doppler 观测敏感性测试，不声称是重新生成了物理场景。
- 指令遵循依赖可核查约束标注；自由文本本身不是成功真值。未覆盖时域、缺少 route 或对应标注的指标为 null 并附覆盖信息。

未启动长期 Radar 训练，也未配置在线监控。正式运行、恢复和手动监测命令见 [DISTRIBUTED.md](../DISTRIBUTED.md)。
