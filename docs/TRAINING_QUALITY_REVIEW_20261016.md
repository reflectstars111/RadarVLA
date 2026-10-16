# 训练质量审查与修复（2026-10-16）

本次审查基于仓库 `d673d29` 的代码及 `output_json/grounding/test_metrics.json`、`output_json/sft/test_metrics.json`。本机没有报告引用的真实 manifest、训练日志、权重和雷达数组；以下区分已复现的代码缺陷、评估事实和待验证推断。代码修复及合成集测试不能代替真实数据重训结果。

## 现有结果说明了什么

两份报告均为 932 帧、4,063 个目标，平均 4.36 个目标/帧。

| 指标 | Grounding | SFT |
| --- | ---: | ---: |
| 检测 precision | 1.69% | 5.65% |
| 检测 recall | 77.48% | 70.39% |
| 检测 FP | 182,684 | 47,747 |
| 平均输出目标/帧 | 199.39 | 54.30 |
| 目标数量 MAE | 195.03 | 49.94 |
| 检出目标未来 ADE / FDE | 4.21 / 7.39 m | 4.19 / 7.09 m |

SFT 自车轨迹 ADE 为 **7.71 m**，3 秒 FDE 为 **13.09 m**，初速误差为 **4.20 m/s**；离线计划碰撞率为 **18.88%**。932 个样本全部生成 LONG，平均输出约 99 个 token、摊销推理约 2.46 秒。

- `schema_valid_rate=1` 主要由受限语法保证，不能证明轨迹正确。
- 原始 `position_mae=0.63/0.59m` 用 GT 在所有候选中匹配，绕过置信度筛选，不能替代实际 detection precision。
- 三个碰撞风险标签的正例只有 15/16/16 个；风险 Brier 为 .030–.046，比全零预测的约 .016–.017 更差。高 AUROC 只说明排序能力，不能据此认定概率已校准。
- 所有测试目标都标为 vehicle；本次没有行人、骑行者类别的测试证据，也不能由此推断训练集中没有这些类别。
- checkpoint 的 epoch=1/3 只说明保存的是哪个 epoch，不能推出训练总共只跑了 1/3 个 epoch。

## 已修复的原因

### 1. 稠密检测的负例被过度降权

旧 `src/losses.py` 对正例 focal loss 和负例 focal loss 分别求均值再相加。默认 3,456 个候选、4 个 GT 时，单个背景梯度约为单个正例的 1/863；没有区分能力的共享 logit 在概率 .5 时总梯度近乎抵消，与真实稀疏程度无关。

实际调用损失函数的 CPU 反例：4 个 TP 置信度 .9、200 个 FP 置信度 .6、其余背景 .01，旧 objectness loss 只有 **.020166**，尽管 precision 只有约 1.96%。

修复为：全部有效候选的 focal loss 求和，统一除以匹配正例数量（至少 1）。相同反例的损失变为 **16.4951**。空帧仍监督背景；未知标注帧不产生检测监督。保留原 .01 置信度初始化和梯度裁剪，没有提高阈值或增加数量上限来掩盖误报。

这证明目标函数已修正，不能直接宣称真实 checkpoint 的 precision 已提升。

### 2. 标注区域与雷达输入范围不一致

采集器只标注约 50 m、水平 ±53° 内相交目标，雷达仍覆盖约 118 m；旧流程把所有未匹配候选都当成已知背景，评估也把超出标注区域的预测全部计为 FP。

现已从 episode metadata 导出显式 `detection_annotation_domain`：

- 保留完整雷达上下文，仅在已知区域监督背景。
- 与边界相交的 GT 保持正例，即使其中心或分配 anchor 在域外。
- 评估先匹配 GT，仅将域外未匹配预测列为未评分，同时输出 `raw_predictions`、`evaluated_predictions`、`ignored_outside_annotation_domain` 和 `raw_count_mae`。
- 解码仍输出所有检测结果；不能把局部域指标说成完整 118 m 检测能力。BEV 扇区也不等价于采集的完整三维垂向视域。
- 任意旧 manifest 不会被猜测为 50 m 标注。必须重新准备原始数据；旧缓存契约失效时从 raw 重建。

没有逐目标预测记录，目前无法估计已有 FP 中多少来自域外。新的局部域分数与旧全范围分数也不应直接当作同口径提升。

### 3. 自车状态条件与初速约束缺失

旧雷达 token 不随自车加速度、角速度改变，规划器只能从风险五维输出间接获得这些信息；部分速度信息虽存在于雷达几何编码中，却不是完整的自车状态条件。关闭 Doppler 监督的探针中，将自车速度从 `[5,0]` 改成 `[35,-10]`，原规划物理损失完全不变。

新配置启用 `model.planner_ego_context=true`，向规划器追加现有 ego encoder 的状态 token；检测和风险分支保持原输入。`planner.ego_initial_velocity_weight=1.0` 约束轨迹 spline 在 t=0 的解析导数与实测自车速度一致。缺失或非有限速度轴、不可用轨迹教师均掩码处理。

`4.20m/s × 3s ≈ 12.6m` 与现有 FDE 量级接近，支持速度失配可能很重要；仍需逐样本分析才能确定其贡献。

### 4. 模式监督被长描述稀释，模型选择缺少自由生成检查

旧模式 token 仅占整段约 100 token 中的一项；训练期验证只给模型正确前缀，`best.pt` 仅按总 loss 选择。

现在：

- 新配置的 `planner.mode_loss_weight=1.0` 为已知 SHORT/LONG 提供独立 CE，不随道路/目标描述长度稀释；未知教师模式不监督。
- 每个 SFT epoch 在固定、按教师模式分层的验证样本上自由生成，默认最多 32 帧，可用 `--validation-generation-samples N` 调整，0 表示关闭。
- 保存样本 ID、模式混淆矩阵、SHORT recall/precision、轨迹误差与覆盖率，以及匀速 CV 基线。CV 同时报告教师要求时间段及与实际生成重合时间点的误差和计数。
- `supervision_summary.json` 记录训练/验证教师模式分布。风险碰撞阳性数不等于 SHORT 教师数，不能替代该统计。
- 保留 `best.pt` 的最低验证 loss 语义；新增 `best_detection.pt`（优先 F1，再数量 MAE）与 `best_planning.pt`（优先格式有效率，再教师要求的时间覆盖，最后 ADE）。
- SHORT 输出不能通过省略 LONG 教师所要求的后半段轨迹获得更好的选择分数。缺失指标不会按零误差处理。

`best_planning.pt` 不是安全性或反应时间的最优保证：全 LONG 仍可能有较低 ADE，使用时必须同时看 SHORT 召回、初速、碰撞和时延。模式分层子集也不是总体类别比例估计。本次没有宣称已经解决真实数据的类别不平衡。

### 5. 转弯时传感器速度漏掉杆臂项

雷达位于自车中心前方，但旧适配器直接用自车中心速度作为雷达速度。现已按当前平面运动模型补上 `v_sensor = v_ego + ω × offset`，使转弯时的 Doppler 物理目标与传感器安装位置一致。

## 如何开展下一轮真实实验

1. 提供可访问的原始采集/预处理数据、`protocol.json`、`metrics.jsonl`、`best.pt` 路径，先核对训练曲线、类别/模式分布、各场景排除比例与划分。
2. 用当前适配器重新 prepare、validate，保留旧版本用于追溯；在新输出目录重训 Grounding，然后用 `best_detection.pt` 初始化 SFT。两阶段使用同一份新模型配置。
3. 先在 validation 看 precision/recall、数量 MAE、距离分桶误差，选择阈值；不能用 test 扫阈值或选模型。
4. SFT 对照 `best.pt` 与 `best_planning.pt`，同时检查匀速基线、初速误差、SHORT 召回、预测覆盖及离线碰撞。固定所有配置后才一次性评估 test。
5. 检测和 SFT 达到可用基线后，再比较 RL 对响应时间和碰撞的影响。RL 无法补救根本的漏检/误检或错误自车状态条件。

从包含 `pyproject.toml` 的仓库根目录操作，环境与路径按 [AGENTS.md](../AGENTS.md) 和 [RUNBOOK.md](../RUNBOOK.md) 配置。当前约定环境为 `/home/luxinlan/.conda/envs/lxl_radarvla`；在 `configs/local_paths.json` 配置实际数据、Qwen 权重及新实验目录。拿到数据后的训练入口示例如下，路径及 GPU 编号需替换为实际可用值，不应覆盖旧实验：

```bash
bash scripts/radarvla.sh doctor
bash scripts/radarvla.sh train-grounding --gpus 0 \
  --manifest /absolute/path/to/new/manifest.jsonl \
  --output /absolute/path/to/new_run/grounding
bash scripts/radarvla.sh train-sft --gpus 0 \
  --manifest /absolute/path/to/new/manifest.jsonl \
  --init-grounding /absolute/path/to/new_run/grounding/best_detection.pt \
  --output /absolute/path/to/new_run/sft
```

旧 checkpoint 缺少新增配置字段时维持旧推理行为。新状态 token 需要重新训练规划器；不要给旧权重静默启用。由于训练源代码、损失、数据和验证协议已改变，严格 resume 旧实验会被 provenance 检查拒绝，应建立新实验目录。

## 尚未声称解决的事项

- HF 推理当前仍逐 token 重算前缀，没有 KV cache；约 2.46 秒延迟需专门优化并检查和 RL replay 的一致性。本次未修改缓存路径。
- 64 档坐标量化有误差下限，但采用 signed-log 编码，并非均匀 2.54 m 网格。10 m/s 匀速轨迹的教师 token 往返 ADE 约 .60m，257 档约 .11m；不足解释 7.71m。更细词表应作为独立重训实验，不能直接用于旧权重。
- 采集器将所有 `vehicle.*` 统一写为 vehicle，包含自行车蓝图；分类映射、行人/骑行者补采要结合真实数据审计。
- Scenario2 固定在 train、最近 8 个 planner 目标之外的危险来源被排除、严格未来完整性过滤等可能造成评估偏差；要检查真实 `report.json` / `exclusions.jsonl`。检测本身仍是动态目标数量。
- detector 结果没有直接作为 planner 的逐目标条件输入，规划器仍从压缩 token 自己生成目标描述；未来可以做显式检测条件对照，避免一次同时改动全部结构。

## 验证范围

专项测试覆盖损失梯度、空帧/未知标注、扇区边界正例、原始检测计数、旧缓存重建、旋转杆臂、自车状态梯度、初速掩码、模式损失、自由生成覆盖和检查点选择。新增合成集两阶段集成测试启用所有新损失，并完成保存、恢复及评估。

按当前环境约定，完整模型测试命令为：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pytest -q tests
```

历史验证记录：本次修复在更新环境约定前，实际使用当时工作区的 `.conda/bin/python` 执行上述 `tests/` 套件，结果为 **322 passed in 55.65s**；这不是对新指定环境的复测结果。本次没有执行依赖 CARLA/采集环境的 `collection/tests/`。开启新目标函数和每轮生成验证的 CPU SFT 对照中，中断恢复与连续训练的权重、质量分数和最优 epoch 一致。

真实质量提升尚待可访问数据上的重训与独立评估；两份现有测试报告保留原样。
