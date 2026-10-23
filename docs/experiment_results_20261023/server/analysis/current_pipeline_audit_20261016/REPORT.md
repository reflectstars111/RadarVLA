# RadarVLA 当前 pipeline 诊断（2026-10-16）

## 优先修改的三项

1. **先修数据字段与类别映射，再重训 Grounding。** 当前 50 m manifest 仍保留旧传感器速度；转弯帧需要从原始 ego 运动和安装位姿重新生成速度字段。原始自行车蓝图被统一标成 vehicle（见下方实测数量）。保留 split 和 Power/Doppler 数组。
2. **Grounding 使用已修复的检测损失与显式监督域，风险权重另做单项对照。** 旧权重未训练这些修复；不要继续增加旧实验 epoch。风险 AP 下降已证实，其成因尚不能单独归为梯度冲突。
3. **随后重训 SFT，先比较冻结与联合训练，再分别增加 ego 状态和初速约束。** 编解码误差较小，自由生成差距较大；任务级选权重替代单看总 loss，SHORT 必须同时看召回、误判和覆盖。

本轮只执行源码/原始数据审计和旧权重推理，没有训练、覆盖权重或按 test 选择参数。

**这轮最重要的新判断：** 抽查未发现普遍的时间、坐标或轨迹转换错误；自行车类别与旧传感器速度是已确认的数据字段问题。被排除危险帧的归因结果没有支持“最近8个目标截断是主要原因”的怀疑。检测损失、模式/风险条件和模型选择仍需通过分开的新训练验证。

![训练退化、检测阈值与轨迹分解](diagnosis_overview.png)

## 证据版本与检查范围

- Git HEAD：`7706bd2e9c33903d83b077c782dda39d17b926e9`；工作区源码摘要见 `provenance.json`，不能把 HEAD 当成全部实际源码。
- Manifest SHA256：`22dfa79b95f69a30590a5cd83ede82ca0bcac359873c58c92c2744eb56ae97c1`；8835 帧/64 episodes。
- 所有 64 episode metadata 确认 Agent/道路 50 m，雷达约 118.04 m。引用文件 27425 个，缺失 0 个，隔离/ID 错误 0 个。
- 原始数值检查 222 帧，检测推理 96 帧（10 个 val episode，5 个分散帧＋全部碰撞正例），轨迹复用固定 96 帧。危险富集子集不代表总体分布。
- v2 protocol 中 loss/model/data/planner 等源码摘要与当前源码不同；新增修复效果需要新训练验证。

- 原始审计执行快照和归因简化的执行记录见 `execution_notes.json` / `audit_executed_snapshot.py`；脚本运行期间的后续补充没有冒充为原始执行版本。

## 当前监督链路怎样影响诊断

采集器保存的是路线连接的 ego 道路走廊（16个采样点），不是视域内所有道路；adapter再拆为road segments。Agent标注为50m视域内目标，风险则可引用完整world snapshot中的risk_agents，因此两者覆盖并不相同。
Grounding监督检测存在性、类别、框/位置/速度/目标未来与五维风险；它没有独立的道路输出头。SFT从压缩雷达token、风险和指令生成模式、道路、目标描述及ego轨迹；当前新增的ego token不在旧权重配置中。
SFT阶段仍把Grounding loss与token/物理loss相加，优化器包含两部分可训练参数，风险条件也没有自动detach。检测框列表并未直接成为规划器逐目标条件。因此“检测不错”不能自动推出“SFT生成目标/道路也正确”，冻结对照需要明确冻结整个上游并保持eval模式。
这些定位分别对应 `src/collection_adapter.py`、`src/model.py`、`src/planner.py` 与 `src/pipeline.py::_loss`。

## 数据与转换审计

| Split | 帧 / episode | 低于1m/s | SHORT / 危险episode | 类别实例 | 速度差>0.1m/s |
|---|---:|---:|---:|---|---:|
| train | 6166 / 45 | 4475 | 486 / 16 | {'vehicle': 32600, 'pedestrian': 5} | 279 |
| val | 1337 / 10 | 1008 | 56 / 4 | {'vehicle': 5012} | 79 |

222帧的帧ID、可见目标集合、道路、未来掩码、风险重算与域字段检查，失败项：`{}`。Agent位置、速度、框及未来轨迹与原始转换的最大差异均为 0。

| 数值检查 | 最大绝对差 |
|---|---:|
| time_delta_s | 0 |
| ego_future_error_m | 0 |
| pose_error | 0 |
| corrected_sensor_velocity_error_mps | 1.46804 |
| dataset_future_error | 4.75374e-07 |
| raw_power_error | 0 |
| raw_doppler_error | 0 |

另外核对 55 帧原始range/azimuth坐标轴，最大差分别为 0m / 0rad；细节见 `radar_axis_checks.json`。

抽查发现自行车蓝图仍标 vehicle：`[{'type_id': 'vehicle.diamondback.century', 'category': 'vehicle', 'observations': 35}, {'type_id': 'vehicle.bh.crossbike', 'category': 'vehicle', 'observations': 31}, {'type_id': 'vehicle.gazelle.omafiets', 'category': 'vehicle', 'observations': 27}]`。这是当前采集分类规则的确定问题；数量为抽查的目标观测次数，不是独立自行车数。应修 collector 和原始 type_id→类别转换，并在原数据上重建分类标签。

传感器速度检查区分当前帧全量统计与四帧历史抽查。它影响几何编码及基于状态的径向速度监督；不能据此认定雷达文件本身有错。超过0.1m/s的只有358/7503个train/val帧，不能用它单独解释全部误报。类别映射修正影响类别监督，但总检测P/R使用类别无关中心匹配，修类别也不会直接解决位置误报。
抽查实测 Doppler 与状态导出径向速度绝对残差均值 `{'samples': 735, 'mean': 0.5458161594801418}`；该差值还包含关联/噪声，不能全归因于杆臂。

## 被筛掉的危险样本

| 原始预处理排除原因（全池既有记录） | 帧数 |
|---|---:|
| ValueError: missing four causal radar products | 159 |
| ValueError: Risk hazard cannot be attributed to a planner-selected visible agent | 350 |
| ValueError: missing or invalid segmented road width | 1 |
| ValueError: Risk safe teacher lacks complete collision coverage | 507 |
| ValueError: Risk safe teacher lacks valid distance or braking evidence | 148 |

| train/val 危险归因检查 | 帧数 |
|---|---:|
| audited | 284 |
| with_outside_visible_hazard | 284 |
| with_visible_but_not_selected_hazard | 0 |
| with_selected_hazard | 0 |
| with_hazard_outside_radar_bev | 258 |
| with_hazard_outside_labels_but_in_radar_bev | 26 |
| no_individual_hazard_reproduced | 0 |
| with_incomplete_future | 26 |

实测 284 帧危险均有标签集合外来源，258 帧涉及118m雷达BEV外来源，26 帧涉及雷达BEV内但标签集合外来源；可见但未进入最近8个的危险帧为 0。这批证据不支持优先扩大max_agents。部分筛除有可观测性理由，不能把排除危险帧本身一概称为pipeline错误。

若要回收这些帧，应先拆开全局风险与当前传感器可监督风险，保留可观测性/未知掩码，再分别决定Grounding和SFT是否可用；不可把域外危险改成安全负例，也不可把单向雷达不可观测事件无条件强加为预测目标。

逐 actor 重新计算反事实危险，保留其 visible/planner_selected 状态，并区分标注域外但仍在118m雷达BEV内、以及雷达BEV外的危险。BEV相交不保证真实可见，仍有垂向视域和遮挡限制。多个原因可同时发生；此分析不将缺失未来标签改为安全。对可见但不在最近8个中的危险，应改选取策略或解除其对 Grounding 样本保留的限制。当前 nearest-8 策略会让准备阶段和规划器容量共同决定数据分布。

## 检测：修正监督域后的旧权重诊断

缓存新增域与位姿字段，绑定 manifest/权重/评分源码摘要；采用旧配置 NMS 与 2m 匹配。域外仅忽略未匹配预测，与 GT 匹配的边界目标仍计 TP。局部域重新计分不能称为模型能力提升。

| 权重 | 阈值 | P | R | F1 | 原始/帧 | 域内计分/帧 | 忽略/帧 | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retrained_grounding_v2_best_sweep | 0.5 | 0.021 | 0.962 | 0.041 | 211.09 | 189.47 | 21.62 | 15 |
| retrained_grounding_v2_best_sweep | 0.9 | 0.442 | 0.351 | 0.391 | 3.34 | 3.25 | 0.09 | 255 |
| retrained_sft_v2_epoch_002_sweep | 0.5 | 0.028 | 0.934 | 0.054 | 154.71 | 137.74 | 16.97 | 26 |
| retrained_sft_v2_epoch_002_sweep | 0.9 | 0.517 | 0.458 | 0.486 | 3.64 | 3.62 | 0.01 | 213 |
| retrained_sft_v2_epoch_003_sweep | 0.5 | 0.024 | 0.934 | 0.047 | 184.65 | 158.61 | 26.03 | 26 |
| retrained_sft_v2_epoch_003_sweep | 0.9 | 0.427 | 0.318 | 0.364 | 3.09 | 3.05 | 0.04 | 268 |
| retrained_sft_v2_epoch_005_sweep | 0.5 | 0.032 | 0.916 | 0.062 | 132.12 | 116.77 | 15.35 | 33 |
| retrained_sft_v2_epoch_005_sweep | 0.9 | 0.489 | 0.440 | 0.463 | 3.71 | 3.69 | 0.02 | 220 |

完整五阈值、逐样本、逐 episode、场景、速度和转弯分组在 `detection/*_sweep.json`。这里的0.9是预先固定展示点，尚未据子集确定部署阈值。

Grounding在0.9阈值仅有9/321个原始预测被域外规则忽略；重新定义评分区域没有消除域内误报和漏检。需要修训练目标，不能将更换计分口径当成模型改善。

## 轨迹与风险

共同时间点 ADE 使用 96 帧、312 个时间点；3秒 FDE 使用 30 帧。FDE不能外推至其余SHORT帧。

| 方法 | 配对 ADE (m) | 配对3秒 FDE (m) |
|---|---:|---:|
| constant_velocity | 1.347 | 4.076 |
| codec | 0.180 | 0.321 |
| teacher_forced | 0.548 | 0.597 |
| free_generation | 1.914 | 4.806 |

教师SHORT同样会缩短编解码/正确前序输出，不能只检查自由生成的长度。各方法未配对均值和分母见 `trajectory_paired_summary.json`，计划坐标见 `trajectory_decomposition.json`。

本轮轨迹推理沿用权重 protocol 的 bfloat16 autocast；历史分解脚本未启用该 autocast，数值不应要求逐位相同。编解码、正确前序和自由生成的误差量级结论应以同一轮的配对结果为准。

风险替换重新推理使用73帧旧诊断子集（56 SHORT、17 LONG）：

| 风险条件 | SHORT选对 | SHORT漏判 | LONG误判成SHORT | SHORT召回 | SHORT精度 |
|---|---:|---:|---:|---:|---:|
| predicted | 37 | 19 | 5 | 0.661 | 0.881 |
| oracle | 55 | 1 | 2 | 0.982 | 0.965 |

真实风险替换只证明该条件对当前模式预测有影响，不能证明风险头是全部轨迹误差的原因，也不是可部署方案。逐 episode、运动和转弯结果见 `risk_oracle_review.json`。

历史文件没有保存权重与执行代码摘要，所以本轮按同一73帧重新运行并补齐摘要。其中 10 帧五维风险标签不完整，已单独分组；未知维度保留模型预测值，不能当成oracle真值。

风险标签来自“自车维持当前速度/转率”的反事实几何 rollout；轨迹教师是实际驾驶未来轨迹。两者不同不必然是标签错误，例如教师已经刹车但反事实仍危险。当前 pcol 为几何事件标签，模型输出概率不等于已校准的真实事故概率。

## 已修正的诊断工具

- RA 标记从 current_ego 转到 current_radar；扇区与雷达原点同位姿，BEV 坐标保持 ego。非零平移/旋转用专项测试验证。
- 完整显示保存的 unfolded Doppler，无可视化遮罩；模型 forward 仍按 radar_doppler_valid 将无效 Doppler 置零。视觉看到的无效区域并不是模型实际 Doppler 数值输入。
- 单独缓存域内/域外计数、样本级 TP/FP/FN；旧不含域的扫描文件保留历史语义。

已保存真实转弯帧的完整叠加图：[turning_overlay.png](turning_overlay.png)；对应sample、安装位姿和权重摘要见 `turning_overlay.json`。该图验证坐标路径并展示漏检，不能代替定量匹配。

## 下一轮对照与训练选择

| 问题 | 修改位置 | 数据/训练影响 | 验收 |
|---|---|---|---|
| 旧传感器速度 | `src/collection_adapter.py::_radar_sensor_velocity`；重新生成 manifest 状态字段 | 当前函数已修，旧数据仍需更新；随后重训 Grounding 和 SFT | 与原始刚体速度一致，转弯组径向残差 |
| 自行车类别统一为 vehicle | `collection/collector/recorder.py::_actor` 的 actor_type 映射；`src/collection_adapter.py::_agent` 的 type_id 映射 | 从原始 type_id 重建类别；两阶段重训 | 自行车映射与未知类掩码专项检查、类别覆盖统计 |
| 检测负例归一化/域 | `src/losses.py::loss_grounding`、`src/annotation_domain.py` | 当前代码已修；Grounding 必须重训，SFT 从新版初始化 | 域内P/R/F1及原始输出数 |
| 域外危险样本排除/可观测性 | `src/collection_adapter.py::_mode_from_risk` / `_attach_supervision` | 本轮未发现最近8截断导致的排除；回收数据前先明确风险监督域与阶段可用性，改变样本池另立实验 | 域内/域外危险来源及未知掩码 |
| 风险退化 | `src/losses.py` collision focal alpha；`src/pipeline.py` 记录分量与任务选择 | G1单因素对照；不同时改采样 | AP/Brier及检测回归 |
| SFT上游退化 | `src/pipeline.py::_loss` / optimizer / train-eval模式 | 新增冻结配置后做S0/S1，保持同一初始化 | 风险/检测保持、模式和覆盖 |
| ego条件/初速监督 | `src/model.py` planner_ego_context；`src/planner.py::planner_physics_loss` | 当前实现已有；分开开关并重训SFT | 运动/转弯轨迹、初速及CV对照 |
| 统一best选择失真 | `src/training_quality.py`、`src/pipeline.py` | 当前已有检测/规划选择；风险任务选择需补 | 不以缩短输出换低ADE；val选权重与阈值 |

具体参数见 `next_experiments.json`：种子42、5 epochs、学习率3e-4、有效batch12、AdamW、相同数据与更新预算。
顺序为 G0 修正基线 → G1 仅风险 focal alpha 调整 → 同一个 Grounding 的 S0冻结/S1联合 → S2仅ego状态 → S3仅初速约束；模式CE另作条件实验。当前默认配置同时开启若干修复，直接整套训练可建新基线，却不能归因每项收益。
冻结开关和风险alpha目前需要新增配置支持；S2仅追加ego token、不改变Grounding权重张量，但现有初始化检查要求ModelConfig完全一致，需要增加显式的context-only迁移选项并记录配置差异与源权重摘要。不能静默篡改旧checkpoint protocol。JSON是实验规格，不是假装现有CLI已经支持这些参数。新实验不得 strict resume 旧权重；固定的数据预处理修正也必须进入 provenance。
检测按域内F1/数量误差选，风险按AP与Brier保留Pareto候选；SFT同时检查教师要求覆盖、模式混淆、运动样本相对CV误差、初速以及上游风险/检测退化。阈值先在完整validation确认，最后才评估test一次。

## 覆盖缺口与未决问题

- scenario4 的唯一实质 episode 留在 train，val/test 缺少该场景，不能据现有验证宣称覆盖。验证无行人/骑行者类别证据；先修类别再统计缺口。
- 补采优先增加独立的运动、转弯、危险交互与行人/骑行者 episode；避免仅加静止或重复帧。
- 风险退化可能涉及不均衡、过拟合、损失竞争和不可观测标签；本轮不宣称已识别梯度冲突，不宣称初速约束已改善自由生成。
- 数据标签数值对齐和小样本推理不能替代完整验证及新训练。所有性能结论限定于报告给出的样本和分母。

## 复运行

仓库根目录运行 `bash scripts/run_current_diagnosis.sh analysis/current_pipeline_audit_new`。环境按 AGENTS.md；默认GPU0，可设 RADAR_VLA_AUDIT_DEVICE=cpu；可用 screen 启动。原始数据路径从现有manifest推导。输出目录必须独立。
