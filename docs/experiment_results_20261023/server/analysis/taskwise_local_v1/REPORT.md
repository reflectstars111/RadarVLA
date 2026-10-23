# Task 3：Local Mode 与按任务保留协议验收报告

已完成独立数据发布、候选回收、Mode/任务 mask 迁移及端到端数据加载验收。未训练 Grounding/SFT，未修改模型结构、自由生成语法或损失权重，未覆盖旧数据/权重。

## 发布结果

- 原始候选 10000；基线 8835；核查新增候选 1165。
- 保留输入完整记录 **9841**；回收 **1006**；硬性排除 **159**。
- **64 个 episode**，split 完全沿用 Task 2 的 episode 映射。
- Grounding 可用 **9841**；SFT 可用 **9697**。输入完整与阶段可用分别计数。
- Manifest：`manifest.jsonl`；配置：`configs/l40s_qwen25_3b_taskwise_local_v1.json`；协议：`taskwise_local_v1`。
- Agent/Risk schema 保持 `agent50_v1` / `local_agent_collision_v1`。

## 修改前后与 Mode

| 来源 | 帧数 | Detection有效 | SHORT | LONG | UNKNOWN | Grounding可用 | SFT可用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原 Task2 基线（迁移后） | 8835 | 8835 | 590 | 8245 | 0 | 8835 | 8835 |
| 新回收帧 | 1006 | 1006 | 0 | 862 | 144 | 1006 | 862 |

基线 Mode 发生变化的帧数：0。逐帧变化在 comparison_report.json 和 samples.jsonl 中列出。

Mode 规则：有效 pcol_1s/2s 为正、dmin≤0.5m 或 areq≥3m/s² 任一项即可确认 SHORT；非空集合须四项条件均有效且未触发才为 LONG。按用户确认保留已知空集合 LONG 例外。其余为 UNKNOWN。

UNKNOWN 共 144 帧，其中 Detection 有效 144，Grounding 可用 144，SFT 可用 0。UNKNOWN 不改为 LONG，也不新增推理模式；整帧仅从 SFT 排除。

### 逐 split 分布

| split | 帧数 | SHORT | LONG | UNKNOWN | Grounding | SFT | Ego速度≥1m/s | 转弯≥0.05rad/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 6847 | 477 | 6251 | 119 | 6847 | 6728 | 2035 | 528 |
| val | 1472 | 56 | 1405 | 11 | 1472 | 1461 | 422 | 150 |
| test | 1522 | 57 | 1451 | 14 | 1522 | 1508 | 592 | 177 |

### 场景覆盖

| scenario | train帧/episode | val帧/episode | test帧/episode |
|---|---:|---:|---:|
| normal | 4114/23 | 766/4 | 1027/4 |
| scenario10 | 122/1 | 193/1 | 82/1 |
| scenario2 | 1024/9 | 71/1 | 81/1 |
| scenario3 | 535/10 | 112/3 | 332/3 |
| scenario4 | 584/1 | 0/0 | 0/0 |
| scenario7_9 | 468/1 | 330/1 | 0/0 |

各 split 的零 episode 即覆盖缺口；回收没有增加独立 episode，也未改变其 split。详见 scenario_coverage.json。

### 归因

- 回收 936 帧，旧原因：ValueError: adaptive Risk teacher is unknown or hazard is outside visible agents。
- 回收 69 帧，旧原因：ValueError: agent capacity or duplicate actor ID。
- 回收 1 帧，旧原因：ValueError: missing or invalid segmented road width。
- 硬性排除 159 帧：missing_four_causal_radar_products。
- baseline：LONG 且 Global 3s 危险 9 帧，LONG 且 Global 未知 0 帧；前者含空间域外来源 0，含忽略类别来源 9。
- recovered：LONG 且 Global 3s 危险 352 帧，LONG 且 Global 未知 509 帧；前者含空间域外来源 350，含忽略类别来源 2。

LONG 的增加按新增样本、Local 风险证据及已确认空集合单列；不能解释成总体驾驶风险降低。旧转换因 Global 风险/教师归因排除的帧，可能在 Local 定义下可确认为 LONG，也可能证据不足而为 UNKNOWN。这里没有把域外危险直接当作局部安全。

## 任务级监督覆盖

| 来源 | pcol1有效 | pcol2有效 | pcol3有效 | dmin有效 | areq有效 | Ego Future任一点有效 | Agent Future有效点 | Road几何有效帧 | Road宽度有效帧 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 基线 | 8835 | 8835 | 8835 | 8678 | 8474 | 8835 | 226547 | 8835 | 8835 |
| 回收 | 906 | 873 | 862 | 856 | 856 | 1006 | 28690 | 1005 | 1005 |

每个 split/scenario/episode 的有效数量、正负分布及静止/运动 Agent 数见 migration_report.json；未知帧各 Risk 分量的有效计数见 comparison_report.json。速度 1m/s、绝对转率 0.05rad/s 仅用于统计，没有用于筛选。

### Agent 集合与上下文

- 合法 Local Agent 观测数 42687；Risk 使用 42687；Planner selected 39832；SFT 描述 36344。这些是跨帧观测数量。
- 有危险来源未选入 Planner 的帧 0；其中 SFT 因上下文不一致排除 0 帧。
- 某个来源未选入与完全无法归因不同：只要仍有符合原规则的 selected 关键目标，继续使用原文本选择；完全无法归因时 Mode 事实仍是 SHORT，SFT 可用性另行置否。
- 四套集合及顺序逐帧记录。Risk 使用全部合法 Agent；Planner 最多 8，SHORT 继续最多描述 2 个关键目标。

### 损失与缺失字段

- Detection/Risk/Agent Future 独立使用有效 mask；无效场景不作为空场景背景。运动派生项仅用共同有效的相邻时间点。
- 已知 Mode 的 SFT 保留有效任务。道路节未知时屏蔽该节数值与结构标签，不监督为没有道路；未知宽度不清空有效几何。
- 轨迹不能合法拟合完整控制点时，控制点 CE 不用外推 GT；metric loss 只用真实有效时间点。
- 超词表 scalar 仅屏蔽其 CE，不裁剪真实未来；保留合法 metric loss。全量 metadata 教师审计的非径向超界字段数：0。测量相关径向字段另由真实样本验证。
- 平滑与一致性按任务/观测区间门控；全 mask 时有限零 loss，可反传且无无效梯度。

## 数据不变量、兼容性及验证

- 8,835 个基线样本的检测标注、Local/Global Risk、传感器引用、道路结构及 Agent/Ego Future 逐字段保持；split 不变。
- 新增缓存只提取保存产品，沿用现有坐标约定；逐份核对数组数值及 SHA256，Power/Doppler/mask 不裁剪，不重算雷达。
- 原 Actor 来源、静止/运动处理、BBox/FOV 规则及行人/cyclist 忽略策略保持。采集未记录的目标不补造。
- 配置差异：data.supervision_protocol。旧协议继续显式分支；混合数据/权重协议拒绝加载，不修改旧 checkpoint protocol。
- 专项回归：`165 passed in 34.65s`。
- 全量验收：9841 帧、64 episode；30795 个传感器输入路径存在，4648 份新增缓存哈希通过。
- 每帧教师语法、Mode 和描述 ID/顺序均核对；metadata 审计不伪造径向测量。
- 真实代表帧：baseline, long, many_agents, short, partial_agent_future, recovered, missing_road, unknown。
- 未在当前真实数据出现、由合成专项测试覆盖的边界：context_inconsistent, partial_ego_future。
- 代表帧通过 canonical Dataset 和共享语法/loss 的 tiny Planner 做端到端前向、反向和混合 batch 检查；参数最大变化 0.0，optimizer steps=0。未加载 Qwen 训练。
- test 仅核验数据和监督定义，未用于调参。

## 文件与版本追溯

- manifest.jsonl：发布数据；samples.jsonl：逐样本任务/Mode/集合/运动统计；hard_exclusions.jsonl：硬排除清单。
- migration_report.json、comparison_report.json、teacher_validation.jsonl、acceptance.json：对照与验收。
- raw_evidence.json、radar_cache_hashes.json、provenance.json：输入、原始证据、缓存和执行源码摘要。
- mode_attribution.jsonl / mode_attribution_report.json：Global/Local、mask、场景与原始危险来源归因。
- source_snapshot/、source_snapshot_hashes.json：执行实现与测试源码快照，可在既有项目中还原；不含数据或模型权重。
- implementation_provenance.json、test_results.log、config_diff.json：最终交付实现、测试和配置差异。
- 协议说明：docs/TASKWISE_LOCAL_PROTOCOL.md。

后续 Grounding/SFT 必须使用本目录 manifest、configs/l40s_qwen25_3b_taskwise_local_v1.json 和 taskwise_local_v1 协议；先重训 Grounding，再由匹配的新版 Grounding 初始化 SFT。本轮没有启动这两项训练。

## 修改文件

- src/collection_adapter.py
- src/data.py
- src/planner.py
- src/pipeline.py
- src/visual_diagnosis.py
- src/local_risk.py
- src/task_supervision.py
- src/taskwise_migration.py
- configs/l40s_qwen25_3b_taskwise_local_v1.json
- scripts/migrate_taskwise.py
- scripts/verify_taskwise.py
- scripts/report_taskwise.py
- scripts/audit_taskwise_modes.py
- tests/test_taskwise_supervision.py
- docs/TASKWISE_LOCAL_PROTOCOL.md

Grounding losses.py 的既有逐维/逐时刻 mask 机制经专项测试验证后保留；Planner 的任务门控与共享 Mode 判定由本次更新接入。
