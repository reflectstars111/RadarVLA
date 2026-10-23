# Task 2：Local Agent Collision Risk 验收报告

本轮完成标签迁移、原始证据核对、Global/Local 对照及教师迁移。没有启动真实数据训练，没有优化器更新，旧数据与权重保留。

## 发布数据

- 样本 8835，episode 64，原 split、检测 GT、Radar 引用及 Agent/Ego Future 不变。
- Local 只使用 Task 1 保留的 Agent；行人/cyclist 仅保留原 Global 诊断语义。
- 输入 Task 1 manifest SHA256：`be5f7463c09926abf6f99284629e6c6b4c05f299fa499c89558cc06e4ac476db`。
- 输出 manifest SHA256：`bfa79ceb3ae6e6380c72b33abe9f037b804a6941abd4092d7fa60c9cb0e456bd`。
- 新 schema：`local_agent_collision_v1`，配置 `configs/l40s_qwen25_3b_localrisk_v1.json`。

## Global / Local 对照

下面危险/安全/未知仅指该窗口累计 pcol；Global 危险 / Local 安全要求 Local 独立计算得到有效负例，不由“危险来自域外”推断。连续风险与 SFT 复合危险另计。

| split | 帧数 | 窗口 | Global危险/Local危险 | Global危险/Local安全 | Global危险/Local未知 | Local正例 | Local负例 | Local未知 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| train | 6166 | 1s | 215 | 0 | 0 | 215 | 5951 | 0 |
| train | 6166 | 2s | 361 | 4 | 0 | 361 | 5805 | 0 |
| train | 6166 | 3s | 477 | 9 | 0 | 477 | 5689 | 0 |
| val | 1337 | 1s | 17 | 0 | 0 | 17 | 1320 | 0 |
| val | 1337 | 2s | 40 | 0 | 0 | 40 | 1297 | 0 |
| val | 1337 | 3s | 56 | 0 | 0 | 56 | 1281 | 0 |
| test | 1332 | 1s | 28 | 0 | 0 | 28 | 1304 | 0 |
| test | 1332 | 2s | 40 | 0 | 0 | 40 | 1292 | 0 |
| test | 1332 | 3s | 57 | 0 | 0 | 57 | 1275 | 0 |

| split | dmin有效 | areq有效 | areq有效零 | areq有效正值 | SFT可用 | 新增可用 | 新增不可用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 6083 (98.7%) | 5902 (95.7%) | 5616 | 286 | 6166 | 9 | 0 |
| val | 1292 (96.6%) | 1283 (96.0%) | 1237 | 46 | 1337 | 0 | 0 |
| test | 1303 (97.8%) | 1289 (96.8%) | 1246 | 43 | 1332 | 0 | 0 |

Local pcol 有效覆盖率：
- train：1/2/3 s = 100.0% / 100.0% / 100.0%。
- val：1/2/3 s = 100.0% / 100.0% / 100.0%。
- test：1/2/3 s = 100.0% / 100.0% / 100.0%。

每个 split/episode/场景的完整计数见 migration_report.json。当前 Local 空集合按旧约定，dmin 和 areq 不适用；无效 areq 存 0 不代表有效零刹车。

### 当前样本的危险来源

- 危险来源观测 705 个；中心超过 50 m 的观测 0 个。中心距离只用于诊断，不用于 GT 纳入。
- 原 BBox/FOV 判定为 50 m 外 0，FOV 外 0。
- Local 保留来源 667，忽略类别来源 38，空间域外来源 0。
- 未进入 Planner selected 的来源观测 38；逐帧、逐 actor 见 risk_sources.jsonl。

### 旧排除帧复核

- 混合风险排除记录 936 条，成功重建 936，无法恢复 0。
- 含空间域外危险来源 322 条，其中通过原 FOV 但在 50 m 外的危险来源 0 条。
- 含忽略类别危险来源 2，含 Planner 未选入的危险来源 324，Local 未来不完整 133。
- 危险来源中心距离超过 50 m 的排除帧 0 条；中心越界与 BBox/FOV 不相交分别记录。
- 这些标志可重叠。旧原因将风险未知与不可见危险混合记录，重建结果不能证明当时排除器的唯一原因。未将任何排除帧补入训练集。
- 诊断重建的 Global 使用原六点协议、全局覆盖和原风险目标界限，已与原标量实现对照测试；现有 8,835 帧的 Global 直接逐值复制。

现有 cohort 的 Global/Local 差异不能代表原始全部采集帧：旧转换曾排除无法归因的危险或未知样本。必须同时查看上述排除帧审计。

## 有效性与教师

- Local 覆盖依赖固定当前 Local 集合和各目标未来，域外 actor 出现/消失不使 Local 变未知。
- 保留六个实际未来轨迹点；原始 30 帧有效性用于防止跨缺失证据插值，不改成 30 点位置轨迹。
- 每个窗口独立确认 pcol：观测危险可确认正例，负例要求完整覆盖。dmin 缺未来时无效，已观测零间距下界除外。
- 全部 17 个恒定刹车候选均已检查并存状态，不假设制动风险单调；当前危险接近、无解、未来不足时 areq 无效。
- SFT 用 Local 风险重判模式，保留原 Planner 目标与顺序；危险无法归因于所选目标或证据不足时只标记 SFT 不可用，Grounding 帧仍保留。
- 旧 risk_label/agent_risk_labels/risk_agents 保持原 Global 含义；新版 Dataset、训练/恢复/评估及可视化检查显式风险协议。

## 验收与追溯

- 全量验收 PASS：8835 帧、64 episode、27361 个传感器文件路径存在。
- 真实样本覆盖：agents, long, empty, areq_invalid, short, teacher_changed。
- 真实样本前向/loss/反向检查通过，参数最大变化 0.0；Radar、mask、GT 和未来张量一致。
- 专项回归：`150 passed in 29.20s`。
- test 仅验收标签和完整性，没有用于参数选择。
- samples.jsonl：逐样本五维对照、mask、来源与教师变化；local_agent_risk_labels：逐 Agent 标签。
- local_risk.braking_candidates：全部候选状态与危险来源；coverage：有效观测区间与时间网格。
- raw_evidence.json、provenance.json：原始帧/metadata/选中 anchor 摘要及执行阶段源码；implementation_provenance.json：最终实现和验收摘要。
- excluded_samples.jsonl：旧排除帧逐条诊断；risk_sources.jsonl：原始空间域归因。

下一步训练必须配套新 manifest 与新配置：先重新训练 Grounding，再从相同 Local Risk 协议的 Grounding 初始化 SFT。本任务没有启动训练。
