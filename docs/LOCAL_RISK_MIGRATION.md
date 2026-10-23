# Task 2：Local Agent Collision Risk

## 数据和配置

统一数据目录为 `/home/luxinlan/data/lxl_data/RadarVLA`。

- 输入：`analysis/grounding_agent50_v1/manifest.jsonl`。
- 输出：`analysis/local_agent_risk_v1/manifest.jsonl`。
- 配套配置：`configs/l40s_qwen25_3b_localrisk_v1.json`。
- 最终数据目录的 `REPORT.md`、`acceptance.json`、`migration_report.json` 提供实际统计与验收。

Local Risk 仅使用 Task 1 最终保留的 Agent，包含全部当前域内目标而非 Planner 最近 8 个。被设为忽略的行人/自行车不贡献 Local Risk。不会新增原采集器排除的目标，不按中心距离删除边界相交目标，不截断当前目标未来驶出域外的轨迹。

## 标签与 schema

新记录显式设置 `risk_supervision_schema=local_agent_collision_v1`。

| 字段 | 含义 |
|---|---|
| `global_risk` | 逐值复制原 `risk_label`；`global_risk_schema=legacy_global_v1` |
| `local_risk` | 新 Local 五维标签、mask、来源、覆盖及无效原因 |
| `local_agent_risk_labels` | 全部保留 Agent 的个体 Local 风险 |
| `risk_provenance` | Local/Global/忽略/Planner ID、时间步长及原始帧证据 |
| `legacy_teacher_mode` | 迁移前教师模式 |
| `legacy_planner_supervision_available` | 迁移前 SFT 可用性 |
| `teacher_mode` | 根据 Local 风险重判的 SHORT/LONG，无法监督时为 null |
| `planner_supervision_unavailable_reason` | Local 证据不足、危险不在 Planner selected 等原因 |

原 `risk_label`、`agent_risk_labels`、`risk_agents` 保留历史 Global 语义。新版 Dataset 显式将 Local 标签接入已有 `risk_target/risk_mask` 与逐 Agent 风险张量；不改模型结构/输出维度。无 schema 的旧 manifest 继续使用旧协议。新版字段不完整时不回退到 Global。混合 schema collate、配置与数据不匹配、跨协议 SFT 初始化/评估明确失败。

原 Planner 目标 ID 与顺序不变。危险来自未选入的合法 Local Agent 时，该帧仍保留 Grounding 监督，SFT 不可用。旧不可用帧按新证据重新判断；不机械继承旧标记。预测风险作为 SFT 输入的现有架构不改，模式与轨迹教师按新协议生成。

## 物理规则和有效性

保留当前速度/转率反事实 Ego、其他 Agent 的 6 个实际未来 GT 点、0.05 s 插值检查、框间距 <0.5 m、累计 1/2/3 s pcol、80 m dmin 截断、0–8 m/s² 步长 0.5 的恒定制动枚举。

Local 覆盖仅取决于当前 Local 集合是否已知及这些目标的未来有效性，域外 actor 的变化不影响它。原始 30 帧有效性仅用于检查插值区间是否缺证据，不用 30 帧位置替换用户选择的 6 点轨迹。

- 观测危险确认对应 pcol 正例；负例要求该窗口完整覆盖。
- dmin 完整覆盖才有效；已观测零间距已确定下界时保留有效。
- 全部 17 个候选均记录 safe/dangerous/unknown；无单调假设。
- areq 仅在完整证据下找到最小安全候选时有效；当前已危险接近、无解及缺失证据均不监督零刹车。
- 空集合保持旧约定：已确认集合为空时 pcol 负例有效，dmin 存 80 但 mask 无效，areq 存 0 但 mask 无效。
- 不因任何单个分量无效删除帧或清空其余 mask。

矩形距离采用 NumPy 批量计算，与原标量 `box_distance` 做数值等价测试；CV/CTRV rollout 复用原函数。旧 Global 标签完全不重算；旧排除帧的诊断重建使用经等价测试的批量旧协议实现。

## 复现（不训练）

在项目根目录使用既有 `lxl_radarvla` Python：

```bash
python scripts/migrate_local_risk.py \
  --source /home/luxinlan/data/lxl_data/RadarVLA/analysis/grounding_agent50_v1/manifest.jsonl \
  --output-dir /home/luxinlan/data/lxl_data/RadarVLA/analysis/local_agent_risk_reproduction \
  --config configs/l40s_qwen25_3b_localrisk_v1.json
```

输出必须是新目录。中断后相同参数加 `--resume`；输入/源码/配置摘要不匹配时拒绝续跑。记录每 50 帧提交输出偏移，未提交尾部在续跑时截断。原始数据缺失或不变量检查失败时停止发布。

```bash
python scripts/audit_local_risk.py \
  --directory /home/luxinlan/data/lxl_data/RadarVLA/analysis/local_agent_risk_reproduction \
  --full /home/luxinlan/data/lxl_data/RadarVLA/full \
  --exclusions /home/luxinlan/data/lxl_data/RadarVLA/prepared/full_v3_0a225d38dde3/exclusions.jsonl

python scripts/verify_local_risk.py \
  --directory /home/luxinlan/data/lxl_data/RadarVLA/analysis/local_agent_risk_reproduction
```

排除帧只做诊断，不补入原 8,835 帧。旧 exclusions 的混合原因不能直接解释为域外危险；统计标志可重叠，按保存的原始帧重新归因，不声称复原当时排除器的唯一原因。

验收逐条比较不变量与 mask，并对真实代表样本检查传感器张量、loss、反向与教师。没有 optimizer step。test 只用于完整性验收，不挑参数。后续训练须显式配套新 manifest 与新配置；本任务不启动训练。
