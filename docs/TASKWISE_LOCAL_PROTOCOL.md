# Task 3：Local Mode 与按任务保留协议

2026-10-19 目录调整：此历史版本的训练数据迁移至
`<data_root>/prepared/taskwise_local_v1/`，原 `analysis/taskwise_local_v1/`
中的数据路径保留兼容链接，统计和诊断报告仍在 analysis。
运动车辆新版使用 `prepared/taskwise_moving_v1/` 和独立配套配置；
新版静止车辆、行人、自行车均作为背景惩罚，Risk 按全部合格运动车辆重算。
具体操作以根目录 `RUNBOOK.md` 的 Current Agent policy 章节为准。

## 使用版本

统一目录：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/taskwise_local_v1/`。

- 输入是 Task 2 的 `local_agent_risk_v1/manifest.jsonl`；原 8,835 帧保留。
- 配置：`configs/l40s_qwen25_3b_taskwise_local_v1.json`。
- 数据协议：`supervision_protocol=taskwise_local_v1`。
- Agent/Risk 协议仍为 `agent50_v1` / `local_agent_collision_v1`。
- 数量、实际回收结果和专项验收见输出目录 `REPORT.md`、`migration_report.json`、`acceptance.json`。

模型结构、物理词表、自由生成语法和原有损失权重保持现状。配置仅增加 data.supervision_protocol；没有启动训练。

## Mode 语义与 SFT 可用性

共同策略函数 `task_supervision.local_mode` 用于数据标注、校验、准备入口及 Planner 教师校验。

1. 有效 pcol_1s 或 pcol_2s >=0.5，或有效 dmin <=0.5m，或有效 areq >=3m/s²：SHORT。
2. 非空 Local 集合：以上四项条件均有效且都未触发：LONG。
3. 已确认 Local 集合为空，且 pcol_1s/pcol_2s 为有效负例：保留原 LONG 例外，不改变 dmin/areq 的无效 mask。
4. 其他：UNKNOWN。

UNKNOWN 不是自由生成的新 token。按用户选择，UNKNOWN 整帧排除出 SFT，manifest 与有效 Grounding 标签保留；其 Ego/Road GT 仍保存在数据中，本轮不用于 SFT。

危险事实和文本可用性分开：危险来源没进入 Planner selected 时，mode_label 仍是 SHORT，mode_mask 仍有效；planner_context_consistent=false，SFT 排除原因单列。不会改成安全或 UNKNOWN。

保持原 selected 规则、容量 8 和选择顺序。SHORT 文本继续按原优先顺序描述最多 2 个关键 Agent。四套集合在 agent_sets 中分别记录：local_agent_ids、risk_agent_ids、planner_selected_ids、sft_described_ids。

## 记录字段

| 字段 | 含义 |
|---|---|
| mode_label / mode_mask / mode_reasons | SHORT/LONG/UNKNOWN 风险事实、有效性、依据 |
| task_masks.detection | 当前 Agent 集合能否监督正例及背景 |
| task_masks.risk | 五维逐项有效性，保持 Task 2 标签与 mask |
| task_masks.agent_future | Agent ID→原 6 时刻有效性 |
| task_masks.ego_future | Ego 原 6 时刻有效性 |
| task_masks.road | 逐 segment 几何点和宽度有效性 |
| grounding_available | 此帧存在有效 Grounding 任务 |
| sft_available / sft_unavailable_reasons | SFT 阶段是否使用及排除原因 |
| previous_teacher_mode / previous_planner_supervision_available | 迁移前对照 |
| teacher_mode / planner_supervision_available | 新协议显式派生的兼容字段，不供旧协议静默复用 |

Dataset 验证语义、mask 和 ID 集合的一致性。collate 拒绝混合任务协议；训练/恢复/评估与可视化入口验证配置/权重协议。旧记录没有该字段时继续走 legacy_tasks_v1。

## 按任务计算损失

- Grounding：Detection 与 Risk 独立；未标注不等于空场景；未来监督只取有效目标/时刻，运动派生项只取共同有效的相邻时刻。
- SFT：UNKNOWN/上下文不一致在训练与验证 cohort 中排除并记录 ID。
- 已知 Mode 帧：道路宽度未知不清空道路几何；整段道路不可用时该段结构/数值标签均屏蔽，不把未知道路监督成空道路；保留有效 Ego 等任务。
- 轨迹端点不足时不生成外推控制点教师；控制点 CE 屏蔽，预测 spline 仍可通过有效真实时刻的 metric loss 学习。
- 数值 GT 超出固定物理词表时仅屏蔽不可表示的 scalar CE，保留原 GT 和合法 metric loss，记录 masked_out_of_range_fields；不裁剪 GT 或扩展词表。
- 平滑项只使用有相邻真实观测支持的区间，运动一致性要求对应状态/未来有效；全 mask 时返回可反传的有限零。

## 回收规则

检查原始 10,000 个候选，episode split 完全沿用 Task 2 映射。已有帧不重算 Risk，不修改物理 GT、传感器引用或数组。新增候选使用原采集 Agent 来源和 BBox/FOV 规则；行人/cyclist 继续忽略，不用 world snapshot 补造当前可见 Agent。

四帧雷达缺失、时间/位姿/必要当前输入不匹配是硬性排除。Risk/Mode 未知、Agent 超过 Planner 容量、道路宽度缺失不再单独触发整帧删除。原始重复标注仅在完全相同的同 ID 记录可证实时去重；冲突记录不能选择性猜测，相关标注有效性置未知并报告。

雷达缓存仅从保存的产品按现有 adapter 约定提取，完整 Power/Doppler、坐标及有效 mask 不裁剪；每份新缓存核对数值并记录 SHA256，没有重跑雷达处理。新帧的 Global 诊断按原 Global 物理规则生成；基线帧 Global/Local 标签逐值保持。

## 复现与验证

使用项目已有 lxl_radarvla 环境，在项目根目录运行。输出必须是独立的新目录：

```bash
python scripts/migrate_taskwise.py \
  --source /home/luxinlan/data/lxl_data/RadarVLA/analysis/local_agent_risk_v1/manifest.jsonl \
  --full /home/luxinlan/data/lxl_data/RadarVLA/full \
  --output-dir /home/luxinlan/data/lxl_data/RadarVLA/analysis/taskwise_local_reproduction \
  --config configs/l40s_qwen25_3b_taskwise_local_v1.json

python scripts/verify_taskwise.py \
  --directory /home/luxinlan/data/lxl_data/RadarVLA/analysis/taskwise_local_reproduction
```

迁移中不修改源码或配置；执行前后源码摘要一致才发布最终 manifest。失败时保留 partial 与 failure.json，不覆盖原数据。

全量 metadata 教师审计检查所有帧的 Mode/文本目标/任务 mask；依赖雷达测量的径向 scalar 不在 metadata 审计中伪造。真实代表帧通过 canonical Dataset 加载全部传感器，并以共享 Planner 语法/损失的 tiny backend 做端到端前向与反向，参数不更新。没有加载 Qwen 做训练。缺失的真实边界案例由合成专项测试覆盖并明确列出。

test 仅用于数据和监督完整性检查，不选阈值、权重或超参。后续新 Grounding 和 SFT 都必须配套此 manifest、配置与协议；旧 checkpoint 不静默迁移。
