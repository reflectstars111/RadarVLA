# Task 1 验收报告：Agent50 与行人/自行车忽略

## 结果

发布通过；未训练模型，未覆盖原 manifest 或 checkpoint。最终 `manifest.jsonl` 保留原 8,835 帧、64 个 episode 及全部 split 分配。按用户追加选择，将行人/自行车移出检测正例并设为忽略区域。

| split | 帧数 | 原 Agent 观测 | 保留 agent 正例 | 忽略自行车 | 忽略行人 | SFT 不可用帧 |
|---|---:|---:|---:|---:|---:|---:|
| train | 6,166 | 32,605 | 29,021 | 3,579 | 5 | 9 |
| val | 1,337 | 5,012 | 4,469 | 543 | 0 | 0 |
| test | 1,332 | 4,919 | 4,300 | 619 | 0 | 0 |
| 合计 | 8,835 | 42,536 | 37,790 | 4,741 | 5 | 9 |

数量为跨帧观测数，不是独立 actor 数。每帧保留 ID 与忽略 ID 的并集等于原 ID 集合，二者无交集。基础空间域核对没有导致额外实例删除。未从 world snapshot 补入原先排除的目标。

## 监督变更

1. 当前 Radar 安装位姿、真实水平 FOV、50 m BEV 扇区统一背景监督与 FP 评分。正例沿用原采集器 BBox/FOV 相交判定；当前纳入目标的未来轨迹不按 50 m 截断。
2. 原 42,531 个 vehicle、5 个 pedestrian 观测先恢复元信息；其中原 vehicle 中识别出 4,741 个自行车观测。保留的 37,790 个观测统一为 agent。摩托车、静止车辆和原筛选保留目标不因合类而删除。
3. 被移除目标仅作为 ignored_agents 保存其忽略几何及诊断元信息。BBox 内或距其中心 2 m 内不作为检测背景/FP；训练同时忽略最近 anchor cell；已匹配保留正例优先。匹配距离与 NMS IoU 配置未改变。
4. 单类复用 objectness，分类 loss 为零。Dataset、collate、loss、解码、NMS、指标、诊断缓存与可视化使用显式 schema。新 num_vehicles/车辆指标不可用，不冒充 Agent 数量。
5. 原 Risk 五维标签、mask、risk_agents 和模式标签未重算。9 个 train SHORT 帧失去所选关键目标，原标签保留并显式标记 SFT supervision unavailable；SFT 入口跳过这些帧，Grounding 保留。不把其改成 LONG，也不补选目标。

## 状态修复与不变量

原始四帧 Ego 速度、世界角速度和完整 Radar 安装偏移用于刚体速度公式，然后转换到当前 ego 坐标系。纠正了 mount BEV 投影。位置、BBox、绝对速度、朝向、保留 Agent Future、Ego Future、道路结构/拓扑、Risk 与原始模式字段的无关变化均被一致性检查禁止。

| split | 当前 Sensor Velocity 差异 >0.1 m/s 的帧 | 四帧历史差异 >0.1 m/s 的向量 |
|---|---:|---:|
| train | 291 | 1,158 |
| val | 79 | 298 |
| test | 101 | 386 |

Sensor Velocity 最大修复量 1.44056 m/s；派生中心视线径向速度最大变化 1.10015 m/s，平均绝对变化 0.01639 m/s。安装投影矩阵元素最大变化 0.146284（矩阵包含不同单位，不作为统一长度解释）。绝对 Agent 速度不改。

Power、Doppler、有效性 mask、坐标轴及路径保持原值。经用户授权，恢复缺失 prepared 路径：24,051 个文件来自 SHA256 一致的现存副本，3,310 个来自已保存雷达产品；全部 27,361 个文件逐一匹配训练时 SHA256，没有重跑雷达处理。详见 sensor_restoration.json。

原始 actor 元信息恢复缺失数为 0。原 recorded snapshot 范围内的对应核对通过；这不证明采集时没有漏记未进入 snapshot 的目标。role_name 未提供可确认的 parking 标记，不据此声称没有停车目标。静止/运动统计仅用于诊断，没有新增筛选规则。

## 验证

- 全量结构验收 PASS：8,835 帧、64 episodes、split/路径/实例对应、保留正例与忽略集合守恒。
- 专项测试 103 passed，覆盖边界相交、50 m/FOV、非零安装偏移及转弯、刚体速度、未来出域、单类前向/loss/梯度、匹配/NMS/解码、忽略梯度与 FP、collate、旧模型兼容及 SFT。
- 最后增加 forward schema 一致性检查后，检测相关回归 28 passed。
- 真实样本验收覆盖有正例、无正例但有忽略、SHORT 可用及 SHORT 不可用四种情况；读取的 Radar 数组/mask 与源样本一致，保留物理监督一致。
- 实际旧 Grounding checkpoint 严格加载成功；跨 schema 加载明确拒绝。只做反向，无 optimizer step，参数最大变化 0。acceptance.json 中随机初始化单类模型的 P/R 仅用于验证评估链路，不是训练成绩。
- test 仅进行迁移完整性检查，没有挑阈值或权重。

## 文件与追溯

- manifest.jsonl：最终发布；manifest.before_filter.jsonl：中间迁移审计副本。
- samples.jsonl：逐样本原始映射、速度/安装姿态修复及不变量核验。
- filter_samples.jsonl：移除类别、ID、保留 ID 和 SFT 可用性。
- migration_report.json、filter_report.json、validation.json、acceptance.json：汇总及验收。
- provenance.json：源 manifest、配置、源码及中间阶段摘要链；implementation_provenance.json：最终实现与验收源码快照。
- 配置：../../configs/l40s_qwen25_3b_agent50_v1.json；使用说明：../../docs/AGENT50_MIGRATION.md。

修改涉及 src/{collection_adapter,annotation_domain,categories,data,detection,losses,model,detection_metrics,pipeline,planner,visual_diagnosis,agent50_migration,agent50_filter}.py，scripts/{migrate_agent50,filter_agent50_labels,restore_verified_sensors,verify_agent50_dataset,diagnose_detection}.py，以及新版配置、专项测试与使用说明。

最终 manifest SHA256：`be5f7463c09926abf6f99284629e6c6b4c05f299fa499c89558cc06e4ac476db`。
原 manifest SHA256：`22dfa79b95f69a30590a5cd83ede82ca0bcac359873c58c92c2744eb56ae97c1`。

下一次训练必须配套新版 manifest 与配置，重新训练 Grounding，再由新版 Grounding 初始化 SFT。旧四类权重保留历史用途，不能加载为单类权重；本轮没有启动这两项训练。
