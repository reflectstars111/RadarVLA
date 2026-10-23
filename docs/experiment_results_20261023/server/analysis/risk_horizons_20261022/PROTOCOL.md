# KRS-v2：0.5/1/2秒风险与episode重划分

日期：2026-10-22。用户确认：前三维改为0.5/1/2秒，dmin和areq均改为2秒；train/val/test趋近合理比例。

## 五维定义

`[P(conflict within0.5s), P(conflict within1s), P(conflict within2s), dmin over0–2s, areq over0–2s]`。

前三维仍由真实未来事件生成0/1监督，网络输出事件概率；窗口均为累计区间，包含当前时刻，满足P0.5≤P1≤P2。几何延续原协议：当前moving Agent完整集合、ego名义CV/CTRV、不增加干预的最小框间距、安全距离0.5m、0.05s采样；areq在0—8m/s²、步长0.5的完整制动网格中寻找最小安全值，不假定风险对制动单调。

只缩短KRS计算窗口，不缩短6个未来轨迹结点（0.5—3s），不裁剪Agent、输入雷达或物理轨迹，不重新解折叠Doppler。SHORT仍使用2秒内的冲突事实：新版三个概率维度均参与，避免改完时间轴后误变成只判断1秒内。

版本标识：`risk_supervision_schema=local_agent_collision_0p5_1_2_v2`，模型和Planner的`risk_horizon_schema=0p5_1_2_v2`。新标签显式记录概率窗口及geometry_horizon_s=2。模型、Planner和数据版本必须一致；跨时间尺度checkpoint加载明确拒绝，即使strict=False。旧配置默认1/2/3秒及3秒几何，旧checkpoint行为保留。

## 重划分原则与结果

以帧数70%/15%/15%为目标，完整episode作为不可拆分单元。原test已被旧实验评估，本轮冻结其全部episode，不把test回流训练；此数据不宣称为从未查看过的新test。训练/验证在其余episode中重分配，并平衡风险正例、类别和Town分布。使用固定seed、单线程混合整数规划；每集合三个时间尺度均要求有正例。

| 集合 | 帧数 | 帧比例 | episode数 | 0.5秒正例 | 1秒正例 | 2秒正例 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 6868 | 69.79% | 40 | 31 | 52 | 144 |
| val | 1451 | 14.74% | 15 | 13 | 15 | 22 |
| test | 1522 | 15.47% | 9 | 12 | 28 | 40 |

全部9841帧、64个episode保留。每个episode及scene只属于一个集合，验证雷达历史文件不存在跨集合共享。原test的episode和sample ID集合保持不变；新划分、标签和路径以独立manifest发布，原manifest不覆盖。

0.5秒事件只存在于3个episode，重划分后每集合仅1个正例episode。验证AP从不可定义变为可计算，但不能把13个相关帧看成13个独立危险场景。重划分不能补充事件多样性，后续仍需要更多独立近碰撞episode。

## 接口与验收

- 数据加载输出显式risk_horizon_schema_id，混合时间尺度batch被拒绝；5维loss形状不变。
- 评估输出键为`0.5s`、`1s`、`2s`，并记录2秒几何窗口。某窗口没有正例时AP为null，保留样本/正例数和Brier。
- 预测JSON显式携带概率窗口和几何窗口；可视化按实际窗口显示名称。
- 标签迁移核对旧1秒/2秒事件与新版对应维度一致；Actor集合、道路、ego和他车轨迹保持不变，仅对新目录重定位传感器路径。
- 原1/2/3秒实验与checkpoint保留。旧实验外部队列/诊断脚本如硬编码`3s`，不得直接用于KRS-v2选择模型；新版应按实际窗口读取指标。
- 训练/验证episode重划分后，旧模型曾用过其中部分新val帧，不应把旧权重直接作为新划分上的独立泛化基线。新协议从头训练；任何显式预训练迁移另行记录。

## 位置与复现

- 隔离worktree：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-risk-horizons-20261022`，基于`c6c1d43a4ea9bd19aa3fe3ab25c0c88c4f8fa3a1`，主源码不修改。
- 新数据：`/home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_krsv2_0p5_1_2/manifest.jsonl`。
- 旧数据：`prepared/taskwise_moving_v1/manifest.jsonl`，原SHA256 `6f2fdedd3cee3abd1fdc6ab9952b267ce5008aa2994aa014241d6b37a867672e`。
- 独立Python：数据根目录`model_runs/risk_horizons_20261022/venv/bin/python`。
- 分配与审计：数据根目录`analysis/risk_horizons_20261022/`。`split_assignment.json`记录固定episode分配，`horizon_prevalence.json`记录改版前各窗口正例。
- 配置：`configs/l40s_qwen25_3b_taskwise_moving_{trodnet,hg,cnn2d}_krsv2.json`。
- 迁移入口：独立Python运行`-m radar_vla.risk_horizon_migration --manifest <原manifest> --output <新目录> --assignment <split_assignment.json>`。每100帧保存可恢复进度，代码/源数据/分配签名不一致则拒绝恢复。

本次完成标签/划分及接口验收，不自动启动新版正式训练。
