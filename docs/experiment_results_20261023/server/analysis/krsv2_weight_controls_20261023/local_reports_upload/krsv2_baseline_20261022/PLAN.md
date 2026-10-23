# KRS-v2：20轮正式基线与后续权重对照

用户已授权按“先新规格基线，再按五维贡献微调权重”执行。2026-10-22 20:25（北京时间）启动独立后台worker；本文件不代表20轮已完成，实际进度见服务器原生status与审计progress。

## 固定基线

| 项目 | 设定 |
| --- | --- |
| 模型 | T-RODNet，Grounder从头初始化，不加载旧checkpoint |
| 训练内容 | 检测、目标运动、五维Risk联合训练；无Planner/SFT/LLM训练 |
| 数据版本 | taskwise_moving_krsv2_0p5_1_2 |
| KRS | Pcol≤0.5/1/2秒；dmin、areq均按2秒计算 |
| 数据划分 | train6868 / val1451 / test1522；完整episode分组 |
| 轮次 / seed | 20 / 42；其他seed暂缓 |
| batch / 累积 | 4 / 2，名义有效batch8；每轮末尾不足窗口按原生逻辑处理 |
| LR | 2e-4；前5轮恒定，之后按优化器更新余弦衰减，最低比例0.1 |
| 精度 | BF16，与原超参实验一致 |
| 损失 | 沿用原损失、系数、Focal和物理量归一化；本轮不调权重 |
| 选模型 | 保留全部逐轮checkpoint；按检测/运动/五维Risk联合审查，不只看某个AP |
| 测试集 | 本轮不评估，不参与权重或checkpoint选择 |

原生每轮约1717个训练batch、859次优化器更新，总预算17180次更新。

## 需要记录的证据

- 每个训练batch：原始与加权损失；三个冲突维度的有效标签数、正例数及正负Focal贡献；dmin危险/非危险贡献；areq正制动需求/其他贡献。
- 每轮最多8个已有前向batch：优先第一批、各窗口首次正例、首次正制动需求及周期样本，记录各维风险、总Risk和总检测对encoder特征与共享Radar tokens的梯度范数。
- 梯度为未作累积缩放/裁剪的batch目标对特征的梯度，不冒充参数梯度，也不当作全训练分布均值。没有额外优化器更新。
- 每轮完整val：三个窗口AP/Brier、危险对象检出、固定0.5工作点的Risk漏报/误报、正制动需求MAE/近零比例、危险dmin误差及逐episode结果。
- 0.5秒val正例只有13帧/1个episode，指标变化不能单独作为权重决策依据。

## 实验隔离与资源

源码worktree：
`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-risk-horizons-20261022`

源码提交：`ecfb71ce53f1aba12b54722d5243cc34a43f73fe`。训练观测器独立保存在数据分析目录，运行脚本哈希记录在plan.json中；主源码及旧实验不改动。

正式输出：
`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/krsv2_baseline_20261022/train_seed42_20ep`

计划、日志与审计：
`/home/luxinlan/data/lxl_data/RadarVLA/analysis/krsv2_baseline_20261022`

启动GPU1，绑定UUID `GPU-4fe38366-e143-fe49-58b4-0b15e5d4de97`；worker PID与命令见launch.json。CPU线程1，data workers0，nice19，ionice idle；显存预算6GiB。显存余量低于8GiB仅退出本worker，不操作其他进程；其他用户进程存在本身不触发暂停。

如果资源退出，保留日志和已完成epoch的checkpoint；恢复前核对源代码、运行脚本、配置和数据指纹，不绕过原生resume校验。

## 已完成的启动验收

- 观测器3项CPU测试通过：启用前后loss、参数梯度、随机状态、一次优化器更新完全一致；缺失标签的Focal分维归约与原生loss一致。
- 真实batch4覆盖危险、空目标和超过8目标的样本；前向、反向、验证记录成功，loss与梯度有限。
- 输出Risk为5维，检测候选保持3456；预检无优化器更新，峰值显存约2.16GiB。
- 正式运行前重新验证新版manifest SHA256：`c6fa46a048fd0577ed0b2dc7a811131eac5d345e75f571749b2b3a82283e1672`。
- 原生训练启动时仍执行雷达输入文件指纹检查；准备/校验阶段不算完成训练更新。

## 20轮完成后

先审查训练/验证曲线与五维损失、正例贡献、共享特征梯度。按证据依次选择短时冲突项、正制动需求项或Risk相对检测总权重中的一组，开展同seed、同划分、同初始化和同预算的单变量对照；不一次改多个因素，也不自动扩到100轮。

对照既看AP，也看Brier、误报、危险召回、运动误差和正制动需求MAE。没有联合收益则不接受权重修改。当前不预先指定“最佳权重”，也不自动触发任意权重扫描。

首轮完成后再按实测速度估计剩余时间；CPU/IO和服务器共享负载会影响吞吐。
