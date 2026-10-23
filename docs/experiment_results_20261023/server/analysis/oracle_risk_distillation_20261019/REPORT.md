# RadarVLA：预测Risk追赶Oracle的受控训练报告

**结论：本轮完整GT风险教师KL未优于仅物理监督。** 同批完整标签样本的SHORT召回均为11/46（23.9%），原模型为10/46（21.7%），完整GT Oracle为44/46（95.7%）；KL组误报31帧，物理组26帧。没有替换默认模型。

完成时间：2026-10-19T14:19:44.146332+08:00（Asia/Shanghai）。

本轮从原5epoch SFT检查点继续只训练风险聚合与MLP；冻结Radar/Ego编码器、检测器和Planner全部参数。两组唯一目标差异为是否增加Oracle模式KL。GT Risk只用于教师与物理监督，Student从未接收GT Risk或GT文本前缀。

## 实验协议

- train共6728帧SFT样本，val共1461帧；完整Risk的val为1407帧。train/val的episode不重叠，test未加载。
- 两组均训练3epoch，batch=8，lr=0.0003，seed=42，AdamW decay0.01，梯度裁剪1。KL权重0或1.0，温度1.0。
- 仅25285个风险参数可更新；原物理loss、正例权重、阈值和projection=none均不变。固定最终epoch，不按val调参或选epoch。
- 8个Radar token和1个Ego token由固定编码器提取并缓存；逐帧证明缓存风险路径与完整传感器路径输出逐位相同。val基线Risk/模式与原报告逐帧复现，Oracle模式混淆矩阵与前轮完整Oracle对照一致。
- Student使用正常反传穿过冻结Planner，教师stop-gradient。两组结束后，全部冻结参数及缓冲区的完整摘要与开始时相同。

## 完整Risk同批验证集的模式效果

| 条件 | SHORT召回 | SHORT精确率 | LONG误报帧数 | 与Oracle分层一致率 | 与Oracle KL | 召回差距闭合比例 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.2174 | 0.3704 | 17 | 0.6074 | 0.0960 | 0.0000 |
| physical_only | 0.2391 | 0.2973 | 26 | 0.6155 | 0.0937 | 0.0294 |
| physical_plus_kl | 0.2391 | 0.2619 | 31 | 0.6136 | 0.0984 | 0.0294 |
| GT Risk Oracle | 0.9565 | 1.0000 | 0 | 1.0000 | 0.0000 | 1.0000 |

分层一致率分别计算Oracle SHORT和LONG样本的一致率再平均；避免总体一致率被大量LONG抬高。召回差距闭合比例只是同批点估计，可能为负或超过1，不代表轨迹质量。Oracle也会出错，不能把蒸馏到它视为获得全部真实风险信息。

| 条件 | 全1461帧SHORT召回 | 全量LONG误报 | Pcol³ˢ AP | Pcol³ˢ Brier | dmin MAE/m | areq正例MAE/(m/s²) | 顺序违反 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.3571 | 18 | 0.4069 | 0.0369 | 8.3567 | 1.2896 | 158 |
| physical_only | 0.3750 | 32 | 0.4160 | 0.0359 | 6.9523 | 1.4854 | 27 |
| physical_plus_kl | 0.3750 | 38 | 0.4260 | 0.0363 | 6.9642 | 1.5551 | 8 |

全部SFT和完整Risk子集的SHORT分母不同，不能混用。Pcol各窗口、各轮学习曲线和逐帧预测均保存在JSON中。

## 可以作出的判断

加入KL相对纯物理监督：同批SHORT召回23.9%→23.9%，LONG误报26→31帧；Pcol³ˢ AP 0.4160→0.4260，Brier 0.0359→0.0363。

按10个episode配对bootstrap，KL相对纯物理监督的SHORT召回差95%区间为[0.0, 0.0]，LONG误报率差区间为[0.0, 0.012562814070351752]。仅一次初始化和短预算，不能推广为跨seed结论。

模式改善同时伴随物理量退化时，不能把预测Risk称为真实Risk的有效替代；必须共同检查AP/Brier、距离和正制动误差。若两组都不能接近Oracle，也不能仅凭这一轮就断言雷达缺少信息，需要进一步区分风险头容量、监督平衡与不可观测未来。

本轮这里完成的是全val风险与BOS模式评价，不是完整自由轨迹rollout。即使模式接近Oracle，也不能据此宣称轨迹误差、碰撞或闭环效果已改善。原模型未覆盖，任何新检查点均未自动提升为默认。

## 产物与复核

- [逐项结果](comparison.json)、[物理监督学习曲线](physical_only_history.json)、[蒸馏学习曲线](physical_plus_kl_history.json)、[配对统计](mode_bootstrap.json)。
- [源码补丁](risk_mode_distillation.patch)、[源码版本](source_versions.json)、[验证记录](VALIDATION.md)、[实现说明](code_review/docs/ORACLE_RISK_DISTILLATION.md)。
- 特征缓存和两个推理检查点保留在服务器model_runs；可审阅产物不包含权重、传感器数组或环境。推理检查点保留原Planner状态并记录risk_refinement元数据，不能用于原训练入口resume。
- 仿真雷达、Local Agent、nominal Ego风险和离线评价的限制沿用原报告；GT未来只作为监督和诊断信息。

## Pcol3教师覆盖补充对照

扩大Pcol3教师监督后，SHORT召回没有超过仅物理监督。当前短预算结果不支持将这套蒸馏方案作为真实风险的有效替代。

Pcol3教师学生在同一1407帧子集上SHORT召回23.9%，LONG误报39帧。详见[完整补充报告](../oracle_risk_pcol3_teacher_20261019/REPORT.md)。

训练集477个Pcol3危险正例中有191个因五维标签不完整而未参与原KL监督；它们一直参与有效维度物理监督。新对照改变教师内容和监督覆盖两项，不能把效果差异全部归因于覆盖率。

错误审计也表明：虽然Oracle SHORT只占验证完整子集的3.1%，它已贡献原模型约70.7%的平均KL；因此不能仅凭LONG样本多就认定蒸馏目标被LONG主导。训练KL下降而验证KL未相应改善，也不支持把单纯延长训练当作已验证修复。下一步应定位冻结特征能否区分危险正负例，以及风险学习目标是否与模式决策方向一致，再决定解冻编码器或调整监督。
