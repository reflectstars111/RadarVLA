# Risk 输入核查与完整 Oracle 对照

当前训练和部署使用 Grounder 的预测风险。真实风险来自未来 GT，只用于显式 Oracle 诊断，不能替换部署输入。

## 输入契约

五维顺序固定为 `[pcol_1s, pcol_2s, pcol_3s, dmin_m, areq_mps2]`。

- Local 协议下，Dataset 只读取 `local_risk`，并使用对应逐维 mask；不回退到 Global 标签。
- Grounder 输出和监督目标都使用物理单位。送入Planner的可学习Risk特征映射层之前，仅除以一次 `[1,1,1,coordinate_limit_m,10]`；当前 coordinate_limit_m=80。
- Grounding 回归损失中的距离/制动尺度仅用于损失计算，不改变预测输出单位。
- 无效 areq 的占位0不表示无需制动。完整 Oracle 对照要求五个分量全部有效、有限；否则在对照两侧同时排除。
- 不能把碰撞头0.5阈值召回直接当作Planner模式召回；当前模式由Planner生成。

这些标签使用当前速度/转率的反事实Ego轨迹，与当前保留Agent的GT未来几何计算；它们不是随后生成的任意Ego候选轨迹的风险。静止Ego的nominal risk很低，与生成加速轨迹后发生冲突并不矛盾。输入风险质量和候选轨迹安全需要分别检查。

## 完整轨迹干预入口

在项目根目录、现有 lxl_radarvla 环境中运行。先检查GPU剩余显存和任务利用率，显式选择有足够余量的卡。

```bash
CUDA_VISIBLE_DEVICES=<spare GPU UUID> OMP_NUM_THREADS=1 \
nice -n 19 ionice -c 3 /home/luxinlan/.conda/envs/lxl_radarvla/bin/python \
  scripts/diagnose_risk_rollout.py \
  --checkpoint <original sft/best_planning.pt> \
  --manifest <original manifest.jsonl> \
  --baseline <prior audit directory containing protocol.json and samples.jsonl> \
  --output <new diagnostic directory> \
  --memory-fraction .22
```

基线目录必须是保存逐帧原始结果的目录，例如原报告的 `e5_complete_validation/`，不能只提供汇总报告所在目录。脚本检查源码、manifest和checkpoint摘要，实际加载canonical Dataset，并逐token复现代表样本后复用原预测侧结果。

默认逐帧、原生greedy、无KV cache、最多512新token；仅替换五维Risk，Radar/Ego token和指令保持不变。可用 `--shard-count N --shard-index i` 划分不重叠样本；每个进程在独立GPU上运行，不使用批量解码来代替原生batch=1结果。

默认启动剩余显存门槛24GiB、运行剩余显存门槛8GiB。低于运行门槛时退出并保留已完成结果，不终止其他任务；相同协议可续跑。

## 判断结果

同时报告模式召回/误报、教师要求时刻覆盖、共同已覆盖时刻的配对轨迹误差。Oracle改变模式时可能改变轨迹时域，不能把遗漏后续时刻当作零误差。

离线碰撞几何仍依赖切线推导的车身朝向和固定他车GT未来；不同规划时域上的碰撞率不能直接称作安全收益，需要同时间范围的案例回放。它不等同CARLA闭环运行或实采雷达驾驶验证。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/luxinlan/.conda/envs/lxl_radarvla/bin/python \
  -m pytest -q tests/test_risk_rollout.py tests/test_local_risk.py tests/test_taskwise_supervision.py
```

## 独立的SFT训练控制

训练入口另提供两个默认关闭的控制，用于检查规划梯度是否影响物理风险表示：

| 开关 | Grounder参数更新 | Risk输入处的规划梯度 | Radar token处的规划梯度 |
| --- | --- | --- | --- |
| 无开关：joint基线 | 是 | 保留 | 保留 |
| `--freeze-grounder` | 否；保持eval模式 | 不更新Grounder | 不更新Grounder |
| `--detach-risk-condition` | 是；保留物理监督 | 在Risk输入处阻断 | 保留 |

两个控制只能用于SFT，不能同时开启。它们不替换预测Risk、不引入真实风险输入、不修改损失权重或检测阈值。控制值写入训练协议，恢复时不允许静默切换。

冻结Grounder不等同于只冻结risk_head：只冻头部、但继续改变其输入特征，仍可能改变风险预测。另一方面，Risk detach也不保证Risk保持不变，因为Radar token路径仍会更新共享特征。

```bash
# 其余数据、配置、Grounding初始化、seed、有效batch、训练步数必须匹配。
python -m radar_vla train <matched arguments> --freeze-grounder
python -m radar_vla train <matched arguments> --detach-risk-condition

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /home/luxinlan/.conda/envs/lxl_radarvla/bin/python \
  -m pytest -q tests/test_risk_training_controls.py
```

本轮输入契约核验没有发现接线错误。风险估计质量、模式条件适配、停车轨迹保持需要受控训练验证；诊断本身不改变推理主路径，训练控制也不代表风险预测已修复。

## 累计冲突概率的一致性修正

`Pcol¹ˢ/Pcol²ˢ/Pcol³ˢ`对应嵌套的累计冲突事件，应满足 `P1≤P2≤P3`。三个独立输出头可能违反这个约束。`risk_projection=isotonic`对三维概率做等权欧氏保序投影；没有待训练参数、不读取GT、不拟合校准器，保留`dmin/areq`及单位。Tiny与HF Planner在同一个入口应用修正，然后才做上述一次单位缩放。

旧checkpoint缺少字段时默认`none`，保持原输出。新训练可选择`configs/l40s_qwen25_3b_taskwise_local_isotonic_v1.json`；对已有checkpoint进行显式离线对照时，在`evaluate`或`predict`后附加`--risk-projection isotonic`。省略开关时沿用checkpoint配置。输出中的`risk`始终保留Grounder原始估计，`conditioning_risk`记录实际送入Planner的物理五维向量，`risk_projection`记录处理方式。

常规评价的Risk指标仍指Grounder原始输出；保序修正后的概率指标单独报告。该约束只能消除累计概率的矛盾，不能保证各时间窗口的概率已校准，不能替代危险正例召回、完整轨迹覆盖和停车保持验证，也不能将nominal Risk改成候选轨迹风险。

启用保序输入进行新训练会改变优化问题；原始joint/frozen/detached一轮控制均使用`none`并保存独立源码摘要，不与保序配置混在一个对照中。

本轮原5epoch检查点的1461帧完整验证中，保序输入消除158帧累计矛盾，但模式未变，共同点ADE从1.1295m变为1.1404m，配对episode bootstrap差值区间为[0.00357,0.01928]m。因此不推荐在当前检查点默认启用；保留`none`及原模型。新增配置用于未来训练对照，不能把约束满足当作规划质量已经改善。
