# Stage 3：快速有效响应 RL

Stage 1 的检测头预测周围车辆、行人等目标的状态与未来轨迹；Stage 2 规划自车轨迹；Stage 3 从 SFT 检查点微调规划策略。检测器冻结，HF 模型更新原有 LoRA／适配器。目标 ID、地图、未来标注只用于离线事件与奖励，不进入策略输入或 LONG/SHORT 路由。

## 时间与执行

响应时间与响应质量独立记录。响应时间是从检测报告到规划轨迹相对当前执行计划开始出现明显、持续变化的时间，不要求先确认整条未来轨迹安全：

```text
reaction_s = first_generation_s + first_waiting_s + trajectory_change_onset_after_first_adoption_s
```

`response_clock` 只读取自车状态、新轨迹和当前执行轨迹，不读取目标未来、地图或其他 GT。默认纵向加速度变化绝对值至少 0.5 m/s²（制动与加速均可），或横向偏移至少 0.3 m，并持续至少 0.1 s，过滤微小抖动。持续条件确认后，把最初越过阈值的时刻作为起点。记录 `response_detected`、`action_onset_s`、`reaction_s` 和 `response_time_reward`。

例如生成耗时 0.1 秒，轨迹在可用后 0.2 秒开始制动，则响应时间为 0.3 秒；后面 3 秒以外是否有标注，不会改变这个结果。这个值是规划动作的预计启动时间，并非实车执行日志。新轨迹仅在远期改变时，不能把其生成完成时刻直接当作动作响应时刻。

一次事件内按时间顺序回放多帧：检测采用从场景首帧开始的 FIFO，规划默认采用事件内 FIFO，可通过 `planner_observation_policy=latest_ready` 在每次空闲时只处理最新已就绪观察。后续规划的等待和生成时间体现在整段执行轨迹中，不能重置首次检测时钟。前一条动作持续执行到下一条完整规划可用；新动作在届时的自车位置、朝向上重新锚定。物理检查从事件触发观测开始，覆盖漏检等待、检测计算、规划排队和生成期间的行驶。

`measure_plan` 是预测和 RL 共用的生成、解码计时入口。训练先在计时外采样动作，再把同一 token 序列送入同一个逐 token 推理路径，测量该动作在部署时的生成耗时；Categorical、采样、log probability 和奖励计算不计入该时间。它是部署路径重放估计，不能当作实际随机采样服务的总耗时。CUDA 在边界同步。输入文件读取、预处理／传输、传感器采集和执行器延迟未包括在内。墙钟耗时依赖设备与负载。

`predict` 在 Grounder 和检测解码后报告检测，再按样本生成规划。原始 `plan` 保持 SFT 解码契约；新增 `executed_plan` 提供延迟后的轨迹、执行时域和动力学诊断，`valid=false` 的候选不能当作可执行命令。`detection_seconds`、`generation_seconds`、`waiting_seconds` 分别记录这些阶段。普通 `predict` 仍按独立观测帧输出，不保存连续场景的执行状态；`evaluate-rl` 使用事件内执行状态。

## 安全与有效响应

安全检查覆盖到 `首次可用时刻 + max(统一安全时域, 完整执行时域)`，后续重规划会扩展到最后采用动作的完整尾段。默认统一安全时域为 3 秒，不能通过生成变慢把碰撞移出检查范围。这项检查用于独立的安全诊断。缺少尾段 GT 时标记 `safety_status=unknown`，保留已经算出的响应时间和时间奖励，不要求为计时另行补齐更长标注；已经观察到的碰撞仍保留并处罚。

SHORT 保留原生 1 秒样条，在末端显式追加加速度、jerk 有界的平滑停车段，再保持静止。该段保持位置、速度、加速度连续；它参与奖励和预测执行输出，不改写监督时域。原生动作本身不可行时仍失败。默认 64-bin 词表的合法 SHORT 已有正奖励回归例，未降低 3 秒安全时域。

`effective=True` 表示有证据支持安全且针对目标有益的响应，与 `response_detected=True` 不同；前者还需要满足：

- 相对于当前执行轨迹，触发目标确实构成威胁。
- 纵向加减速或侧向变化持续达到阈值，且该变化本身改善目标间距。通过分别移除纵向／横向变化的几何反例检查贡献。原始轨迹变化时间仍保留，针对目标有益的启动另记 `effective_action_onset_s`／`effective_reaction_s`，不能把前面的有害制动称为成功避险。
- 整条延迟执行轨迹改善目标最小间距，所有已标注目标均满足安全距离，tracking 覆盖完整。
- 初始速度、每次规划切换的速度／加速度连续性，以及完整样条的加速度、jerk 合格；检查控制点、切换点、停车段关键时刻。
- 车体采样点位于明确关联左右边界的车道并集内。缺少明确道路边界时不能标成已验证安全，但不抹掉响应时间。

自然三次样条的首端加速度为零，因此在已有制动过程中切换到新样条可能不连续。当前实现明确判失败，没有车辆控制器替模型修复这种动作。无效后续输出继续保留前一条执行轨迹用于碰撞检查，并保留输出失败惩罚。

几何检查采用离散时刻的有向框与道路采样点，不构成连续时间安全证明；目标未来来自录制数据，不会因自车动作改变。

## 奖励与学习

无 GT 依赖的响应时间奖励：

```text
response_time_reward = 2 - latency_weight × reaction_s / response_deadline_s
reward = response_time_reward + quality_adjustment
```

存在明确证据的碰撞、动力学无效等失败仍采用共同基础惩罚 `-6`，再扣除首次检测到该事件最后一条输出可用之间的延迟成本（包含后续规划），避免“延迟到发生碰撞反而比无效输出得分高”。只有未来／道路覆盖未知时，保留时间奖励，不把未知当作失败，也不把它计为已验证有效。缺标的动作同样能按早／晚变化得到不同奖励；不会因未知尾段继续扣除已经响应后的排队时间。已知无威胁控制事件中，合理保持行驶得正奖励，不必要的制动／偏移受罚。默认 train 和 val 都需要危险与无威胁事件，未知威胁不能充当安全控制样本。

每个事件采样 G 条完整的多帧动作序列。使用其他 G−1 条序列的平均奖励作为独立基线：

```text
A_i = R_i - mean(R_j, j != i)
L = mean_i[-A_i × sum_decision,token log πθ + β × conditional_SFT_KL]
```

一组采样完成后只更新一次；参考 SFT 固定。采样、重评分共用语法和温度，逐 token 校验概率，关闭 dropout。HF 在 eval 模式下的梯度重评分仍实际使用检查点重算；基础权重保持冻结。没有 PPO clipping、价值网络或过期样本复用。时间不可直接微分，通过策略梯度学习。相同奖励的组响应梯度为零，日志记录非零优势数量。

先评估第 0 轮 SFT 并保存 `best.pt`，之后按已知安全失败率、危险响应成功率、控制场景误动作率、平均奖励顺序选优。覆盖未知单独报告，不冒充安全失败或成功；都缺少尾段证据的候选仍可按响应奖励比较。因此差于 SFT 的 RL 轮次不会仅因“是第一轮”成为最佳模型。时间噪声仍可能影响排序，需要相同设备负载下重复评价。

## 数据与事件

输入为连续 prepared manifest，带稳定目标 ID、时间戳、tracking 覆盖与道路边界。v2 事件支持初始危险、已可见目标从安全变危险、首次出现、重新出现，以及同一目标后续独立危险事件。未知标注不当作已经恢复安全；同一目标重叠的有效事件窗口会拒绝。窗口被数据末尾、缺帧或下一事件截断时标记 `response_window_complete=false`。

原有三字段事件仍可读取：

```json
{"event_id":"appearance_000001","target_agent_id":"track_19","sample_ids":["before","appearance","next"]}
```

v2 另外记录 `trigger_sample_id`、`event_kind`、`trigger_timestamp_s`、`episode_index` 与覆盖信息。目标关联使用每帧全部 GT 与检测框的一对一中心匹配；检测局部编号不是跟踪 ID。GT 只用于离线评分，检测漏检不会通过规划 RL 获得修复梯度。

规划输入的 `future_times_s` 固定为 checkpoint 的原生协议。安全诊断可用 `reward_record` 补充现有 GT（响应计时不依赖它）：仅拼接同场景／split 的有姿态标注帧，通过 `ego.pose` 转到触发观测坐标系。保留原始时域和掩码，间断处插入未知标记，不外推目标运动。没有姿态时保留原标注，也可提供当前坐标系的 `response_evaluation`，包含更长的 `future_times_s`、tracking、ego／agents 未来数组；原有 GT 前缀必须完全一致。缺少这些证据时尾段不能判安全，但仍能测量并训练响应时间。默认最长拼接 10 秒，超过仍为未知。

## 命令与产物

```bash
.conda/bin/python -m radar_vla prepare-rl-events \
  --manifest /data/radar/labeled.jsonl \
  --output /data/radar/response_events.jsonl \
  --config configs/response_rl.json

.conda/bin/python -m radar_vla train-rl \
  --checkpoint /data/runs/sft/best.pt \
  --manifest /data/radar/labeled.jsonl \
  --events /data/radar/response_events.jsonl \
  --config configs/response_rl.json \
  --output /data/runs/response_rl --device cuda --epochs 3

.conda/bin/python -m radar_vla evaluate-rl \
  --checkpoint /data/runs/response_rl/best.pt \
  --manifest /data/radar/labeled.jsonl \
  --events /data/radar/response_events.jsonl \
  --output /data/runs/response_rl/test_response.json --device cuda
```

仅支持单设备；HF 基础权重使用 checkpoint dtype，适配器与 Grounder 为 FP32。`last.pt` 保存优化器、RNG、配置、源码、数据、事件和 SFT 内容指纹。恢复加 `--resume`，仍指定原 SFT；`--epochs` 为目标总轮数。配置／数据变化拒绝恢复，不承诺墙钟或训练轨迹 bitwise 重现。

`event_coverage.json` 保留全部事件。有效响应率的分母包含已标注危险事件中的漏检、未知覆盖与失败；`response_p50_s/p95_s` 统计所有可计时的轨迹变化，包含质量未知或后来失败的动作；`effective_response_p50_s/p95_s` 单独统计已验证有效的响应。另报 `safety_unknown_plans`，两种统计不能混用。碰撞率只对应有可评分执行轨迹的样本，同时报告可评分数量，不能把漏检当作零碰撞。`validation_epoch_0.json` 保存 SFT 基线，`history.json` 记录每轮训练与验证，test 不参与优化或选优。

## 验证边界

回归覆盖了真实语法 token 的延迟尾碰撞、无效动作抢先计时、已知碰撞被缺标注覆盖、跨帧排队、切换不连续、SHORT 完整执行、HF 梯度检查点、标注拼接与训练恢复。受控有限候选策略的真实奖励／REINFORCE 测试证明更偏好快速安全动作且降低控制场景误制动；这不是完整语言模型驾驶性能提升的证据。

每个独立事件从记录的自车状态和明确 CV/CTRV 基线开始，事件内保留动作与规划积压；不同事件不共享未记录的真实活动计划或规划队列。传感器观测固定来自录制数据，未实现动作改变世界后的再观测、车辆控制器或 CARLA 联机闭环。需要真实数据和训练好的 SFT 检查点才能验证实际响应改善。
## Optional adoption and scheduling repair (2026-10-23)

`configs/response_rl_adoption_v1.json` enables a fixed 0.5 s quintic adoption
transition and `planner_observation_policy=latest_ready`. Both are opt-in;
legacy configs keep zero transition and FIFO. Export, response timing, reward
and replay share the prepared trajectory. Infeasible bridges are rejected;
replay continues the prior execution. Planner replay can skip stale pending
observations without resetting the first-detection clock; perception remains
FIFO. Bridge construction time is not yet included in generation timing.
See [the repair design](BOTTLENECK_REPAIR_V1.md) for compatibility, validation,
and the distinction between implemented repairs and proposed model changes.

## Continuous action experiment compatibility

`parallel_acceleration_v1` currently supports SFT and offline prediction/evaluation.
`train-rl` / `evaluate-rl` and token policy scoring reject this representation:
a continuous policy distribution, action density and matching rescore have not
been implemented. The token baseline remains supported. Do not reuse token
log-probabilities for continuous actions or interpret this restriction as completed
continuous-policy RL. See [the experiment runbook](BOTTLENECK_8XL40S_RUNBOOK.md).
