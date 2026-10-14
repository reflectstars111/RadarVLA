# 批次 smoke 首次实测：FAIL

2026-10-14 只运行了一次 Scenario2 资格 pilot；失败后自动停止，未运行三个正式 smoke episode、雷达批次或 1 万帧正式采集。原始诊断保存在 `/home/luxinlan/data/lxl_data/RadarVLA/smoke/scenario2_pilot/`，状态在同级 `status.json`；未删除或改写该 episode。

## 已记录事实

- 官方 Town02 tiny route 0 长 509.1 m。候选 trigger 为 `(123.181, 187.609)`，位于 road 5/lane 1 的起始处；离线 CARLA 地图查询曾报告 `get_waypoint_in_distance(25)` 前进 25.0 m。
- Scenario2 被选择并激活，`following_started` 发生于 CARLA frame 2608，前车 actor 500 于 frame 2609 恢复到路面。没有 `braking_started` 或 `scenario_completed`。
- frame 2609：Ego `(123.104, 187.273)`、5.86 m/s、制动 0.3；前车 `(123.181, 187.609)`、0 m/s。中心距离约 0.35 m，两个 bbox 纵向半长合计约 4.75 m，恢复时已经重叠。
- Leaderboard 记录 1 次与 actor 500 的车辆碰撞，路线完成率 34.76%。随后采集器报 `ego has no current driving waypoint`，episode 状态为 `failed`。

因此 pilot 未取得资格证书，完整 smoke 和正式批次均未放行。前车实际恢复在 trigger 附近，说明先前候选预检测得的 25 m 没有反映场景实际落点。**运行时 helper 的返回距离未被记录**；“默认 waypoint 投影落在路口，导致 helper 提前停止”与代码和几何相符，但仍是待下次 pilot 核实的解释。

## 已做的静态修正

- Scenario2 候选预检改为使用场景运行时相同的默认 `get_waypoint()` 投影，并要求它与官方路线车道一致、不在路口；候选还须离最近的非 `LANEFOLLOW` 路线点至少 12 m。
- `events.jsonl` 合并保留 ScenarioManager 的选中帧，同时补入 Leaderboard 的场景名、trigger 与 actor ID，避免真实事件选帧被缺失字段误拒。
- pilot 放行检查核对前车真实纵向/横向位置、bbox 间距、事件顺序、运动、Ego 在前车附近制动及无碰撞。

这些修正仅完成离线测试，**尚未通过新的 CARLA pilot**。原失败目录保持原样；下一次人工授权的小批次应使用新 `--smoke-name`，由脚本创建新目录，先做一轮 Scenario2 pilot。失败则再次停止，不连续试触发点。
