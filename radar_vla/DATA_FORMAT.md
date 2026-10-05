# 数据与物理坐标契约 v2

原始归档是 UTF-8 JSONL，schema 为 `radar_frame_v2`；准备后 schema 为 `radar_vla_v2`。运行 `python -m radar_vla synthetic-frames --output /new/path` 可生成完整、可实际读取的格式样例。此命令的数据是明确标记的合成格式样例，不是仿真器性能数据。

## 原始字段

| 路径 | 类型与语义 |
|---|---|
| schema_version | `radar_frame_v2` |
| sample_id / scene_id | 非空且 sample_id 全局唯一；一个 scene 只能属于一个 split |
| split / timestamp_s | train、val 或 test；一个公共时钟的秒值，每个 scene 时间戳唯一 |
| coordinate_frame | world 或 ego；作用于没有单独声明坐标系的几何标注 |
| ego.pose | `[4,4]` ego→world 齐次矩阵，右手系，平面车辆姿态 |
| ego.velocity / acceleration | `[3]` 或 `[2]`，可用 ego.vector_frame 覆盖坐标系；绝对地面速度、加速度 |
| ego.yaw_rate / box_size | rad/s；正的 length,width[,height] |
| ego.future_trajectory | 列表，每点含 timestamp_s、position、yaw、valid；只用于教师，可缺失 |
| sensors.radar | 下述 Radar 契约 |
| sensors.camera | 按 front_rgb、left、right 等名称索引的字典；每项 path、timestamp_s、intrinsics `[3,3]`、T_ego_sensor |
| sensors.lidar | pointcloud NPY 路径、timestamp_s、T_ego_sensor；点数组 `[N,D≥3]` 前三列 xyz |
| agents | 当前可靠标注的列表，空列表表示当前无人；观测推理可不含此字段 |
| agents[].id | 稳定字符串/整数 ID；同一帧不能重复，不按邻近位置推断跨帧 ID |
| agents[].bbox3d | center `[3]`、size `[length,width,height]`、yaw（rad） |
| agents[].heading | 可选；存在时必须与 bbox3d.yaw 一致 |
| agents[].velocity | 绝对地面速度 `[2]` 或 `[3]` |
| agents[].future_trajectory | 与 ego future 相同字段；agent.coordinate_frame / future_coordinate_frame 可明确覆盖 |
| map.lane_centerlines | 列表：id、points `[N,3]` 或 `[N,2]`、left_boundary_id、right_boundary_id |
| map.lane_boundaries | 列表：id、points；道路宽度使用中心线法向与两侧边界交点计算 |
| map.traffic_elements | 列表：id、type、position，可含 heading、灯态或速度限制等属性 |
| map.route_lane_id | 可选，指定生成局部走廊的 lane；缺失时选择当前最近的前向 lane |
| map.route | 可选：centerline、goal_s_m；只有明确 route 才评价路线进度/完成比例 |
| language.instruction | 字符串；UTF-8 字节数不能超 planner.max_instruction_bytes，超限明确报错 |

Power 和 Doppler 每帧数组 `[R,A]`，正式配置 `[256,107]`；range_m、azimuth_rad 的长度与维度严格一致。数据保留原生栅格，不 resize。Camera、LiDAR 是实际解码并校验的归档/离线教师资产，在线网络按文档只读取 Radar、ego、instruction。

## Radar 和同步

每个 radar 必须含：

```json
{
  "power": "sensors/power.npy",
  "unfolded_doppler": "sensors/doppler.npy",
  "range_m": [1.0, 2.0, 3.0],
  "azimuth_rad": [-0.2, 0.0, 0.2],
  "timestamp_s": 12.3,
  "T_ego_sensor": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

上例仅展示字段，真实 256×107 输入需填全部标定坐标。Power 为非负线性功率，不能将 dB 值直接当线性值输入。Doppler 为 m/s，接近负、远离正。

可以同时保存 folded_doppler。只有 folded 时，额外要求：

- `doppler_prior`：独立因果速度先验 NPY，与 RA 同形。
- `doppler_prior_source`：causal_tracker、multi_prf 或 sensor_firmware。禁止由未来 GT 产生该先验。
- `max_unambiguous_velocity`：正的无模糊速度半区间；混叠周期为两倍此值。
- `max_prior_error`：先验允许误差。模糊候选或超误差 cell 置为无效，不伪造唯一解。
- `doppler_valid`：可选同形布尔/0-1 NPY；训练编码及实测物理监督都使用它。

`raw_points` 支持 `[N,D≥3]` NPY，`raw_cube` 支持至少三维的数值/复数 NPY；保留原始数据及校验，不在缺少采集标定时臆造 FFT 处理。传入的路径都必须真实存在。

`T_ego_sensor` 将 sensor 点变换到该时刻 ego；`ego.pose` 将 ego 变换到 world。默认最大传感器与帧时间差 0.05s，但容差内也不会把异步数据当作同一位姿：时间差超过 1e-6s 必须提供 `sensor.ego_pose_at_timestamp`。异步 Radar 额外提供：

```json
{"ego_state_at_timestamp": {"velocity_world": [10.0,0.0,0.0], "yaw_rate": 0.1}}
```

Radar 观测不能晚于 frame_t。历史 `time_offsets_s` 相对于 frame_t，可全部小于等于零；延迟雷达的末项允许小于零。运动补偿使用实际测量时刻位姿，sensor velocity 包含 yaw rate 与安装杆臂叉积。异步 Radar 没有同步目标 GT 时，不能用当前 bbox 关联为当前 Doppler 教师，对应监督会屏蔽。

道路与风险几何是明确的地面平面模型：ego pose 保持竖直轴（容差 1e-4），ego/agent 速度及 ego 加速度的 z 分量不超过 1e-4。传感器外参支持任意 SE(3)；倾斜雷达的水平 LOS 投影保留其范数，不当成单位水平向量。

## 历史、未来与标注覆盖

`prepare-frames` 按 scene/time 排序构建历史，split 不可跨 scene。默认 4 帧、相邻最大间隔 0.5s；没有足够历史或存在间隔时，具体样本和原因写入 preparation_report.json。历史 RA 标定轴发生改变时报错，要求显式外部重采样。

未来查询默认 `[0.5,1,1.5,2,2.5,3]` 秒。显式 timestamped future 优先；未提供时只使用同 scene 的后续 ego pose/稳定 agent ID。只在已观测、有效且时间间隔允许的两端插值，不外推末尾，也不把目标尚未出现前的位置标为已知。未来进入目标放入 prepared `risk_agents` 供标签和评价，不成为当前观测输入。

未来 tracking 覆盖须有明确证据：

- 原始记录可声明 `tracking.coverage="complete_relevant_agents"`，表示标注方保证相关目标和指定未来时域完整。它是数据提供方的标注契约，不能随意添加以通过检查。
- 或提供 `tracking_coverage_valid`，长度等于未来查询点数，逐点标明完整覆盖。
- 否则准备器从同 scene 后续帧的标注可用性推导覆盖；未覆盖的负风险保持未知。

即使部分未来未知，已观测碰撞仍可提供阳性证据；未观测到碰撞不足以证明安全。`areq` 无可行减速度时为截尾未知，附 braking_feasible=false。`prepare` 缓存 overall risk_label 和 agent_risk_labels[id]，保持单位、掩膜和制动可行性标记。

正式 SFT 默认 strict_supervision，要求模式可确定、ego 端点可拟合；LONG 还要求真实地图走廊和每个当前目标的轨迹端点。`build-cohort` 对所有消融分支同时应用这些要求，并输出完整保留/排除清单。部分监督可通过显式 `data.strict_supervision=false` 使用掩膜，但必须作为不同实验协议报告。

## 纯观测预测

使用 `prepare-frames --observations-only` 后 `predict`。不要求 agents、map、ego future；不会从未来帧补标签。`RadarDataset(supervision=False)` 清除标签张量和掩膜，即便原始记录含 GT 也不带入预测监督。模型 forward 只读取观测字段。

结构化推理的局部 agent 序号不是稳定 tracking ID。长度不足的生成标记 invalid；SHORT 的未规划后续保持缺失，不补全为静止轨迹。v2 checkpoint 不兼容 v1 模型结构/物理词表。

## 指令约束评价

自然语言文本没有通用、可靠的自动成功判据。可在 `language.constraints` 提供只供评价的明确条件，坐标统一为当前 ego；它们不会输入模型。

```json
{"instruction":"Stop before the marked region.","constraints":[
  {"type":"speed_limit","max_mps":12.0,"until_s":3.0},
  {"type":"stop_by","time_s":2.0,"max_mps":0.3,"until_s":3.0},
  {"type":"goal_region","at_s":3.0,"min_xy":[10.0,-1.0],"max_xy":[14.0,1.0]}
]}
```

speed_limit 和 stop_by 不写 until_s 时使用记录的完整未来时域；短计划覆盖不足不记成功。heading_at 需要 at_s、yaw_rad、tolerance_rad，且仅在计划明确输出 ego_yaw 时评价车体朝向，不用倒车时的速度方向冒充朝向。没有约束、没有所需时域或没有方向预测时，对应指标为 null 并报告覆盖率。
