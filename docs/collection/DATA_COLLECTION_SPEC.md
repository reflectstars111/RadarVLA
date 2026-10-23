# RadarVLA 采集与 K-Radar 数据契约（`radarvla_collection_v5`）

## 2026-10-22：新采集的完整 BBox / moving v2 约定

今后新采集及 prepare 使用 `moving_vehicle_01_fullbbox50_background_v2`。
沿用当前世界水平速度 ≥0.1 m/s 的运动车辆标准；当前帧车辆世界 BBox 全部顶点的
XY 投影必须严格位于雷达中心 50 m 圆内。接触或跨越圆周的车辆从标签集合移除，
保留整帧并作为背景，不设置 ignore 区。±53° 继续使用原 BBox 相交规则。
未来停车或驶出范围不裁剪真实 future。完整世界快照、observed_agents、RGB、
原始语义 LiDAR 和雷达信号继续保存。Risk、planner 和任务监督按新集合同步重算。
历史 moving v1 仍可读取；下面的旧版说明仅用于历史数据解释。
本补充优先于下文旧口径。雨雾批次入口、配额及目录见 [雨雾采集说明](RAIN_FOG_COLLECTION.md)。


## 2026-10-19：运动车辆标签补充约定

新 episode 用 `metadata.agent_selection.policy=moving_vehicle_01_background_v1`
标记标签口径。`agents.json.agents` 仅包含 BBox 与前向 50 m、±53° 视野相交，
且当前帧对地水平速度 `sqrt(vx²+vy²) >= 0.1 m/s` 的车辆（排除自行车，摩托车沿用车辆规则）。
不按距离或数量截断检测、Local Risk 集合；planner 保留现有最近 8 个和 SHORT 最多 2 个规则。
静止车辆、行人、自行车在检测监督范围内作为背景，检出计 FP，不再设忽略区，也不参与新版 Risk。

`agents.json.observed_agents` 保留全部视野内原始 actor，`frame.json.actors` 和 anchors
继续保存完整世界快照与真实 future；雷达信号与原始传感器不裁剪。是否运动只由当前帧世界速度决定，
不使用自车相对速度、Doppler 或未来信息；当前运动目标之后停车仍保留其未来轨迹。
旧 episode 没有此 metadata 时按原口径读取；旧原始数据不覆盖，新版训练标签独立生成。
训练数据位于 `<data_root>/prepared/taskwise_moving_v1/`，审计报告位于
`<data_root>/analysis/taskwise_moving_v1/`。本补充优先于下文历史 agent 集合说明。

## 目的与边界

每个 episode 保存 VLA 所需的同步 ego、动态 agent、道路级 map、官方路线、真实未来轨迹和原始传感器。场景事件只是采集诊断，不是样本筛选条件。Expert 保持 `BasicAgent.run_step()` 与 Leaderboard 原始稠密路线及 RoadOption；ScenarioManager 仍是唯一 tick owner。CARLA、ScenarioRunner、C-Shenron、Risk 公式和 VLA 模型均不修改。旧 v2/v3/v4 episode 保持旧格式；本轮不迁移或覆盖历史数据。

## 文件与坐标

```text
episode/
  metadata.json             # episode_id、帧率、传感器位姿、来源及采集状态
  route.json                # Leaderboard 原始稠密路线、逐点 RoadOption 和 road/section/lane/s
  static_map.json           # OpenDRIVE hash、车道采样、同方向车行道外边界、拓扑
  events.jsonl              # 独立于训练筛选的事件历史
  anchors.jsonl             # 每帧 ego 和当前 actor 的真实未来 30 帧及 mask
  frames/000000/
    frame.json              # 同步 CARLA frame、timestamp、ego、全部 actor、导航来源、文件引用
    agents.json             # bbox 与前向 50 m 雷达视野相交的 vehicle/pedestrian
    map.json                # Ego 单条道路走廊：16 点中心、逐点宽度与有效性 mask
    front_rgb.png
    semantic_lidar.npy      # N×6: x,y,z,cos_incidence,ObjIdx,semantic_tag
```

`frame.json` 的 sensor frame ID 必须与 CARLA snapshot 完全一致。Actor 使用真实 CARLA `actor_id`、类型、世界位姿、世界速度/角速度、bbox 和控制量。跨 episode 的唯一键是 `(metadata.episode_id, actor_id)`；离线校验检查帧内重复、跨帧类型稳定和 future ID 引用。`ObjIdx` 不是 runtime actor ID，不能直接当作 actor 的跨帧标识。`agents.json` 仅保存当前 bbox 与雷达前向 ±53°、距离 ≤50 m 视野相交的 actor；`frame.json` 仍保存全部 actor，供 future 追踪。`anchors.jsonl` 对每个当前 ID 查询 t+1…t+30 的真实 CARLA 帧；缺帧或 actor 消失为 `null/false`，不插值。所有常规驾驶帧同样有 anchor。`metadata.json` 记录 50 m 的 agent/map 标签范围，原始雷达量程仍约 118 m。

`static_map.json` 是 episode 级原始几何档案：从 CARLA/OpenDRIVE 的 Driving waypoint、lane width 和显式 `get_topology()` 构建，保留精细边界和拓扑。v5 逐帧 `map.json` 使用 `ego_road_corridor_50m_v1`：先以 Ego waypoint 与官方稠密路线的同 road/section/lane 段定位；该匹配失败时，仅在当前路线进度附近、Ego 到路线小于 2 m、路线位置到对应 OpenDRIVE 车道中心小于 2 m 时使用路线内候选。官方路线进度优先决定重合路口的分支；有拓扑连接的候选记作 `route_fallback`，无连接但满足路线及几何门限的候选记作 `nearest_route_segment`，不进行全局最近道路搜索。沿选中路线在 0–50 m 采样 16 点。每点的 `centerline_ego_m` 和 `width_m` 来自同一 road/section 所有驾驶车道的最外侧边界，`source_ids` 标明 road/section/lane 来源；`valid` 为几何及雷达水平 50 m、±53°扇形 mask。附加的 `matching.ego` 和 16 条 `matching.points` 记录 `source`（`exact_match`、`route_fallback`、`nearest_route_segment`；未处理点为 `null`）、`distance_to_route`、`distance_to_candidate`（米，无法计算时为 `null`）及 `valid`；逐点 `valid` 与主 mask 完全一致。无来源或扇形外的位置为 `null/false`；`break_before` 阻止预览把宽度变化处画成虚构连接。`topology_links` 只记录实际存在的拓扑边。这里的线是 **OpenDRIVE 地图先验**，不代表雷达必然能看到；v4 的多 road/LiDAR 支持标签仍按旧格式读取。

官方路线仍原样保存在 `route.json`；逐帧 `frame.json.navigation` 从当前官方稠密路线接下来的 RoadOption 产生英文 instruction，和道路标签分离，不引入人工场景文本。当前新 collector 的 `frame.json.navigation.instruction` **尚未接入**现有训练记录的 `language.instruction`。现有 loader 对显式道路监督还要求 16 点 mask 全真和单个宽度；v5 的逐点宽度及部分有效点不能直接进入这条训练通路，本轮不改 loader。

语义 LiDAR 的采集量程设置为 118.037109375 m，与 C-Shenron `kradar_dra` 的模拟最大量程一致；校验拒绝 LiDAR 量程小于雷达量程的 episode。

## C-Shenron 雷达与解折叠

离线运行 C-Shenron 的 `e2e_agent_sem_lidar2shenron_package.carla_kradar_episode`，使用其 `kradar_dra` 模型。256 个 range bin、64 个 Doppler bin 由 C-Shenron 雷达参数计算，107 个方位角为 −53°…53°；无需 K-Radar 数据集文件。若额外提供 `--kradar-resources`，只对照校验保存的轴，不参与生成。每帧用 CARLA 一致的三维位姿变换，将语义 LiDAR 点按类别和唯一世界空间 bbox 关联动态 actor，计算相对雷达径向速度；无法唯一关联的动态点剔除，静态点按零世界速度计算。K-Radar 专用 ADC→DRA 路径对距离、Doppler、方位施加相干增益归一化的三维 Hann 窗；通用 C-Shenron FFT 不改变。生成以下文件：

C-Shenron 的逐散射点 ADC 计算成本随点数增长；离线生成默认最多使用 2048 个散射点。采集层先按真实雷达量程及 ±53° 方位、±18° 俯仰裁剪，再按 0.5 m×1° RA 格保留每格首个返回，在有点的格中确定性均匀取样，并为 vehicle/pedestrian 返回预留至多四分之一预算。`radar_meta.json` 逐帧记录原始可用点数、FOV 内点数、实际散射点数和规则。原始语义 LiDAR 不下采样；生成的雷达图是稀疏散射近似，不能称作原始全点云的精确雷达真值。可通过 `--max-scatterers` 调节计算预算。

从 v4/v5 episode 生成时，`radar_meta.json` 的 `moving_agent_ids_in_fov` 读取 `agents.json`，因此该诊断列表表示**前向 50 m 的动态标签 agent**；雷达散射点仍由完整 `frame.json.actors` 和原始语义 LiDAR 生成，雷达信号及 256×107 轴没有缩到 50 m。

```text
radar_output/
  radar_axes.npz
  radar_meta.json
  frames/000000/
    power_dra.npy           # float32 [64,256,107]
    folded_ra.npz           # power_peak_ra、doppler_at_peak_ra_mps、peak_doppler_bin
    unfolded_ra.npz         # radar-only dense Doppler、ambiguity_order、source_code、reliable_mask、单/双侧分量、match_margin
```

稠密解折叠不使用 CARLA 真值选择 `k`。每格枚举 `v_fold + k·period`，利用相邻帧时间和 ego 雷达位姿预测回波位置，按 3×3 RA 功率块的去均值纹理与中心幅度比较。按双侧匹配、单侧匹配、邻域投票、上一帧直接匹配继承、`k=0` 的顺序分配来源代码 1～5。无论来源，每格都有有限候选速度；这不表示每格都成功解折叠。`reliable_mask` 是双侧可靠与单侧可靠的并集，两个分量各自保存：双侧要求前后直接匹配一致、裕量 >0.05；只有一侧时间邻帧时，单侧唯一匹配要求更严格的裕量 >0.2。两者都要求功率不低于全运行峰值 −40 dB，均不是速度真值。历史 `oracle_hann3d_v1` 和 `dense_hann3d_v1` 产品仍可由校验器读取，新 `dense_hann3d_v2` 产品不保存 Oracle 字段。

C-Shenron 的 `Carla/AAA_tools/kradar_carla_dense_viewer.ipynb` 提供 Jupyter 帧滑块，底层 `kradar_carla_viewer.py` 也支持静态 PNG 导出。四图依次为相对全运行峰值 −60～0 dB 的 RA 功率、固定 ±35 m/s 的稠密候选 A、灰色表示不可靠的直接匹配候选，以及五类来源。标题显示真实 CARLA frame ID、可靠格数/比例和来源计数。查看时逐帧加载，不修改原始数组。

`collection/notebooks/joint_radar_map_trajectories.ipynb` 是采集验收用联合预览：滑块按 CARLA frame ID 同步显示原有量程的 RA power 和稠密 unfolded 候选、v5 的 Ego 单条道路中心线，以及当前位于 50 m 内的 Ego/agent 真实未来 30 帧。设置 `EPISODE` 为含完整 `anchors.jsonl` 的原始 episode、`RADAR_OUTPUT` 为对应的 dense 产品目录；不能只传缺少未来帧的雷达短切片。旧 v2/v3 地图在内存中重投影为旧多路格式，v4 直接按旧多路格式读取；缺失 future 按 mask 断线，未来驶出 50 m 的有效轨迹仍显示。这些轨迹是验收用 CARLA 真值，不是模型输入。

## 运行与验收

当前 v5 单次实采、非零起点雷达窗口、报告门禁及停机命令见 [RAW_COLLECTION_RUNBOOK.md](RAW_COLLECTION_RUNBOOK.md)。道路走廊投影已增加官方路线约束的匹配回退；2026-10-14 端到端实测证据见 [v5 验收报告](reports/v5_smoke_20261014.md)。

先在 CARLA 3.7 环境做采集输入预检，路线 XML、route ID、场景 JSON 必须显式给出。真实运行写入全新 episode，不覆盖历史数据。随后在 C-Shenron 环境预检轴文件、生成雷达、校验并出图：

```bash
/home/luxinlan/.conda/envs/lxl_carla/bin/python RadarVLA/collection/collector/run.py \
  --routes /absolute/routes.xml --route-id ID --scenarios /absolute/scenarios.json \
  --output /absolute/new_episode --carla-python /home/luxinlan/.conda/envs/lxl_carla/bin/python \
  --check-inputs

/home/luxinlan/.conda/envs/lxl_carla/bin/python RadarVLA/collection/collector/episode.py \
  /absolute/new_episode --require-complete

export PYTHONPATH="$PWD/Carla/C-Shenron/team_code"

/home/luxinlan/.conda/envs/lxl_cshenron/bin/python -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  /absolute/new_episode --output /absolute/new_radar --check-inputs

/home/luxinlan/.conda/envs/lxl_cshenron/bin/python -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  /absolute/new_episode --output /absolute/new_radar --limit 50

/home/luxinlan/.conda/envs/lxl_cshenron/bin/python -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  /absolute/new_episode --output /absolute/new_radar --validate

/home/luxinlan/.conda/envs/lxl_cshenron/bin/python Carla/AAA_tools/kradar_carla_viewer.py \
  /absolute/new_episode /absolute/new_radar --output /absolute/new_radar_examples
```

运行 C-Shenron 模块时将 `Carla/C-Shenron/team_code` 加入 `PYTHONPATH`。Notebook 打开上述 `.ipynb` 并设置 `RADAR_OUTPUT`。去掉采集输入预检命令的 `--check-inputs` 才启动真实 CARLA。短片段验收要求同步、RoadOption、地图来源、agent ID、future mask、64×256×107 与 256×107 尺寸、折回一致、五类来源计数、可靠覆盖率及动态图像人工检查。不以合成测试冒充真实 PASS。
