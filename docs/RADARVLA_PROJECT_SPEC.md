# RadarVLA 项目设定、实现与实验总览

审计日期：2026-10-18（Asia/Shanghai）。本文是当前证据的共同入口；后续更新须同时注明数据、配置、权重及评估范围。

**审计基线**：Git HEAD `bb91d74889dc654515a4a10950e7ae10008ed7ab`（2026-10-16，`rm ckpt`），工作区不干净；本文描述当前工作区，不等于该 commit 的可复现内容。
审计前已有修改涉及 `.gitignore`、`ENVIRONMENT.md`、adapter、Dataset、model、pipeline、planner、可视化和脚本；Local Risk/Taskwise 源码、配置、测试和迁移文档等尚未跟踪。未覆盖这些修改。
已先检查 README、RUNBOOK、DATA_FORMAT、Pipeline V1/V2、数据流和三份迁移文档；此前没有同名总览，故新增本文，保留原专项文档。

路径约定：`D=/home/luxinlan/data/lxl_data/RadarVLA`；仓库 `analysis/` 是指向 `D/analysis` 的符号链接；`R=D/model_runs/taskwise_local_v1_20261017_003057`。
请求中的 `/home/luxnlan/data/lxl_data` 不存在。Manifest 内还使用可访问的 `/storage/data/metaiot_data/metaiot_guest/lxl_data/...` 绝对路径，搬迁时须保留或显式映射。

| 状态 | 本文含义 |
| --- | --- |
| **已确认** | 当前代码、配置或可读产物有直接证据；不自动表示效果达标 |
| **已实现，未充分验证** | 有执行路径，但缺少充分真实实验或边界验证 |
| **历史设定** | 旧配置、报告或权重的语义，不能替代当前结果 |
| **待核实** | 证据不足、来源冲突，或项目决策尚未确定 |

本轮仅静态审阅、只读统计/哈希、checkpoint 元数据加载及首样本数组抽查；没有训练、推理重评、测试套件重跑或全量传感器逐字节复验。既有验收结果明确按保存产物引用。

## 1. 项目概览

RadarVLA 研究以雷达观测和导航指令进行物理感知、风险估计与风险自适应轨迹规划。Grounding 输出动态数量 Agent 检测、运动及五维 Risk；SFT 学习 SHORT/LONG 结构化输出和 Ego 轨迹，而非自由文本驾驶控制器。
**已确认**：当前学习输入为四帧 Power/unfolded Doppler、有效性和几何、Ego 状态；Planner 另读导航 instruction。RGB/语义 LiDAR 用于采集、仿真雷达生成与核验，当前 Grounder/Planner 不编码它们。

```mermaid
flowchart LR
  A[CARLA / Leaderboard 同步采集] --> B[原始帧、Actor、路线地图、真实未来]
  B --> C[C-Shenron ADC / DRA / RA / Doppler 解折叠]
  B --> E[collection adapter / 版本化 Manifest]
  C --> E
  E --> F[Agent50 / Local Risk / Taskwise 标签和 mask]
  F --> G[Grounding 训练及验证]
  G --> H[SFT：Grounder 与 Qwen LoRA 联合优化]
  F --> H
  H --> I[受限语法生成 / spline 解码 / 离线评估]
```

流程图中的解折叠是离线双侧处理；模型调用只读观测字段不等于整条数据链已满足在线因果性，见第 2、6 节。

## 2. 数据采集与数据约定

| 设定 | 当前值、证据与状态 |
| --- | --- |
| 仿真与 Expert | **已确认**：CARLA、Leaderboard ScenarioManager 唯一 tick owner、BasicAgent；`collection/collector/{run,agent,recorder}.py`。实际 CARLA 二进制精确版本/构建摘要 **待核实**，不能把文档中的 Python 3.7 当成 CARLA 版本 |
| 地图/场景/天气 | **已确认**：配置含 Town01–04，normal、Scenario2/3/4/7_9/10，晴/湿阴/雨/雾/低太阳；`collection/configs/collection_10k.yaml`。原配额 10000、8000/1000/1000 不代表当前训练划分 |
| 原始记录版本 | **已确认**：本轮枚举到的 112 份 `full/jobs/*/episode/metadata.json` 均为 `radarvla_collection_v5`、10 Hz、Agent/Map 范围 50 m；含失败 job，不能当作 112 个有效训练 episode |
| 传感器 | **已确认**：RGB 800×600、FOV 90°；语义 LiDAR 64 线、1,300,000 点/s、10 Hz、垂直 +10°/−30°、118.037109375 m；`run.config()` 与抽查 metadata 一致 |
| 外参 | **已确认**：相机/LiDAR 安装为 CARLA Ego `(1.5,0,2.4)m`、roll/pitch/yaw=0；仿真雷达复用 LiDAR 安装。完整旋转/平移读取 metadata，不仅取 x 偏移；`radar_bev_pose()`、`radar_pose()` |
| Power 来源 | **已确认**：C-Shenron `kradar_dra`，ADC→三维 Hann DRA `[64,256,107]`，`folded_ra.npz.power_peak_ra` 为输入 Power；稀疏散射近似，抽查产品 `max_scatterers=2048`，不是实采硬件雷达 |
| Doppler 来源 | **已确认**：`dense_hann3d_v2` 的 `doppler_unfolded_dense_ra_mps` 与 `reliable_mask`；radar-only 时间纹理匹配，不用 CARLA 速度真值选择歧义阶数。有限候选不等于可靠测量 |
| 输入数组 | **已确认**：每时刻 float32 `[2,256,107]`；四帧堆叠 `[4,2,256,107]`，mask `[4,256,107]`。range 0–118.037109375 m，azimuth −53°…53°，107 格；首样本数组形状、有限性和轴已实读 |
| 历史/未来 | **已确认**：历史时间约 `[-0.3,-0.2,-0.1,0]s`；原始未来 30×0.1 s 实际观测，训练取索引 `[4,9,14,19,24,29]`，即 0.5…3 s 六点；`_build_one()`、`_future()` |
| 状态与同步 | **已确认**：Ego/Actor 世界状态来自同帧 snapshot，camera/LiDAR/CARLA frame ID 对齐，anchor timestamp 与当前帧校验；Actor future 按 episode+actor ID 查询，缺失保存 mask，不凭空外推 |
| 模型坐标 | **已确认**：当前 Ego 原点，x 前、y 左，m/rad/s；世界速度投影到当前 Ego 后仍是绝对地面速度。Doppler 为相对传感器径向速度、远离为正；adapter 反转 RA 方位轴，保留径向标量基底 |
| 运动补偿 | **已确认**：四帧分别携带当前 Ego 系的 pose/LOS/XY、时间和传感器速度；`_radar_sensor_velocity()` 含 `v_ego + omega × r`，不把 Doppler 图像直接旋转为二维速度 |
| 原始 Agent 集合 | **已确认**：排除 Ego，仅 `vehicle.*` / `walker.pedestrian.*`，含静止 actor，无速度阈值；`recorder.record()`。`actor_intersects_fov()` 检查旋转 BBox 的 18 个采样点，任一点落入前向、三维距离≤50 m、水平±53°、俯仰±18°即纳入；不是精确连续体相交求解，也不是中心距离筛选 |
| 检测监督域 | **已确认**：显式 `detection_annotation_domain=radar_bev_sector`，50 m、±53°，围绕当前真实雷达 BEV 安装位姿；只限制背景监督/未匹配 FP 计分，不裁剪完整雷达；边界正例保留。BEV 扇区不等于三维可见性认证 |
| 类别 | **已确认**：当前 `agent_v1` 单类 index 0（padding −1）；移除行人/三种自行车蓝图正例至 `ignored_agents`，摩托车保留。忽略框内或距中心≤2 m 区域，训练另忽略最近 anchor；保留正例优先。`agent50_filter.py`、`annotation_domain.py` |

**已确认的因果性限制**：抽查真实产品 `match_lag_frames=2`；外部 `../Carla/C-Shenron/team_code/e2e_agent_sem_lidar2shenron_package/kradar_dense_unfold.py:180` 的 `unfold_frame()` 读取 `index-lag` 和 `index+lag`，可靠 mask 还使用运行窗口峰值阈值。通常内部帧需未来 0.2 s 回波/位姿。
因此当前数据可支持离线实验；严格零等待在线输入尚未成立。实际缓存与所有生成器版本的完整对应、分块峰值的未来依赖，以及能否接受固定延迟，均须专项核实；本轮未改产品。

### 有效数据版本与 provenance

**已确认**：最新两阶段训练的共同 manifest 是 `D/analysis/taskwise_local_v1/manifest.jsonl`，SHA256：
`083532761bf1dd11c39342b6b1f7e044a1ac533c00275481797db7dee2676cb5`。
本轮实算摘要并与两阶段 `protocol.json.data.manifest_sha256` 一致；协议记录的全部 44 个 `src/*.py` 摘要也与当前源码逐项一致。

| split | Grounding 帧数 | SFT 帧数 | SHORT / LONG / UNKNOWN | episode 数 |
| --- | ---: | ---: | --- | ---: |
| train | 6847 | 6728 | 477 / 6251 / 119 | 45 |
| val | 1472 | 1461 | 56 / 1405 / 11 | 10 |
| test | 1522 | 1508 | 57 / 1451 / 14 | 9 |
| 总计 | 9841 | 9697 | 590 / 9107 / 144 | 64 |

上述帧数、模式和 scene 分组本轮重新统计；scene 三个 split 交集为零，迁移报告以同一 64 episode 划分记录。分层重划逻辑见 `scripts/rebalance_manifest.py`：目标约 70/15/15，episode 级、seed 42，兼顾危险/场景/天气/指令，Scenario4 固定 train；不是原采集地图隔离划分。
`taskwise_local_v1/scenario_coverage.json` 显示 Scenario4 在 val/test 为零、Scenario7_9 在 test 为零；Scenario2 已进入 val/test，旧“仅 train”结论属于 **历史设定**。episode 无交叉不能证明路线、地图或天气完全独立。
版本链：`prepared/full_v3_*` → `analysis/rebalanced_v4` → `rebalanced_50m_domain` → `grounding_agent50_v1` → `local_agent_risk_v1` → `taskwise_local_v1`；旧 8835 帧基础上回收 1006，159 帧因缺四帧雷达产品硬排除，详见各目录 provenance/REPORT。
原始文件在 `full/jobs/<job>/episode/`，雷达在该 job 的 `radar/*/attempt_*/`；最终候选索引 `full/dataset_index.jsonl`，prepared 缓存与新回收缓存由 manifest 引用，未复制完整原始数据。
保存验收 `acceptance.json=PASS`：9841 帧、30795 个传感器路径存在、4648 份新增缓存哈希通过；这是此前验收记录，本轮未重跑。`full/status.json` 的 `RAW_RADAR_COMPLETE/training_ready=UNTESTED` 是原始采集阶段状态，不应覆盖后续验收事实。

## 3. 标签与监督协议

| 标签 | 语义、来源与计算 | 有效性规则 | 消费模块 |
| --- | --- | --- | --- |
| Agent 检测/状态 | 当前合法 Agent；CARLA bbox 中心、长宽、heading；state=`[x,y,vr,vx,vy]`。`vr` 优先取可靠雷达框内关联、功率加权观测；不是直接用 GT 速度伪造测量；`_agent()`、`associate_doppler()` | 当前标注未知与已知空集合不同；分状态轴/径向测量 mask；单类 objectness 代表存在性 | Grounding 检测、运动及物理 loss；Planner 有自己的状态描述教师 |
| Agent Future | 同一当前 Actor 的真实未来六点 XY/yaw，0.5…3 s；未来驶出当前域仍保留 | 逐 Agent、逐时刻 mask；原始 30 点有效性用于检查中间观测缺失 | 检测未来头、Planner LONG 的所选 Agent spline 教师 |
| Ego Future | CARLA 实际执行轨迹，当前 Ego 系六点；不是反事实 rollout | 逐时刻 mask；控制点拟合要求观测覆盖所需区间，不以外推补教师 | SFT metric loss、控制点教师、离线轨迹误差 |
| Road corridor | OpenDRIVE + 官方路线定位的 Ego 单条道路走廊，0–50 m 16 点，宽度来自对应 road/section 驾驶车道外边界；不是雷达可见道路真值 | `valid/break_before/source_ids`；`road_segments()` 按断点/road-section 分段，至少两点；宽度用中位数，完整且跨度≤0.5 m 才有效 | LONG 最多 8 段道路中心线 spline/宽度；未知宽度不抹除有效几何 |
| Risk 五维 | `pcol_1s,pcol_2s,pcol_3s,dmin,areq`；Ego 反事实与其他 Agent 实际未来的框间距，细则如下 | 五维独立 mask，不能把占位 0 当有效负例 | Grounding 全局 Risk 头、Planner 风险 token、Mode 教师 |
| Mode | Local Risk 派生风险事实 SHORT/LONG/UNKNOWN；`task_supervision.local_mode()` | `mode_mask=false` 即 UNKNOWN，整帧不进入 SFT；Grounding 有效任务保留 | SFT 首 token 独立 CE 和生成诊断 |

**已确认**：当前 adapter 已把 `frame.navigation.instruction` 接到 `language.instruction`，也已用分段道路接入训练；采集专项文档中“导航尚未接入、要求 16 点全有效单宽度”的描述是 **历史设定**，不能作为当前 loader 契约。

### Risk 的物理定义及范围

**已确认**：`src/local_risk.py.compute_local_risk()`、`_trajectory()`、`_label()` 与 `src/geometry.py.nominal_rollout()` 定义如下：

- Ego 从当前速度向量和 yaw rate 出发，按 CV/CTRV 反事实延续；制动是恒定减速度，停止后不倒车且停止转动。不使用实际 Ego Future 计算这条 Ego 风险轨迹。
- 对方用当前位置及六个真实未来点插值，yaw unwrap 后插值；0–3 s 按 0.05 s 采样，保留六个结点。原 30 帧 mask 只阻止跨缺失区间插值，不替换成 30 点位置轨迹。
- 距离为定尺寸、旋转二维矩形边界的欧氏距离，重叠为零；危险定义距离 **<0.5 m**。三个 pcol 是截至 1/2/3 s 的累计二值冲突标签，不是仿真频率概率；模型 sigmoid 输出才是概率估计。
- `dmin` 为名义不制动反事实的最小框间距，上限 80 m；`areq` 枚举 0…8 m/s²、步长 0.5，共 17 档，完整检查每档，取最小安全值，不假定制动风险单调。
- pcol：观测到危险即可确认正例；负例要求该窗口集合已知且未来完整。dmin 要完整 3 s，或已观测零距离确定下界；areq 要完整 3 s、当前未危险接近且有安全解，当前接近/无解/证据不足均 mask 掉。
- 已知空 Local 集合：pcol 负例有效，dmin 存 80、areq 存 0，但后二者无效。无效值不得参与回归、阈值判断或 Oracle 替换。
- Local 固定使用当前最终合法 Agent **全集**，不是最近 8 个；包含其驶出域后的有效未来，排除 ignored 行人/自行车，不补造原采集未记录目标；未来新进入者不自动加入。

| 概念 | 当前字段/用途与边界 |
| --- | --- |
| Global Risk | `risk_label/risk_agents/agent_risk_labels` 保留旧世界 actor 及全局覆盖语义，`global_risk` 复制旧标签，schema=`legacy_global_v1`；含域外/忽略目标，诊断留存。目标集合经 `_could_affect_risk()` 的保守 80 m 风险界限筛选，并非无限世界真值 |
| Local Risk | `local_risk/local_agent_risk_labels`，schema=`local_agent_collision_v1`；Dataset 明确映射到当前 `risk_target/risk_mask`。当前配置不回退到 Global |
| 模型 Risk | Grounder 全局五维输出，三个 sigmoid、两个 softplus；SFT 使用预测值，`gt_risk_mix=0`。结构本身不强制概率单调或严格 areq 上限，由监督/正则学习 |
| Oracle Risk | 标签替代风险条件的诊断，不是可部署输入，不替代轨迹 GT；当前 `pipeline.select_risk()` 要五维全部已知的共同 cohort。历史诊断曾只替换有效分量，必须单列，见第 5 节 |

### Mode 与任务 mask

**已确认**：有效 pcol1 或 pcol2 ≥0.5、有效 dmin ≤0.5 m、有效 areq ≥3 m/s²，任一成立为 SHORT；非空集合四项都有效且均未触发才为 LONG；已确认空集合保留 LONG 例外；其余 UNKNOWN。pcol 的 `<0.5m` 与 Mode 的 `≤0.5m` 是代码中的边界差别。
SHORT 事实不依赖 Planner selected；若所选目标中完全没有可描述的关键来源，保留 SHORT/mode_mask，但 `planner_context_consistent=false`、SFT 不可用。当前验收统计此类帧为 0，真实边界验证不足。
`task_masks` 记录 detection、五维 risk、mode、Agent/Ego future 和逐段 road 几何/宽度；Dataset 校验语义及集合。混合 schema/任务协议 batch、跨协议权重加载被拒绝；`strict_supervision=true` 在 Taskwise 下不等于删除所有部分未来帧。
SFT 超词表 scalar 只屏蔽相应 CE、不裁剪 GT；无法完整拟合时屏蔽控制点 CE，真实有效时刻仍供 metric loss；运动一致性和平滑仅取有观测支持区间，全 mask 返回有限可反传零。实现见 `planner.create_supervision()`、`planner_physics_loss()`；完整性验收通过不等于学习效果充分。

## 4. 模型架构与数据流

| 阶段 | 输入 → 核心模块 → 输出 | 监督与可训练参数 | 下游及状态 |
| --- | --- | --- | --- |
| Grounding | 四帧雷达/有效性/校准几何 + 当前 Ego `[vx,vy,ax,ay,yaw_rate]`；Conv stride 4，时空位置编码；每帧池化≤8×8，4 头注意力压缩为 8×64 token；Ego MLP；稠密检测及 Risk 分支 | 全 Grounder 可训练；空间 anchor Hungarian 匹配，objectness focal、框/heading/速度/径向/未来回归、Doppler/运动一致性/平滑、Risk focal BCE+归一化回归+单调正则；`losses.py.DEFAULT_WEIGHTS` | **已确认启用并训练**：Doppler、Ego、temporal、dense detection、Risk；Planner 收 8 雷达 token + 1 Ego token 和预测 Risk |
| SFT/Planner | 上述 token、Risk 投影、英文导航指令 → Qwen2.5-3B HF + LoRA + 物理词表 → SHORT/LONG、道路/Agent/轨迹控制点；自然三次 spline 解码 | 原生 Qwen 参数冻结；LoRA r=8/alpha=16/dropout=.05，注意力/MLP 投影适配器、物理 embedding/head、radar/risk projector 可训练；**Grounder 同时继续训练** | **已确认启用并训练**：`pipeline._loss()`=Grounding + token CE + Planner physics；`AdamW` 纳入两模块所有 requires_grad 参数 |

关键接口/限制：

- Dense detector 当前特征为 64×27、每格 2 槽，共 3456 候选；输出 x/y、长宽/heading、vr/vx/vy、六点 future/velocity。置信度 0.5、旋转 NMS IoU 0.3、中心匹配 2 m；不设输出数上限。单类无冗余分类器，分类 loss=0；检测不是“最多 8 个”。
- **已确认检测损失修复进入新训练**：`losses.py:148` 全部有效候选 focal 求和/匹配正例数（至少 1），已知空帧训练背景，未知标注不训练背景；监督域及 ignored 区域独立门控。新 run 的 losses.py 摘要与当前一致。
- Planner teacher 的 selected 上限 8，按当前最近目标规则保留顺序，SHORT 再按个体风险选择最多 2 个关键目标。LONG 输出道路、所选 Agent 状态/未来和 3 s Ego；SHORT 输出关键动态和 1 s Ego，不生成 LONG 全部字段。
- **已确认**：Planner 不直接消费解码检测框/预测 Agent Future，而从压缩雷达 token、Ego token 和五维 Risk 自行生成描述；GT Agent/道路/未来只用于教师和 loss，不在推理时注入。
- 轨迹/道路各 4 个控制点；Ego t=0 原点隐式固定。物理词表 64 档 signed-log，位置限±80 m、速度限±40 m/s，max_length 512；不能解释成均匀 2.54 m 网格。几何 spline 是代码表示，不是另一个已训练轨迹编码器。
- **已确认当前配置启用**：`planner_ego_context=true`、mode 独立 CE 权重 1、Ego 初速解析导数损失权重 1；Planner Ego/Agent/road 权重各 1，Doppler .2、kinematic .1、smooth .001、jerk .1。物理项是训练软约束，不保证生成必然满足初速或无碰撞。
- Ego 当前五维状态与四帧雷达位姿/传感器速度已输入；独立 Ego 历史序列编码器未见于当前 forward，不能把“历史雷达几何”写成“完整 Ego 历史编码”。HF 生成禁用 KV cache、逐前缀计算；当前新 run 未保存时延测量。

## 5. 当前训练与实验结果

### 实验身份与完成状态

**已确认**：`R/{grounding,sft}/status.json` 均 `complete`，5/5 epoch，`test_evaluated=false`；各 5 条 metrics 和 epoch/last/best 权重存在。本轮已用 `weights_only=True` 读取七个主要 checkpoint 的 epoch/协议元数据，没有实例化训练。
共同配置 `configs/l40s_qwen25_3b_taskwise_local_v1.json`（以 checkpoint 内展开配置为准）；本轮逐键对照一致，SFT 的 Qwen 路径展开为绝对路径，与仓库权重链接解析结果相同。AdamW lr=3e−4、weight_decay=1e−4、seed=42，每 rank batch=1、world_size=2、累积 4、有效 batch=8、bfloat16、workers=2；GT Risk 混入为 0。
Manifest 与当前源码摘要一致，故 50 m、单类、Local Risk、Taskwise/Mode mask 及检测 loss 修复均已进入该轮数据/训练，而非仅停留在迁移报告。精确 `global_step` **未保存**，本文不由 epoch 猜测 optimizer step。
**已确认 SFT 初始化**：protocol 的 `init_grounding_sha256=9a9fe293236b471faa6991fd76423c4d312eb3f6c3fabedf4db6fd3f7febdd4f`，本轮哈希匹配 `R/grounding/best.pt`（epoch 3），不是 epoch 5 的 `best_detection.pt`。两阶段数据协议与源码相同。

### checkpoint 选择与检测

`best.pt` 按最低 validation loss；`best_detection.pt` 优先 F1、再 count MAE；`best_planning.pt` 优先语法有效率、教师要求时间覆盖率，再覆盖点 ADE（固定 val 子集）。三者用途不同，尚无项目统一部署候选。

| `R/` 下权重 | epoch / val loss | val 帧/GT 数 | Precision / Recall / F1 | count MAE | 检出匹配 Agent ADE/FDE (m) |
| --- | --- | --- | --- | ---: | --- |
| `grounding/best.pt` | 3 / 1.7864 | 1472 / 4961 | .8256 / .2709 / .4080 | 2.3743 | 3.5832 / 5.9964 |
| `grounding/best_detection.pt` | 5 / 1.9406 | 1472 / 4961 | .6656 / .3836 / .4867 | 1.7065 | 3.6731 / 6.2851 |
| `sft/best.pt` | 4 / 4.7712 | 1461 / 4928 | .6951 / .3793 / .4907 | 1.8528 | 4.1334 / 6.9966 |
| `sft/best_detection.pt` | 3 / 4.7924 | 1461 / 4928 | .6191 / .4730 / .5363 | 1.6064 | 3.9255 / 6.6706 |
| `sft/best_planning.pt` | 5 / 5.0440 | 1461 / 4928 | .6609 / .3176 / .4290 | 1.9055 | 4.2148 / 7.1250 |

来源：各阶段 `metrics.jsonl.validation.detection`，epoch 与 checkpoint 实读一致。域内未匹配预测计 FP，域外/ignored 未匹配预测不计分；这不是 118 m 全域 AP，匹配轨迹误差只覆盖检出目标。
例如 Grounding 检测候选 TP/FP/FN=1903/956/3058，raw/evaluated predictions=5028/2859，漏检仍显著；旧四类/不同 split/不同监督域指标不能直接与此表作因果改进对比。

### Risk 与 SFT 生成结果

| 权重 | pcol1/2/3 AUROC | pcol1/2/3 Brier | dmin MAE (m) / areq MAE (m/s²) |
| --- | --- | --- | --- |
| Grounding `best.pt` (e3) | .9201 / .9071 / .9045 | .03269 / .04610 / .05564 | 10.0834 / .11583 |
| Grounding `best_detection.pt` (e5) | .6852 / .6935 / .7587 | .02569 / .04170 / .05201 | 10.4404 / .12560 |
| SFT `best_planning.pt` (e5) | .7178 / .7404 / .7992 | .01538 / .02671 / .03693 | 8.3567 / .07411 |

Grounding pcol 有效数 1462/1461/1461、正例 17/40/56；dmin/areq 有效数 1416/1407。SFT 五轮 pcol 均在 1461 帧，正例仍 17/40/56。以上来自 validation，不是 test。
Grounding e5 全零风险基线 Brier 为正例率约 .01163/.02738/.03833，均低于模型；高排序分数或较小 areq MAE不能证明校准/危险样本预测可靠，需分正负例及有效 mask 诊断。

SFT 自由生成仅固定 **32 帧 val**（16 SHORT+16 LONG，IDs 在 `sft/supervision_summary.json` 与每轮日志），不是 1461 帧总体生成评估：

| 候选 | 生成 SHORT/LONG | SHORT recall / precision | 教师要求点覆盖 | 要求点 Ego ADE (m) | 生成有效点 ADE / 3s FDE (m) |
| --- | --- | --- | --- | --- | --- |
| `sft/best.pt` e4 | 7 / 25 | .375 / .8571 | .96875 | .8342 | 1.5229 / 3.7590（25 帧有 3s） |
| `sft/best_planning.pt` e5 | 5 / 27 | .3125 / 1.0 | 1.0 | .8003 | 1.4549 / 3.7229（27 帧有 3s） |
| 同组匀速 CV | 0 / 32 | 不作模式基准 | 1.0 | .6768 | 1.9716 / 4.4207（32 帧有 3s） |

语法有效率均 1；e5 对真实 SHORT 仍漏 11/16。e5 生成点覆盖全六点网格为 .8958，教师要求点为 1.0；二者分母不同。模型要求时间段 ADE 高于 CV；生成点 ADE/FDE 覆盖不同，不能据其低于 CV 就声称整体胜出。
**待核实**：新协议下全 val 自由生成、初速误差、道路/Agent 文本准确性、离线碰撞和真实时延尚无完整保存评估；test 无新结果，闭环安全性未验证。

### 历史对照的适用范围

**历史设定**：`analysis/current_pipeline_audit_20261016/risk_modes_fresh.json` / `risk_oracle_review.json` 对 `analysis/retrained_sft_v2/epoch_003.pt`（摘要 `538260dd…a8cd`）、旧 50 m manifest（`22dfa79b…97c1`）的 73 帧 cohort 比较模式首 token。
其中 56 SHORT/17 LONG，预测 Risk 的 SHORT recall=37/56=.6607，Oracle=55/56=.9821；10 帧有部分 Risk mask，旧脚本只替换已知分量，其余保留预测。此为旧 Global 协议、首 token 诊断，不能称作当前 Local SFT 的完整 Oracle rollout 收益。
`docs/TRAINING_QUALITY_REVIEW_20261016.md` 的 932 帧测试及“真实重训尚待完成”属于旧实验时点；本轮新训练已完成，但不能用新 val 结果覆盖旧 test，也不能把旧 Oracle 结论迁移到新权重。

## 6. 当前状态与问题清单

优先级：P0 影响实时/因果或实验身份主张，P1 影响核心研究结论，P2 为复现与覆盖改进；不是所有条目都阻断离线研究。

| 类别 / 问题 | 当前证据 | 影响 | 下一步验证 | 优先级 |
| --- | --- | --- | --- | --- |
| 已确认并完成：新版监督及两阶段训练 | manifest/源码摘要与 protocol 一致，5/5 complete | 可作为当前离线基线 | 维护数据/配置/权重身份表 | P2 |
| 已实现但需验证：任务 mask 与物理软约束效果 | 当前实现+保存验收；真实 context_inconsistent/partial_ego_future 案例缺失 | 边界正确性及物理收益未知 | 独立边界样本复核，做初速/Mode 消融 | P1 |
| 已发现未解决：Doppler 双侧/窗口峰值未来依赖 | 真实 v2 metadata 与外部 unfold 源码 | 不能直接主张零延迟在线感知 | 对齐缓存生成版本；明确延迟或因果解折叠对照 | P0 |
| 已发现未解决：默认入口仍指旧版本 | local_paths.model_config 为旧 json，CLI `_latest()` 仅搜 prepared | 无参启动不自动选 Local/Taskwise；可能协议拒绝或跑旧实验 | 成员确认发布入口；当前必须显式 manifest/config | P1 |
| 已发现未解决：漏检与 Risk 质量 | e5 Grounding recall .3836，dmin MAE 10.44m，Brier 劣于全零 | 目标召回、概率校准和安全解释受限 | val 距离/运动/危险分层、校准及固定口径对照 | P1 |
| 已发现未解决：模式召回与基线差距 | 32 帧 e5 SHORT recall .3125，要求点 ADE .8003>CV .6768 | 不能证明风险自适应规划已优于简单基线 | 全 val paired 生成、场景分组和初始化对照 | P1 |
| 设计决策待确认：单类忽略行人/自行车与 Local 固定集合 | 当前明确实现，Global 可危险而 Local LONG | 定义不覆盖全部交通风险/未来进入者 | 确认论文/产品任务范围，单列未覆盖风险 | P1 |
| 设计决策待确认：正式候选与分层 split | 多种 best，SFT 从 G best loss 初始化，场景覆盖有缺口 | 选模目标不同，泛化结论有限 | 冻结候选/阈值，审查 route hash 与 episode 来源 | P1 |
| 未来实验：新 Local Oracle / 无 Risk / 初速消融 | 当前只有旧协议 Oracle | 风险条件的因果贡献尚未证实 | 新权重共同有效 cohort，配对样本和种子 | P1 |
| 未来实验：固定方案后的 test/闭环/时延 | 新 test_evaluated=false，生成延迟 null | 无独立 test 或实时安全结论 | 先 val 决策，再固定 test；独立延迟与闭环评测 | P2 |

## 7. 关键文件索引

| 职责 | 最重要入口及用途 |
| --- | --- |
| 采集 | `collection/configs/collection_10k.yaml`：场景/配额；`collection/collector/{run,agent,recorder,episode}.py`：传感器、同步记录、真实未来；`road_map.py`：FOV/地图来源 |
| 雷达产品 | `collection/collector/radar.py` 是外部 C-Shenron wrapper；外部 `carla_kradar_episode.py` / `kradar_dense_unfold.py` 才是当前生成与双侧解折叠实现 |
| 数据处理 | `src/collection_adapter.py`：原始到 manifest；`src/data.py`：校验、Dataset/collate；`src/radar_processing.py`：几何和实测 Doppler 关联；`scripts/rebalance_manifest.py`：episode split |
| 标签/几何 | `src/{annotation_domain,agent50_filter,geometry,local_risk,task_supervision,road_segments}.py`：覆盖域、忽略策略、反事实、Mode/mask 与分段道路 |
| 模型 | `src/{model,detection,hf_planner,planner,tokenizer,curves}.py`：雷达、稠密检测、Qwen 接口、教师/物理损失及 spline |
| loss/训练 | `src/losses.py`：Grounding 目标；`src/pipeline.py`：两阶段训练/协议与 checkpoint；`configs/l40s_qwen25_3b_taskwise_local_v1.json`：当前实验配置 |
| 评估 | `src/{evaluation,metrics,detection_metrics,training_quality}.py`：评估口径与选模；`scripts/diagnose_{detection,risk_modes,sft_trajectory}.py`：专项诊断 |
| 当前数据产物 | [Taskwise 验收](../analysis/taskwise_local_v1/REPORT.md)、[Local Risk 验收](../analysis/local_agent_risk_v1/REPORT.md)、[Agent50 验收](../analysis/grounding_agent50_v1/REPORT.md)：迁移、统计及 provenance |
| 当前训练产物 | `R/{grounding,sft}/{protocol.json,status.json,metrics.jsonl,best*.pt,last.pt}`：训练身份/选择/完成证据；`R/sft/supervision_summary.json`：教师分布与生成验证 IDs |
| 协议补充 | [Taskwise](TASKWISE_LOCAL_PROTOCOL.md)、[Local Risk](LOCAL_RISK_MIGRATION.md)、[Agent50](AGENT50_MIGRATION.md)：细节维护入口；[采集契约](collection/DATA_COLLECTION_SPEC.md) 中旧 loader 限制须按本文注明的现状解读 |

## 8. 共同核实清单

以下只列尚需决定或补证据的问题；manifest/两阶段身份、当前模块开关及实际初始化来源已经核实，不再作为疑问重复列出。

- [ ] **在线时间定义**：依据真实 `match_lag_frames=2`、`unfold_frame(index±lag)` 和窗口峰值，确认接受固定观测延迟还是要求纯历史版本；补齐全部缓存的生成版本/时间依赖证据。
- [ ] **发布入口**：依据旧 `configs/local_paths.json` 与 CLI prepared 自动选择规则，决定是否正式以 Taskwise manifest/config 为团队默认；本次仅记录、不改配置。
- [ ] **任务范围**：依据忽略类别、固定当前 Local 集合及 Global/Local 对照，确认“Agent 风险”是否有意排除行人/自行车、域外危险和未来进入者，统一论文与项目表述。
- [ ] **正式选模目标**：依据 best loss/detection/planning 的不同 epoch，确定后续使用哪份权重，以及是否比较 SFT 从 Grounding best_detection 初始化；现有初始化事实无需再猜测。
- [ ] **泛化定义**：依据 episode 划分、Scenario4/7_9 覆盖缺口，核查 route_sha256、Town/天气和同路线重复的跨 split 关系；决定是否需要独立路线或地图评估。
- [ ] **Risk 可用性标准**：依据低正例数、Brier/CV 基线和 dmin 误差，确定校准、危险子集召回及有效 mask 分层的验收标准；高 AUROC 是否足以支持拟发表结论仍需证据。
- [ ] **SFT 收益**：补全 val 自由生成、同时间范围 CV、初速/碰撞/道路/Agent 描述和分场景结果；32 帧分层诊断不足以确认整体收益。
- [ ] **新协议 Oracle/消融**：使用匹配当前权重的共同五维已知 cohort，区分真实轨迹 GT、反事实标签、预测 Risk 与 Oracle 条件；旧 73 帧首 token 实验不能代替。
- [ ] **复现和最终评估**：保存 CARLA/C-Shenron 精确构建与依赖版本、搬迁路径映射及正式源码版本；在 val 决策冻结后确定 test、时延和闭环评测方案。当前日志未记录精确 global_step，勿补写推算值。
