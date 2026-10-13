# RadarVLA 输入与监督数据流核查

**2026-10-13 增补：**检测分支新增 `agent_class[B,M]`、`agent_class_mask[B,M]`、`agent_size[B,M,2]`、`agent_heading[B,M]` 和 `agent_box_mask[B,M]` 监督。原始 `agents[].category` 保留到 prepared manifest，缺失类别不作为车辆。正式配置要求类别标注；纯观测模式清除这些监督。详见 [检测数据与输出契约](../DENSE_DETECTION.md)。检测路径的 M 是每个 batch 的最大实际目标数，不再固定为 8；空场景允许零个监督槽。下文固定 M=8 的说明仅适用于旧查询架构，其他输入和监督仍适用。

核查日期：2026-10-09。依据当前实现与 `configs/l40s_qwen25_3b.json`，只读审查，未修改运行代码、未启动训练。以下是代码实际数据契约；张量带 `B` 时表示 batch，`T=4` 历史帧，`R=256` 距离单元，`A=107` 方位单元，`M=8` 最大当前目标数，`H=6` 未来采样点。

## 1. 正式配置与单条数据完整路径

```text
原始 frame_t：radar_frame_v2
  power[R,A]、unfolded_doppler[R,A]
  或 folded_doppler[R,A] + 因果 doppler_prior[R,A] → 解模糊
  + range_m[R] / azimuth_rad[A] / 传感器时间和外参
  + ego / agents / map / language / 可选 camera、lidar
        ↓ load_frame：单位与形状验证、变换到当前 ego、生成未来标签
每场景按时间排序，取同场景因果历史 T=4
        ↓ prepare_frames：保持原生 RA 栅格，保存历史→当前ego几何
manifest.jsonl（radar_vla_v2）
  power[T,R,A] / unfolded_doppler[T,R,A] / doppler_valid[T,R,A]
  + poses[T,4,4] / sensor_velocities[T,2] / time_offsets[T]
        ↓ RadarDataset.__getitem__：按 channel 轴 stack
radar[T,C,R,A] = [4,2,256,107]，float32
        ↓ collate_batch：直接 stack，增加 batch 维
radar[B,T,C,R,A] = [B,4,2,256,107]
        ↓ 模型内部物理归一化（Dataset 不归一化）
power → log1p(power×1.0)，unfolded_doppler → doppler/15.0
```

- `C=2` 的顺序固定为 **线性非负功率、解模糊径向速度**。`folded_doppler` 可归档，但不作为第三个 CNN 通道；原始 cube/points 也不替代 RA 输入。[输入读取](/ssd1/code/yqq/radar_vla/data.py:203)、[channel 堆叠](/ssd1/code/yqq/radar_vla/data.py:289)、[归一化](/ssd1/code/yqq/radar_vla/model.py:81)
- `[2,256,107]` 是**单帧**形状；真正送入时序模型的是 `[B,4,2,256,107]`。正式配置提供 `radar_shape` 和 `history_frames`，pipeline 检查样本一致性。[正式配置](/ssd1/code/yqq/radar_vla/configs/l40s_qwen25_3b.json:2)、[检查](/ssd1/code/yqq/radar_vla/pipeline.py:152)
- 原始 `.npy` 功率必须是线性值，不能直接把 dB 图当作功率；数组必须为有限实数，读取后转 `float32`。Doppler 单位 m/s，**远离传感器为正**。[原始读取](/ssd1/code/yqq/radar_vla/records.py:262)、[径向速度约定](/ssd1/code/yqq/radar_vla/radar_processing.py:1)
- 没有硬编码雷达量程、角度范围或帧率。`range_m`、`azimuth_rad`、`time_offsets_s` 来自数据，不能仅凭 `256×107` 推出分辨率。

## 2. 时间、坐标与缺失数据

所有变换使用右手系、SI 单位；`T_ego_sensor` 将 sensor 坐标映射到该时刻 ego，`ego.pose` 将 ego 映射到 world。原始 `coordinate_frame` 可为 world 或 ego，prepared 记录为 `current_ego`。使用列向量变换含义，NumPy 实现为右乘旋转矩阵转置。[坐标实现](/ssd1/code/yqq/radar_vla/coordinates.py:28)、[原始约定](/ssd1/code/yqq/radar_vla/records.py:1)

历史雷达**不将 Doppler 图重采样到当前栅格**；每帧仍保留原 RA 像素，同时提供每个像素在当前 ego 下的位置和视线方向：

```text
本帧 radar 点 = [r cos(a), r sin(a), 0]
radar_cartesian[t,r,a,:] = 历史radar→当前ego 的变换后的 xy
radar_los[t,r,a,:] = 同一旋转变换后的 [cos(a),sin(a)]
```

这是几何对齐，不是已经运动补偿的 BEV 图；保留径向速度原本的传感器测量含义。[RA 几何](/ssd1/code/yqq/radar_vla/radar_processing.py:45)、[历史变换](/ssd1/code/yqq/radar_vla/records.py:547)

- 默认准备参数：历史间隔不得超过 `0.5 s`，传感器与帧时间偏差最多 `0.05 s`，未来轨迹可插值缺口最多 `1.1 s`。这些是准备函数默认值，不是正式 JSON 中重复列出的常量。[prepare_frames](/ssd1/code/yqq/radar_vla/records.py:491)
- 历史不足四帧或时间间隔超限的窗口会被排除并记录原因；**不复制历史帧、不做历史 padding**。相邻历史的 range/azimuth 轴不同则报错，要求显式重采样。[窗口规则](/ssd1/code/yqq/radar_vla/records.py:528)
- `time_offsets_s` 严格递增且不为正。v2 最后一帧允许略早于当前时刻；异步雷达必须提供测量时刻 ego pose/state。异步末帧不允许用当前 GT 框监督 Doppler 关联。[异步校准](/ssd1/code/yqq/radar_vla/records.py:55)、[关联屏蔽](/ssd1/code/yqq/radar_vla/data.py:232)
- Doppler 解模糊要求显式因果先验，可接受 `causal_tracker`、`multi_prf`、`sensor_firmware`；不是根据 future/GT 速度构造输入。公式 `v_unfold=v_fold+2*v_max*round((v_prior-v_fold)/(2*v_max))`，超过先验残差阈值的像素 mask=false 且输入 Doppler 置零。[解模糊](/ssd1/code/yqq/radar_vla/radar_processing.py:20)、[原始输入规则](/ssd1/code/yqq/radar_vla/records.py:276)
- 默认未来采样时刻为 `[0.5,1.0,1.5,2.0,2.5,3.0] s`，即 `H=6`。正式模型配置写的是 `horizon_steps=6`；真实时刻以 manifest 的 `future_times_s` 为准。未来标签允许精确采样或受限的线性插值，缺失值为零且 mask=false，不外推。[未来插值](/ssd1/code/yqq/radar_vla/records.py:74)
- 原始 ego/agent 速度要求平面运动，ego→world 姿态要求保持垂直轴，服务二维 BEV 规划。传感器 `T_ego_sensor` 使用一般4×4刚体变换验证，雷达、camera、LiDAR 外参允许三维旋转和平移；雷达坐标变换后再取 xy。不能将平面规划约束误写成雷达外参只能平面旋转。[平面约束](/ssd1/code/yqq/radar_vla/records.py:26)、[传感器外参](/ssd1/code/yqq/radar_vla/records.py:59)、[雷达几何投影](/ssd1/code/yqq/radar_vla/radar_processing.py:45)

## 3. Batch 观测张量

下表是 `RadarDataset` 与 `collate_batch` 实际返回形状，不表示每个张量都进入神经网络的主干；模型消费路径须继续对照 forward。

| 键 | batch 形状 | dtype | 含义 / 单位 |
|---|---|---|---|
| `radar` | `[B,4,2,256,107]` | float32 | 原始线性功率、解模糊 Doppler m/s |
| `range_m` | `[B,256]` | float32 | 雷达原生距离坐标，m |
| `azimuth_rad` | `[B,107]` | float32 | 雷达原生方位角，rad |
| `time_offsets_s` | `[B,4]` | float32 | 每次雷达测量相对当前帧时间，s，≤0 |
| `radar_pose` | `[B,4,4,4]` | float32 | 历史帧数4 × 齐次变换4×4；每帧sensor→当前ego |
| `radar_cartesian` | `[B,4,256,107,2]` | float32 | 每个 RA 单元在当前ego的 xy，m |
| `radar_los` | `[B,4,256,107,2]` | float32 | 每个 RA 单元在当前ego的视线方向 |
| `radar_sensor_velocity` | `[B,4,2]` | float32 | 传感器原点在当前ego坐标表达的速度，m/s；包含角速度×杠杆臂 |
| `radar_doppler_valid` | `[B,4,256,107]` | bool | Doppler 像素可用性 |
| `ego` | `[B,3]` | float32 | `[norm(v_xy), a_x, yaw_rate]`，单位分别 m/s、m/s²、rad/s |
| `ego_velocity` | `[B,2]` | float32 | `[v_x,v_y]`，m/s |
| `ego_acceleration` | `[B,2]` | float32 | `[a_x,a_y]`，m/s² |
| `future_times_s` | `[B,6]` | float32 | 未来查询相对时刻，s |

关键区别：Dataset 的兼容字段 `ego` 是**3个数**，但完整 v2 数据流进入 Grounder 的实际条件向量是 **`ego_state[B,5]=[vx,vy,ax,ay,yaw_rate]`**。`model.py` 将 `ego_velocity[B,2]`、`ego_acceleration[B,2]` 与 `ego[:,2:3]` 拼接，再逐维除以 `[15,15,5,5,1]` 后进入 ego encoder。只有缺少完整向量的 legacy 输入，才用三维兼容字段补出横向零值。ego 全局 pose 保留在 metadata，并参与前期坐标对齐，未直接作为4×4矩阵拼入条件。[Dataset字段](/ssd1/code/yqq/radar_vla/data.py:293)、[实际模型条件](/ssd1/code/yqq/radar_vla/model.py:183)、[sensor velocity](/ssd1/code/yqq/radar_vla/coordinates.py:49)

## 4. Batch 监督张量与 padding

当前 agent 先按距离当前ego原点从近到远排序，ID 作为稳定次排序键。目标数 `<8` 时在末尾补零、相关 mask=false；`>8` 时直接报错，不能静默丢掉目标。监督槽位排序与 `critical_agent_order` 的风险排序是两套索引。[capacity](/ssd1/code/yqq/radar_vla/data.py:193)、[排序](/ssd1/code/yqq/radar_vla/data.py:217)

| 键 | batch 形状 | dtype | 含义 |
|---|---|---|---|
| `agent_state` | `[B,8,5]` | float32 | `[x,y,v_r,v_x,v_y]`，位置m/速度m/s；`v_r`为标注速度减sensor速度后在关联视线上的投影 |
| `agent_state_mask` | `[B,8,5]` | bool | 逐状态维有效性；末帧异步时 `v_r` 维无效 |
| `agent_mask` | `[B,8]` | bool | 当前真实目标是否存在；padded槽false |
| `agent_supervision_mask` | `[B]` | bool | 本样本当前agent标注覆盖是否已知；已观测空场景与未知不同 |
| `agent_future` | `[B,8,6,2]` | float32 | 各目标未来xy，当前ego坐标，m |
| `agent_future_mask` | `[B,8,6]` | bool | 每个目标、每个未来时刻有效性 |
| `ego_future` | `[B,6,2]` | float32 | 实际ego未来xy，当前ego坐标，m |
| `ego_future_mask` | `[B,6]` | bool | ego未来标签有效性 |
| `agent_radial_observed` | `[B,8]` | float32 | 最新雷达帧内目标框覆盖单元的功率加权**实测**Doppler，m/s |
| `agent_radial_mask` | `[B,8]` | bool | 雷达关联是否存在且同步；不是模型预测的objectness |
| `agent_doppler_confidence` | `[B,8]` | float32 | `1/(1+框内测量方差)`，仅离散度诊断，不是校准概率 |
| `agent_los` | `[B,8,2]` | float32 | sensor原点指向agent中心的单位方向 |
| `agent_doppler_projection` | `[B,8,2]` | float32 | 框内实际匹配单元LOS的功率加权均值，可非单位长度 |
| `agent_sensor_velocity` | `[B,8,2]` | float32 | 对每个agent重复最新sensor速度 |
| `risk_target` | `[B,5]` | float32 | 场景级 `[Pcol_1s,Pcol_2s,Pcol_3s,dmin,areq]` |
| `risk_mask` | `[B,5]` | bool | 场景级各风险标签是否已知 |
| `agent_risk` | `[B,8,5]` | float32 | 每个当前agent单独构造的五维风险标签 |
| `agent_risk_mask` | `[B,8,5]` | bool | 逐agent风险有效性 |
| `critical_agent_order` | `[B,8]` | int64 | 先按首次冲突窗口，再按dmin、ID排序的GT槽位索引，padded在后 |
| `road_centerline` | `[B,16,2]` | float32 | 从地图选择的前向centerline采样点，m；默认最多60m |
| `road_centerline_mask` | `[B,16]` | bool | 道路采样点是否已知 |
| `road_width` | `[B,1]` | float32 | 对应左右边界的平均走廊宽度，m |
| `road_mask` | `[B,1]` | bool | 是否有完整有效左右边界宽度监督 |

返回张量集中定义于 [data.py](/ssd1/code/yqq/radar_vla/data.py:288)。道路16点是**标签采样点数**，与正式配置 `road_controls=4` 的4个输出样条控制点不同。[道路标签](/ssd1/code/yqq/radar_vla/records.py:166)

实测Doppler与标注状态中的径向速度不同：前者从RA测量关联产生，后者由GT速度与sensor速度投影产生。关联只用于训练监督；重叠目标框里的栅格按归一化中心距离唯一分配，避免重复标注。[关联实现](/ssd1/code/yqq/radar_vla/radar_processing.py:57)、[状态投影](/ssd1/code/yqq/radar_vla/data.py:241)

## 5. 五维风险标签的真实含义

风险标签不直接沿GT ego future评估，而是用**当前速度与yaw rate的CV/CTRV反事实ego轨迹**，与GT agent futures比较。默认在0–3秒以 `0.05 s` 密集采样，并加入标注节点及1/2/3秒节点；距离是两个有向BEV矩形边界的欧氏距离。[标签实现](/ssd1/code/yqq/radar_vla/geometry.py:81)

| 维度 | 监督值 | 默认规则 |
|---|---|---|
| 0–2 | `Pcol_1s, Pcol_2s, Pcol_3s`，二值0/1标签 | 对应累计时间窗是否发生接触或间距 `<0.5m` 的冲突；模型预测值才称概率。已观测到正例即可有效，负例需要完整追踪覆盖 |
| 3 | `dmin`，m | 0–3s采样点的最小矩形间距，上截断80m；coverage完整或已观测0距离时有效 |
| 4 | `areq`，m/s² | 从0到8按0.5步长搜索第一个安全纵向制动强度；未覆盖完整未来或无可行制动时mask=false，存储零不表示“无需刹车” |

`risk_agents` 可包含从未来track已知会进入场景的agent，它们用于场景级风险监督，但不注入当前agent状态监督及在线输入。该标签是采样时间的反事实目标，不能声称连续时间避碰保证。[future entrants](/ssd1/code/yqq/radar_vla/records.py:444)、[风险mask](/ssd1/code/yqq/radar_vla/geometry.py:158)

## 6. Camera / LiDAR / map / language 到哪里

- Camera：原始导入时读取RGB为 `[H_img,W_img,3]` uint8，保留光学sensor像素坐标及3×3内参、4×4外参；没有固定图像分辨率。prepared manifest仅保留sensor资产与标定metadata，**没有在线camera tensor进入模型**。[camera读取](/ssd1/code/yqq/radar_vla/records.py:299)
- LiDAR：原始导入时读取 `[N,D>=3]`，前三列xyz转ego坐标，其他属性保留；临时payload为float64。没有固定N；prepared Dataset没有在线点云张量。[LiDAR读取](/ssd1/code/yqq/radar_vla/records.py:317)
- Map：车道线、边界、交通元素变换到当前ego并保留metadata；道路监督目前通过centerline与左右关联边界产生。交通元素没有独立模型输入张量。[map转换](/ssd1/code/yqq/radar_vla/records.py:111)
- Language：`instruction` 是字符串，batch为B条字符串；到planner再编码。Dataset阶段没有 `[B,L]` token张量。[字符串及metadata返回](/ssd1/code/yqq/radar_vla/data.py:314)
- `metadata` 为B条字典（map、sensors、ego_pose），`agent_ids` 为 `[B][8]` 的字符串列表，`sample_id`、`scene_id` 为B条字符串。它们不会被default_collate强行张量化。[batch collate](/ssd1/code/yqq/radar_vla/data.py:320)
- `supervision=False` 时agent状态/future/风险GT清零且mask屏蔽，即使源manifest还带GT也不会形成物理监督输入；实际运行前向用到的risk应来自grounder预测，而非缓存标签。[观测模式](/ssd1/code/yqq/radar_vla/data.py:278)

## 7. Cohort 与需要保留的边界

`build_cohort` 用同一manifest分别检查 adaptive、always_long、always_short 的teacher监督可用性，缺GT current agent coverage、ego轨迹端点、所需road corridor或LONG agent future端点的样本被显式排除并写报告。它按**标注可用性**筛选，不按预测质量筛选；`require_oracle=True` 额外要求五维risk全部已知。[cohort构造](/ssd1/code/yqq/radar_vla/cohort.py:24)

本文没有实际用户雷达文件，无法从代码确定真实帧率、雷达量程、azimuth覆盖范围、相机分辨率、点云数量、真实标注覆盖率及cohort保留率。这些值必须从真实manifest和preparation/cohort报告读取，不能由模型默认值代替。
