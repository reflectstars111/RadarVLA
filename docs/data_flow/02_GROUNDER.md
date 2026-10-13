# Stage 1 张量与参数流核验

**2026-10-13 增补：**本页保留原有查询/风险分支的核验。当前 Grounder 将固定对象查询头替换为 [多类别稠密检测头](../DENSE_DETECTION.md)，从池化前 `[B,64,64,27]` 特征预测类别、BEV 框、状态和未来轨迹，再输出可变长度结果。当前总参数 **180,527**，其中检测分支 **54,474**；按实际目标数提供监督，目标未来轨迹不经过大模型。下文 **157,763** 为旧架构总量。“不预测 heading/类别”等旧说明仅适用于原查询头。

核验日期：2026-10-09（Asia/Shanghai）。依据正式配置 `configs/l40s_qwen25_3b.json`、`model.py`、`losses.py`、`pipeline.py` 和 `data.py`。使用 `radar_vla/.conda/bin/python`、PyTorch 2.5.1，`CUDA_VISIBLE_DEVICES=''`、CPU、`torch.no_grad()` 做了完整 Grounder 单次前向与模块参数统计；未加载 Qwen、未训练、未使用 GPU。

## 1. 符号及输入

- `B`：单进程 / 单卡此次前向的样本数，启动脚本默认 1；不能把 8 卡乘梯度累积得到的 32 直接放进每卡形状。
- `T=4`：历史帧数；`R=256`：距离 bins；`A=107`：方位 bins。
- `D=64`：Grounder 隐藏维度；注意力 `num_heads=4`，每头维度 16。
- `Q=8`：压缩雷达 token 的查询数；`M=8`：物体查询数 / 最大 agent 数。两者是独立配置，虽然当前都等于 8。
- `H=6`：未来采样点数。采样时刻来自 manifest 的 `future_times_s`，必须严格递增且大于 0；配置只指定 6 个点，并不固定采样间隔。

| 输入 | 批量形状 | 语义 / 单位 |
|---|---|---|
| `radar` | `[B,4,2,256,107]` | 第 0 通道非负线性 power；第 1 通道 unfolded Doppler，m/s |
| `range_m` | `[B,256]` | 真实距离坐标，m，非负且严格递增 |
| `azimuth_rad` | `[B,107]` | 方位角坐标，rad，严格递增 |
| `time_offsets_s` | `[B,4]` | 相对当前 ego 时刻的观测时间，s，递增且不晚于 0 |
| `radar_doppler_valid` | `[B,4,256,107]` | Doppler 有效标记，bool；不是置信度 |
| `radar_cartesian` | `[B,4,256,107,2]` | 每帧 RA cell 在当前 ego 坐标系中的 `(x,y)` |
| `radar_los` | `[B,4,256,107,2]` | 相应测量的雷达视线方向 |
| `radar_pose` | `[B,4,4,4]` | 每历史帧传感器到当前 ego 的刚体变换 |
| `radar_sensor_velocity` | `[B,4,2]` | 雷达传感器地面速度，当前 ego 坐标系，m/s |
| `ego_velocity` | `[B,2]` | ego 的 `(vx,vy)`，m/s |
| `ego_acceleration` | `[B,2]` | ego 的 `(ax,ay)`，m/s² |
| `ego` | `[B,3]` | 数据兼容字段 `[速度模长,ax,yaw_rate]`；完整 v2 分支只从这里取 yaw rate |
| `future_times_s` | `[B,6]` | 需要预测的未来时间点，s；查询规格而非未来观测 |

单样本磁盘上的 power 与 unfolded Doppler 各是 `[4,256,107]`，Dataset 沿通道轴堆叠为 `[4,2,256,107]`，collate 再添加批次轴。读取转换为 float32；训练启用 bfloat16 autocast，并不意味着磁盘数据、模型参数和所有中间张量一律为 bfloat16。

依据：[data.py](/ssd1/code/yqq/radar_vla/data.py:201)、[输入校验](/ssd1/code/yqq/radar_vla/model.py:87)。可选 geometry 字段在 Grounder 中有 legacy fallback，完整 v2 Dataset 会提供这些字段。

## 2. CNN 与时空压缩

Power 归一化为 `log1p(clamp_min(power,0) × 1.0)`，Doppler 除以 15；无效 Doppler 置零。归一化后依然只有两个 CNN 输入通道。

| 步骤 | 运算 | 输出形状 |
|---|---|---|
| 历史帧并入 batch | reshape | `[4B,2,256,107]` |
| CNN 第 1 层 | Conv2d `2→32`, kernel 5, stride 2, padding 2；GroupNorm 8 组；GELU | `[4B,32,128,54]` |
| CNN 第 2 层 | Conv2d `32→64`, kernel 3, stride 2, padding 1；GroupNorm 8 组；GELU | `[4B,64,64,27]` |
| CNN 第 3 层 | Conv2d `64→64`, kernel 3, stride 1, padding 1；GELU | `[4B,64,64,27]` |
| 自适应平均池化 | 最多保留 `8×8` 空间网格 | `[4B,64,8,8]` |
| 还原帧轴与移动通道轴 | reshape / permute | `[B,4,8,8,64]` |
| 有效性编码 | 原始 bool 有效图池化 → `[B,4,8,8,1]`，Linear `1→64`，加至 CNN 特征 | `[B,4,8,8,64]` |
| 几何编码 | 10 维描述 → Linear `10→64` → GELU → Linear `64→64`，加至特征 | `[B,4,8,8,64]` |
| 展平时空位置 | `4×8×8=256` 个 key / value token | `[B,256,64]` |
| 可学习雷达查询 | 参数 `[8,64]` 广播至每样本 | `[B,8,64]` |
| 时空 cross-attention | `Q=[B,8,64]`, `K=V=[B,256,64]`, 4 头 | `[B,8,64]` |
| 查询残差 + LayerNorm | `LN(query + attention)` | `[B,8,64]` |
| token MLP 与残差 | `64→128→64`，GELU，`LN(token + MLP(token))` | `[B,8,64]` |

这里的 attention 同时跨全部 4 帧、全部 8×8 空间位置，不是对每个空间位置单独做 4 帧 attention。每帧 CNN 参数共享，时间信息由几何描述显式加入；没有在 CNN 输入端将 4 帧拼成 8 通道。

每个池化位置的 10 维几何描述严格为：

```text
[当前 ego 原点距离 / 100,
 当前 ego 方位方向的 y 分量, 当前 ego 方位方向的 x 分量,
 time_offset_s,
 雷达实际 LOS 的 x 分量, 雷达实际 LOS 的 y 分量,
 传感器原点 x / 100, 传感器原点 y / 100,
 传感器 vx / 15, 传感器 vy / 15]
```

先对 cell 的当前 ego 坐标和 LOS 做池化，后计算距离、归一化方向；观测 age 保留秒单位。CNN 的两通道、10 维 geometry、1 维 validity 是不同分支：它们分别映射成 64 维后相加，**不是 12 通道或 13 通道 CNN 输入**。

依据：[CNN 定义](/ssd1/code/yqq/radar_vla/model.py:50)、[几何构造](/ssd1/code/yqq/radar_vla/model.py:122)、[前向](/ssd1/code/yqq/radar_vla/model.py:156)。本节表中形状已通过 B=1、T=4 CPU forward hooks 实测。

## 3. 自车、风险和物体分支

完整 v2 自车状态为 `[vx,vy,ax,ay,yaw_rate]`，逐维除以 `[15,15,5,5,1]`，MLP `5→64→64` 得到 `[B,64]`。速度是带符号二维向量，不是仅传入速度模长。

**风险分支：**一个可学习 risk query `[B,1,64]` 对雷达 token `[B,8,64]` 做 4 头 attention，结果压成 `[B,64]`；与 ego `[B,64]` 拼接得到 `[B,128]`；MLP `128→64→5`。前 3 个 logits 经 sigmoid，后 2 个经 softplus。

**物体分支：**8 个可学习 object query `[B,8,64]` 对雷达 token `[B,8,64]` 做 4 头 attention；`LayerNorm(object_query + attention + ego[:,None])` 得到 `[B,8,64]`。随后分别经过 objectness、state、future residual、future velocity residual 头。

| 字段 | 形状 | 含义 |
|---|---|---|
| `radar_tokens` | `[B,8,64]` | 给 Stage 2 的连续雷达 token |
| `risk_logits` | `[B,3]` | 三个碰撞风险 logits，供 focal BCE |
| `risk` | `[B,5]` | `[Pcol(1s),Pcol(2s),Pcol(3s),dmin,areq]`；前三个无量纲，后两个为 m、m/s² |
| `object_logits` | `[B,8]` | 每个物体槽的存在性 logit；不是 softmax 类别 |
| `agent_state` | `[B,8,5]` | `[x,y,vr,vx,vy]`；位置 m，速度 m/s |
| `agent_future` | `[B,8,6,2]` | 当前 ego 坐标系中的未来 `(x,y)`，m |
| `agent_future_velocity` | `[B,8,6,2]` | 同坐标系中的未来 `(vx,vy)`，m/s |

Grounder 不直接预测 bbox3d、heading、类别、轨迹 ID 或 ego spline。agent state 的 `vr` 是相对传感器的径向速度，`vx/vy` 是地面绝对速度在当前 ego 坐标系的表达；两者不能混为同一种速度。监督时使用 Hungarian matching，物体槽不与固定 agent ID 绑定。

未来位置和速度的计算不是单纯匀速外推：

```text
future_xy[b,m,h] = current_xy[b,m]
                 + current_vxy[b,m] * future_times_s[b,h]
                 + learned_offset[b,m,h]          # [B,8,6,2]

future_vxy[b,m,h] = current_vxy[b,m]
                  + learned_velocity_residual[b,m,h]
```

风险头和 agent 头是雷达 token 的两个并列分支。风险头的输入没有直接连接 `agent_state` / `agent_future`；Stage 2 的连续前缀输入也只有 radar tokens 与 risk，物理 agent 输出用于监督及结果记录，不能将它画成直接进入 Qwen 的 agent token。

依据：[头部定义](/ssd1/code/yqq/radar_vla/model.py:64)、[分支前向](/ssd1/code/yqq/radar_vla/model.py:185)、[Stage 2 调用](/ssd1/code/yqq/radar_vla/pipeline.py:223)。

## 4. Grounder 参数量（CPU 实测）

| 模块 | 参数量 |
|---|---:|
| CNN encoder | 57,248 |
| validity encoder `1→64` | 128 |
| position encoder `10→64→64` | 4,864 |
| radar queries | 512 |
| temporal attention | 16,640 |
| token norm | 128 |
| token MLP | 16,576 |
| output norm | 128 |
| ego encoder | 4,544 |
| risk query | 64 |
| risk attention | 16,640 |
| risk head | 8,581 |
| object queries | 512 |
| object attention | 16,640 |
| object norm | 128 |
| objectness head | 65 |
| state head | 4,485 |
| future position residual head | 4,940 |
| future velocity residual head | 4,940 |
| **总计 / 可训练总计** | **157,763 / 157,763** |

分组总量：雷达 CNN/几何/validity/时空压缩 96,224；ego 4,544；risk 25,285；object/physical 31,710。只统计 Grounder，不包含 Qwen、LoRA 和 Stage 2 adapters。

Stage 1 训练全部 Grounder 参数；Stage 2 先加载 Stage 1 checkpoint，然后仍训练全部 Grounder 参数。训练入口对所有 `requires_grad=True` 参数共同建立 AdamW，没有冻结 Grounder。Qwen 的冻结规则不适用于 Grounder。

## 5. Grounder 监督与 Stage 2 梯度路径

| 标量损失 | 默认系数 | 依据 |
|---|---:|---|
| collision | 1 | 三时域风险 focal BCE，alpha 正例 .25 / 负例 .75，gamma 2 |
| distance | 1 | dmin Smooth L1，回归尺度 10 m |
| brake | 1 | areq Smooth L1，回归尺度 5 m/s² |
| monotonicity | .1 | `relu(P1-P2)` 与 `relu(P2-P3)` 的有效标签均值 |
| objectness | 1 | 物体槽 BCE；已观察无物体是负例，未知标注场景不参与 |
| position | 1 | 匹配物体 x/y Smooth L1，尺度 10 m |
| velocity | 1 | vx/vy Smooth L1，尺度 10 m/s |
| radial | 1 | vr Smooth L1，尺度 10 m/s；存在实测 Doppler 字段时使用有效关联观测 |
| agent_future | 1 | 未来位置 Smooth L1，尺度 10 m |
| future_velocity | 1 | 仅显式提供未来速度标签与 mask 时有该项，否则为 0 |
| doppler | .1 | `(预测 vx/vy - 传感器速度) · Doppler projection` 对比实际有效雷达观测 |
| kinematic | .01 | 相邻位置差与梯形速度积分的残差 |
| smooth | .001 | 预测未来速度的加速度平方 + jerk 平方 |

所有缺失或 censored 标签按显式 mask 忽略。风险标签并不进入 Grounder.forward；GT agent 与 future 不作为模型观测输入。

Stage 1 为上述各项加权和。Stage 2 为 `L_grounding + L_token + L_planner_physics`，没有自动冻结 Stage 1：token 与 planner physics 梯度可经雷达/risk projector 反传至 Grounder，保留的 grounding loss 继续直接约束物理头。SHORT/LONG 的离散选择、量化与 teacher target 构造本身不因此变成全可微操作。

依据：[losses.py](/ssd1/code/yqq/radar_vla/losses.py:10)、[总损失](/ssd1/code/yqq/radar_vla/pipeline.py:223)、[参数纳入优化器](/ssd1/code/yqq/radar_vla/pipeline.py:363)。

## 6. 实测边界

上述精确形状及参数量来自当前代码和正式配置的 CPU 前向。探针用 6 个示例未来时刻 `.5,1,1.5,2,2.5,3` 秒，仅为满足输入校验；不表示真实训练协议已经规定此时距。尚未用真实雷达数据、真实 Qwen 权重或 8 张 L40S 做此次核验，因此不由本审计推导训练显存、吞吐或最终精度。
