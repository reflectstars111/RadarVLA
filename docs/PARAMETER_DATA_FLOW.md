# RadarVLA 全流程：参数、张量尺寸与数据去向

**2026-10-13 增补：**共享 CNN 的池化前特征新增 [多类别检测与轨迹分支](DENSE_DETECTION.md)，默认每帧 3456 个空间候选，筛选后输出可变长度 `detections`、`num_agents`、`num_vehicles`。当前 Grounder 共 **180,527** 参数，其中新增检测头 **54,474**（CPU 实例化计数）。新的检测分支替代固定对象查询头，按实际目标数训练并给每个检测目标预测轨迹，不调用大模型。下文 2026-10-09 的参数表、流程图和物体查询说明保留为旧架构审计，**157,763 不再是当前总参数量**；旧图未画新增检测分支。`B` 仍是 batch size。

核验日期：2026-10-09。依据当前源码与正式配置 [l40s_qwen25_3b.json](../configs/l40s_qwen25_3b.json)。本文描述**已实现的运算路径**，包含导入、监督、两阶段训练、推理和结果保存；不将方案设想写成已有模型连接。

阅读顺序：本页总览 → [输入与全部 batch 字段](data_flow/01_INPUTS_AND_LABELS.md) → [Grounder 逐层尺寸及参数](data_flow/02_GROUNDER.md) → [Qwen、物理词表与输出](data_flow/03_QWEN_AND_OUTPUTS.md)。

## 1. 先看主链路

单帧是 `[C,R,A]=[2,256,107]`；加上历史与 batch 后，真正输入是 **`[B,4,2,256,107]`**。`B` 是每卡一次前向的样本数，8 卡启动器默认为 1。

```text
原始每帧：Power [256,107] + unfolded Doppler [256,107]
  ↓ 同场景4帧，携带真实时间、几何、有效性
radar                         [B,4,2,256,107]
  ↓ 归一化；历史轴并入batch；共享CNN
CNN                           [4B,32,128,54]
                           → [4B,64,64,27]
                           → [4B,64,64,27]
  ↓ AdaptiveAvgPool 8×8；加geometry/validity embedding
时空特征                       [B,4,8,8,64] → [B,256,64]
  ↓ 8个可学习query，4头cross-attention + 残差MLP
radar_tokens                  [B,8,64]
  ├─ risk head + ego          → risk [B,5]
  ├─ physical heads + ego     → object [B,8]
  │                            state [B,8,5]
  │                            future_xy / future_vxy [B,8,6,2]
  └─ radar projector 64→D     → [B,8,D] ──────────┐
risk [B,5] → projector 5→D→D → [B,1,D] ──────────┤
instruction → Qwen tokenizer → [B,Ntxt,D] ────────┤
                                                ↓ concat
Qwen prefix                  [B,9+Ntxt,D]
  ↓ 冻结Qwen主体 + 可训练LoRA；物理序列自回归生成
physical logits              [B,Lout,398]
  ↓ grammar限制 + 数值反量化 + 自然三次样条
LONG：road + agent states/futures + ego trajectory
SHORT：critical agent states + ≤1s ego trajectory
```

`D` 从本地 Qwen embedding 读取。标准 Qwen2.5-3B 为 **2048**，因此雷达投影是 `[B,8,64]→[B,8,2048]`，默认 prefix 为 `[B,9+Ntxt,2048]`。这是[官方模型配置](https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/config.json)的参考值；此次未加载完整 3B 权重，实际运行以指定本地模型为准。

![RadarVLA 张量流总图](figures/radar_vla_data_flow.svg)

图文件：[SVG](figures/radar_vla_data_flow.svg) · [PNG](figures/radar_vla_data_flow.png) · [可编辑 Mermaid 源码](figures/radar_vla_data_flow.mmd)。图中虚线表示训练监督，不是在线观测输入。

## 2. 维度与关键参数

| 符号 / 参数 | 当前值 | 含义与来源 |
|---|---:|---|
| `B` | 启动器默认 1 | 单卡 microbatch；不是全局有效 batch 32 |
| `T` | 4 | 包含最新可用帧的因果历史；不能据此推断覆盖几秒 |
| `C,R,A` | 2,256,107 | Power、unfolded Doppler；距离/方位单元数 |
| `d=hidden_dim` | 64 | Grounder 特征宽度 |
| `max_grid_size` | 8 | 每帧池化为 8×8；四帧共 256 个时空位置 |
| `num_queries=Q` | 8 | 压缩雷达 token 数，属于学习到的查询，不与8个agent一一对应 |
| `max_agents=M` | 8 | 物体查询/标签容量；与 Q 是独立配置 |
| `num_heads` | 4 | Grounder attention 每头16维 |
| `horizon_steps=H` | 6 | 未来采样点数，具体时刻来自样本 |
| `D` | 标准模型2048 | Qwen特征宽度；标准模型36层、16个attention头、2个KV头 |
| `trajectory_controls / road_controls` | 4 / 4 | 轨迹/道路样条控制点数，不是未来采样点数 |
| `short_max_agents` | 2 | SHORT最多输出2个关键agent，来自 PlannerConfig 默认值 |
| `short_horizon_s` | 1.0 | SHORT ego 输出时域 |
| `bins` | 64 | position/velocity各64个signed-log数值bin |
| `coordinate_limit_m / velocity_limit_mps` | 80 / 40 | 可量化范围分别为±80m、±40m/s |
| `max_instruction_bytes` | 96 | UTF-8字节预算，不是96个Qwen token |
| `max_length` | 512 | prefix加物理序列的总上下文预算 |
| `backend / risk_source / reasoning_policy` | hf / predicted / adaptive | 本地Qwen；预测风险；同一生成器选择SHORT/LONG |

`planner.hidden_dim=64, num_layers=2, num_heads=4` 是共享接口/tiny后端字段，**不会覆盖正式Qwen自身的2048维/36层/16头配置**。JSON省略项由 [ModelConfig](../model.py)、[PlannerConfig](../planner.py) 补齐。当前权重路径仍为 `/path/to/Qwen2.5-3B`，启动时用 `QWEN_MODEL_PATH` 指向真实目录。

## 3. 从 frame_t 到可训练数据

| 环节 | 输入 | 输出 / 去向 |
|---|---|---|
| `prepare-frames` | `radar_frame_v2` 原始JSONL与实际传感器文件 | 检查坐标、时间、标定；构造同场景因果历史；写 prepared `manifest.jsonl` 和历史数组 |
| 历史文件 | 单帧Power/Doppler各`[256,107]` | 每窗口各`[4,256,107]`，另存Doppler有效性、pose、sensor velocity、时间偏移 |
| `prepare` | prepared manifest + ego/agent观测及future覆盖 | 缓存场景/逐agent风险标签与mask，输出 labeled manifest |
| `build-cohort` | labeled manifest | 为LONG/SHORT/adaptive冻结共同样本集；排除原因写`.cohort.json`；oracle对照可要求五维风险全已知 |
| `RadarDataset` | 一个窗口 | Power与Doppler堆成`[4,2,256,107]`；构建观测张量和带mask的GT张量 |
| `collate_batch` | B个样本 | `radar[B,4,2,256,107]`；数值默认float32、mask为bool；instruction保留B条字符串 |
| `to_device` / autocast | CPU batch | 张量送对应GPU；正式训练使用BF16 autocast，不把原始数据文件改成BF16 |

在线可见：雷达、历史几何/有效性/时间、当前ego状态、instruction，以及指定的未来查询时间。Camera/LiDAR导入时实际解码与校验，作为离线资产保存；map、agents和future用于教师/评价，**没有Camera或LiDAR神经编码分支进入当前模型**。

所有几何转换到当前ego坐标：x向前、y向左、z向上。距离m、速度m/s、角度rad；Doppler是相对sensor的径向速度，接近为负。历史RA保持原生栅格，通过坐标/LOS编码表达对齐，不是预先warp后的BEV图。

历史不足4帧会排除，不复制补齐；agent不足8个则补零并用mask表示不存在，超过8个报错。缺失标注另有mask，不能等同于已观测的空场景。真实量程、角度范围、帧率及保留样本数只能从数据读取。

## 4. 归一化和并行输入分支

| 数据 | 处理 | 送入哪里 |
|---|---|---|
| 线性Power | `log1p(clamp_min(power,0) × 1.0)` | CNN第0通道 |
| unfolded Doppler | 除以15m/s；无效cell置零 | CNN第1通道；保留正负方向 |
| Doppler validity | 池化为`[B,4,8,8,1]`有效比例，Linear `1→64` | 加到池化CNN特征 |
| 几何/时间 | 每位置10维：距离/100、方向y/x、time、LOS x/y、sensor原点x/y除100、sensor vx/vy除15；MLP `10→64→64` | 加到池化CNN特征 |
| ego | 实际使用`[vx,vy,ax,ay,yaw_rate] / [15,15,5,5,1]`；MLP `5→64→64` | 风险头拼接、物理对象特征相加；不是独立Qwen token |
| 预测risk | `[P1,P2,P3,dmin,areq] / [1,1,1,80,10]` | Stage2 MLP `5→D→D`，生成1个连续token |

Dataset的兼容字段 `ego[B,3]=[速度模长,ax,yaw_rate]` 与实际五维ego编码器输入不同：v2从 `ego_velocity[B,2]`、`ego_acceleration[B,2]`、`ego[:,2]` 拼接真正的五维向量。CNN始终只输入2通道；geometry和validity单独投影后相加，不组成12或13个CNN通道。

## 5. Stage 1：物理与风险输出

Grounder的风险与物体分支**并列**读取 `radar_tokens[B,8,64]`。

| 输出 | 形状 | 含义 |
|---|---|---|
| `radar_tokens` | `[B,8,64]` | 连续雷达表征，供Stage2投影 |
| `risk_logits` | `[B,3]` | 碰撞分类logits |
| `risk` | `[B,5]` | `[Pcol_1s,Pcol_2s,Pcol_3s,dmin_m,areq_mps2]`；前三项sigmoid，后两项softplus |
| `object_logits` | `[B,8]` | 每个查询槽的存在性logit；sigmoid后是独立概率 |
| `agent_state` | `[B,8,5]` | `[x,y,vr,vx,vy]`，没有bbox/heading/track ID预测 |
| `agent_future` | `[B,8,6,2]` | `当前xy + 当前vxy×查询时间 + 学习残差` |
| `agent_future_velocity` | `[B,8,6,2]` | 当前vxy + 学习速度残差 |

风险头不读取预测agent future再显式算碰撞，而是学习预测五维风险。风险GT由**无新增干预的ego CV/CTRV轨迹**与GT agent futures几何比较产生；不是由实际ego teacher轨迹直接定义。前三维GT是累计1/2/3秒窗口内是否冲突的0/1标签，模型才预测概率。默认安全间距0.5m、风险计算步长0.05s、制动搜索0–8m/s²步长0.5；不可行制动标签按mask屏蔽。

物体监督用Hungarian匹配，不要求固定查询槽对应永久agent ID。当前Dataset未返回GT未来速度，所以 `future_velocity` 直接回归项为0；预测未来速度仍接受运动学/平滑约束。精确逐层运算、损失表与各模块参数量见 [Grounder明细](data_flow/02_GROUNDER.md)。

## 6. Stage 2：Qwen输入与训练序列

正常路径只传 `planner(radar_tokens, predicted_risk, instruction, target_ids)`。前三项是条件，最后一项只在训练中提供。

| 步骤 | 张量 |
|---|---|
| 雷达Linear投影 | `[B,8,64] → [B,8,D]` |
| 五维风险投影 | `[B,5] → [B,1,D]` |
| 原生Qwen文本tokenizer + embedding | B条字符串 → `[B,Ntxt]` → `[B,Ntxt,D]` |
| Prefix | `[B,9+Ntxt,D]`；无risk消融为`[B,8+Ntxt,D]` |
| teacher目标 | `target_ids[B,Lout]`；连续GT保留为`continuous_targets[B,Lout]` |
| shifted输入 | `[BOS, target_ids[:,:-1]]` → 可训练物理embedding `[B,Lout,D]` |
| Qwen完整输入 | `[B,9+Ntxt+Lout,D]`，总长度≤512，带attention mask与position IDs |
| 取物理位置hidden | `[B,Lout,D]` |
| 物理输出头 | Linear `D→398`，得到`[B,Lout,398]` |

词表398 = 14个特殊符号 + 256个byte保留槽 + 64个position bin + 64个velocity bin。HF指令使用Qwen原生tokenizer，不走byte槽；grammar生成不允许byte/PAD/BOS。当前任务不用Qwen原生大词表head生成自然语言回答。

训练由GT构建LONG/SHORT教师；推理由同一个生成器预测首个 `<LONG>` 或 `<SHORT>`，之后按grammar生成。默认使用greedy argmax、逐样本生成、`use_cache=False`。这是当前推理实现，应据此测延迟；不能按启用了KV-cache估算吞吐。

上表的 `[B,Lout,398]` 是批量teacher-forcing形状。实际自由生成时逐个处理样本：已有 `k` 个输出token，就输入 `[BOS]+已有token`，物理输入为 `[1,k+1]`，Qwen总输入为 `[1,Lprefix+k+1,D]`，输出 `[1,k+1,398]`；只取末位398个logits来选下一个token，直到EOS或预算耗尽。CLI默认 `max_new_tokens=1024` 仍受总上下文512和grammar长度约束。

## 7. 输出究竟有多少数据

设LONG生成 `Na=0..8` 个agent，SHORT生成 `Nk=0..2` 个critical agent：

| 项目 | LONG | SHORT |
|---|---|---|
| 物理token数，含EOS、不含BOS/prefix | `23+15Na`，23–143 | `12+6Nk`，12–24 |
| road | 控制点`[4,2]` + 宽度标量 | 无 |
| agent states | `[Na,5]`对应x,y,vr,vx,vy + local_id | `[Nk,5]`同字段 + local_id |
| agent轨迹控制点 | `[Na,4,2]` | 无 |
| agent未来采样点 | `[Na,6,2]` | 无 |
| ego控制点 | `[4,2]`，首点固定原点 | `[4,2]`，首点固定原点 |
| ego轨迹采样点 | 默认`[6,2]`，0.5至3秒 | 默认`[2,2]`，0.5和1秒 |

上述默认采样时刻来自 `prepare-frames` 默认 `[0.5,1,1.5,2,2.5,3]s`；真实decode使用manifest时刻。四个控制点经**自然三次样条**解码，不等同于六个预测时刻。SHORT采样数随1秒内查询时刻而变，不能对任意数据固定写2。`local_id` 是本次生成的局部序号，不是永久track ID。

Grounder输出与Qwen输出是**两组预测**：JSONL顶层 `agent_state[8,5] / agent_future[8,6,2]` 来自Grounder；`plan.agent_states / plan.agent_trajectories / plan.ego_trajectory` 来自Qwen物理token解码，两者不保证逐槽相同。当前 `prediction_rows` 不序列化Grounder的 `agent_future_velocity`，尽管forward与loss中存在它。

`predict` 每个样本写一行：sample/scene ID、risk及conditioning_risk、8个object probabilities、Grounder state/future、plan、生成token数和均摊耗时。只有grounding checkpoint时 `plan=null`。计划输出是坐标轨迹，不是已执行的方向盘/油门/刹车控制指令。

## 8. 两阶段训练：什么更新、损失怎么来

| 参数组 | Stage 1 | Stage 2 | 数量 |
|---|---|---|---:|
| Grounder | 全部训练 | 从Stage1载入后继续全部训练 | **157,763，CPU实测** |
| Qwen基础模型 | 不加载 | 冻结 | 由本地权重决定 |
| LoRA | 无 | 训练；r8、alpha16、dropout0.05 | 标准Qwen配置推导14,966,784 |
| radar/risk projector + physical embedding/head | 无 | 训练，float32参数 | `D²+868D`；D2048时5,971,968 |
| 全部可训练 | 157,763 | Grounder + LoRA + adapters | 标准配置预计**21,096,515** |

LoRA覆盖实际存在的 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。预测token损失能经连续prefix反传到Grounder；保留的grounding loss继续监督物理头。Qwen LoRA/adapter数字是按官方结构推导，非完整3B实载测量。

```text
Stage1 loss = L_grounding
Stage2 loss = L_grounding + L_token + L_planner_physics

L_grounding默认系数：
  collision/distance/brake/objectness/position/velocity/radial/agent_future = 1
  future_velocity = 1（当前无直接GT，项值0）
  monotonicity = 0.1；doppler = 0.1；kinematic = 0.01；smooth = 0.001

L_token：结构硬CE + 物理数值Gaussian soft CE；忽略缺失/PAD
L_planner_physics：ego轨迹1 + agent轨迹1 + road1
                  + Doppler0.2 + kinematic0.1 + smooth0.001
```

Grounder回归尺度是位置/距离10m、速度10m/s、制动5m/s²；不改变输出单位。Stage2 smooth内部为加速度平方加0.1×jerk平方的时间均值。不同损失中的同名项属于不同预测分支，不能合并视作一个头。完整定义见两个模型明细及 [losses.py](../losses.py)、[planner.py](../planner.py)。

8×L40S启动器当前参数：每卡batch1 × 梯度累积4 × 8卡 = **完整累积窗口有效batch32**；尾窗按实际样本数归一化。两个阶段分别默认5epochs、lr3e-4、AdamW weight_decay1e-4、梯度裁剪5、seed42、workers2、BF16；Stage2启用gradient checkpointing。每卡持有模型副本，DDP不把Qwen切成8份。此为运行初始配置，尚无本次真实8卡显存/吞吐测量。

## 9. 命令、产物和证据边界

数据准备、两阶段启动及预测的完整命令沿用 [README](../README.md)、[8卡训练说明](../DISTRIBUTED.md)。关键调用关系为：

```text
frames.jsonl → prepare-frames → windows/manifest.jsonl
             → prepare → labeled.jsonl
             → build-cohort → cohort.jsonl + cohort.cohort.json
             → train --stage grounding → grounding/best.pt
             → train --stage sft --init-grounding grounding/best.pt
             → sft/best.pt → evaluate / predict
```

| 产物 | 内容 |
|---|---|
| `protocol.json` | 展开的实际配置、训练预算、world size、数据/源码/权重指纹 |
| `metrics.jsonl` | 每epoch训练/验证损失、耗时、各rank峰值allocated显存 |
| `last.pt / best.pt` | Grounder、planner可训练权重、optimizer/RNG、历史；best按验证总loss最低选 |
| `status.json` | complete/paused/failed、epoch、best loss等；不等同逐step实时监控 |
| evaluate JSON | 风险、自由生成、物理规划、指令遵循等离线指标；训练期不看test |
| predict JSONL | 无标注也可运行的结构化结果；冻结Qwen仍需从原本地目录加载 |

本次核验包含：正式Grounder配置CPU前向及全部模块参数统计；物理词表、LONG/SHORT所有允许人数对应长度与grammar枚举；源码与配置交叉核对；本地Mermaid语法解析、SVG/PNG渲染与图片检查。它不代表真实Radar数据效果、完整Qwen2.5-3B加载、8卡训练或闭环驾驶已验证。

## 10. 对照你给出的架构图

整体的“两通道历史 → grounding → radar/risk前缀 → Qwen+LoRA → LONG/SHORT”与代码一致。精确画图时需要落实以下连接：

1. 4帧保持 `[B,T,C,R,A]`，每帧共享CNN；attention跨全部256个时空位置。
2. geometry/validity在CNN池化后映射到64维相加，CNN仍为2输入通道。
3. ego经五维MLP进入risk/physical分支；没有单独ego→Qwen prefix投影。
4. physical head与risk head并列；没有“预测agent state/future→risk head”的直接连接。
5. Qwen直接接收radar tokens、一个risk token和instruction；physical adapters指物理词表输入/输出层，不是Grounder agent预测到Qwen的投影。
6. Stage1 checkpoint初始化Stage2 Grounder，Stage2继续联合训练；最终ego样条来自Qwen，不是Stage1输出。

这些差异已反映在随附流程图中；本次仅整理文档与图，没有改变模型行为或医学实验。
