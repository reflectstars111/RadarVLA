# Stage 2 / Qwen 数据流代码审计

**2026-10-13 增补：**Grounder 新增 [检测结果接口](../DENSE_DETECTION.md)：JSONL 顶层 `detections`、`num_agents`、`num_vehicles`、`class_counts` 表示置信度/NMS 筛选后的多类别检测。新模型的 `agent_state/agent_future` 与 detections 数量、顺序一致；每个目标的轨迹由 Grounder 直接给出，不经过 Qwen。它们独立于本页的 Qwen `plan`，不能与 plan 按下标混用。Qwen prefix 和 LONG/SHORT grammar 保持本页所述连接。

审计日期：2026-10-09，依据当前工作区源码。本文记录实际实现；没有加载完整 Qwen2.5-3B 权重，也没有启动训练。形状推导以 `configs/l40s_qwen25_3b.json` 为准。CPU 导入验证了词表大小和全部 LONG/SHORT 长度。

## 1. 维度符号与配置边界

| 符号 | 含义 | 当前配置 |
|---|---|---:|
| B | 单进程 microbatch | 启动器示例为 1；函数默认 2 |
| Q | Grounder 输出 radar token 数 | 8 |
| d | Grounder 特征维度 | 64 |
| D | 本地 Qwen 输入 embedding 维度 | 从实际模型权重读取，不硬编码 |
| Ntxt | 当前 batch 中指令原生 tokenizer 序列的最大长度 | 动态，至少分配 1 个槽位 |
| Lp | 送入 Qwen 的 prefix 长度 | 8 + 1 + Ntxt = 9 + Ntxt |
| Lt | batch 物理目标序列最大长度 | LONG 最多 143，SHORT 最多 24 |
| Vp | 物理词表大小 | 398 |
| Na | LONG 输出 agent 数量 | 0–8，生成时由模型决定 |
| Nk | SHORT 输出 critical agent 数量 | 0–2，生成时由模型决定 |

`planner.hidden_dim=64`、`planner.num_layers=2`、`planner.num_heads=4` 是 tiny 后端的网络配置以及共同接口字段；**它们不把 Qwen 变成 64 维、2 层、4 头网络**。HF 后端从 `native.get_input_embeddings().weight.shape[1]` 读取 D，层数、注意力头数与 MLP 宽度来自本地 Qwen 自身配置。当前配置的 `/path/to/Qwen2.5-3B` 仍须替换为真实目录。

公开的 [Qwen2.5-3B 官方配置](https://huggingface.co/Qwen/Qwen2.5-3B/raw/main/config.json) 为 `D=2048, intermediate_size=11008, num_hidden_layers=36, num_attention_heads=16, num_key_value_heads=2`，所以标准模型的 prefix 可标注 `[B,9+Ntxt,2048]`。这些是官方配置参考值；实际代码仍读取用户本地模型。本次未加载完整 3B 权重。

证据：[hf_planner.py](../../hf_planner.py) 24–25、35–49；[pipeline.py](../../pipeline.py) 91–129；[配置](../../configs/l40s_qwen25_3b.json)。

## 2. Stage 1 → Stage 2 的实际接口

Grounder 调用返回：

```text
prediction['radar_tokens']          [B, 8, 64]
prediction['risk']                  [B, 5]
prediction['object_logits']         [B, 8]
prediction['agent_state']           [B, 8, 5]
prediction['agent_future']          [B, 8, 6, 2]
prediction['agent_future_velocity'] [B, 8, 6, 2]
```

Stage 2 planner 的直接输入只有 `radar_tokens`、`risk`、`instruction`；训练时另提供用于 teacher forcing 的 `target_ids`。Grounder 的预测 agent state/future/objectness **没有再编码成 prefix 传给 Qwen**，它们用作 Grounding 辅助监督并单独保存于推理 JSONL。

`ego_velocity`、`ego_acceleration`、yaw rate 经 Grounder 的 ego encoder 进入风险和对象分支。当前 Qwen prefix **没有单独 ego token**；ego 运动也通过 radar 几何编码中的 sensor velocity 等字段影响雷达表征。不能画成“ego 向量经独立 projector 直接拼到 Qwen prefix”。同样，风险 head 与物理 head 是 Grounder 的并行分支，不能画成“预测对象状态直接计算风险 head”。

证据：[model.py](../../model.py) 185–204；[pipeline.py](../../pipeline.py) 223–241、554–570。

## 3. Prefix 组装：8 个 radar + 1 个连续 risk + 原生文本

| 环节 | 输入 | 变换 | 输出 |
|---|---|---|---|
| 雷达适配器 | `[B,8,64]` | Linear(64,D)，含 bias | `[B,8,D]` |
| 风险归一化 | `[B,5]` | 除以 `[1,1,1,80,10]` | `[B,5]` |
| 风险适配器 | `[B,5]` | Linear(5,D) → GELU → Linear(D,D) | `[B,D]` |
| 连续风险 token | `[B,D]` | unsqueeze(axis=1) | `[B,1,D]` |
| 指令编码 | B 条字符串 | 本地 Qwen tokenizer，`add_special_tokens=False` | `[B,Ntxt]` 原生 token IDs |
| 指令 embedding | `[B,Ntxt]` | Qwen 原生冻结 embedding | `[B,Ntxt,D]` |
| Prefix 拼接 | 上述三部分 | 在 token 轴 concat | `[B,9+Ntxt,D]` |

风险五元组顺序是 `Pcol@1s, Pcol@2s, Pcol@3s, dmin, areq`，前三项为概率，后两项单位分别 m、m/s²。5 个连续值共同投影成 **一个** D 维 token，并非 5 个离散 token。risk_source=none 时少掉该 token，Lp=8+Ntxt；oracle 只用于显式消融，要求全部五维标签已知。

指令长度限制 `max_instruction_bytes=96` 是 UTF-8 字节数限制，**不等于固定 96 个 Qwen token**。文本超过预算报错，不截断；空文本仍分配 1 个 padding 槽。没有调用 chat template，也没有自动增加 system prompt 或自然语言问答格式。

`prefix_mask` 为 `[B,Lp]` bool，True 表示 padding。radar/risk 前缀位置均有效，文本右侧 padding 被屏蔽。适配器保持 float32，输出在进入 Qwen 边界时转为配置的 bfloat16；Qwen 主体使用 bfloat16 配置。

证据：[hf_planner.py](../../hf_planner.py) 87–138；[pipeline.py](../../pipeline.py) 212–233。

## 4. 398 个物理 token 与数值范围

| 区间 | 数量 | 用途 |
|---|---:|---|
| 0–13 | 14 | PAD/BOS/EOS、SHORT/LONG、ROAD、AGENTS、AGENT、END_AGENTS、AGENT_FUTURE、EGO_FUTURE、CRITICAL_DYNAMICS、IMMEDIATE_ACTION、TRAJECTORY |
| 14–269 | 256 | UTF-8 byte token 保留区；tiny 后端编码指令使用，HF 指令使用原生 tokenizer |
| 270–333 | 64 | position，范围 `[-80,80]` m；道路宽度也使用此类，但必须为正 |
| 334–397 | 64 | velocity，范围 `[-40,40]` m/s |
| 总计 | **398** | 两个 planner 后端共享的物理词表 |

量化采用 signed-log companding，`alpha=1`，不是等距米制栅格；靠近零点更密。当前 64 为偶数，最近零的 position bin center 约 ±0.07224m，不存在恰好零中心。Ego 起点 `[0,0]` 由坐标定义隐式给出，不通过量化 token 表示。超出坐标或速度范围的监督报错，不静默裁剪。

HF tokenizer 注册 398 个 `<|radar_vla_0000|>` 等 namespaced special token。模型保留原生语言 embedding；另外创建可训练 `physical_embedding[398,D]` 和 `physical_head[D→398]`，初始化自新增的原生行。原生完整词表 head 不用于此任务的输出；任务 logits 的最后一维是 398，不是 Qwen 完整词表大小。

HF 后端仍保留 256 个 byte 槽位以共享完整词表定义，但正式 HF 指令不用该区间编码。生成 grammar 的 `_allowed_next()` 只返回结构标签或 position/velocity IDs，因此生成结果不会出现 byte token，也不会自行生成 PAD/BOS。训练时 398 类仍全部进入 log-softmax，byte 类拿到的概率会作为错误分配受罚。

物理序列使用本地物理 IDs 0–397。`physical_hf_ids` 负责映射到原生词表，用于注册与保存一致性；不是对输出 argmax 后再做自然语言解析。

证据：[tokenizer.py](../../tokenizer.py) 15–89；[hf_planner.py](../../hf_planner.py) 43–73、98–104、150–156。

## 5. 训练时 Qwen 输入、标签和损失

```text
标注对象/道路/ego未来 + 风险教师
  └─ create_supervision
       target_ids          [B,Lt]      int64，未知物理量和尾部补齐用 PAD=0
       continuous_targets  [B,Lt]      float32，保留未量化数值；标签位置为 NaN
       rows metadata                  曲线拟合索引、有效 mask、控制点时间等

teacher forcing:
  input_ids = [BOS, target_ids[:, :-1]] [B,Lt]
  physical_embedding                 [B,Lt,D]
  concat(prefix, physical_embedding) [B,Lp+Lt,D]
  attention_mask                     [B,Lp+Lt]，1 有效 / 0 padding
  position_ids                       [B,Lp+Lt]，有效位置累计计数
  Qwen causal backbone               [B,Lp+Lt,D]
  只取物理部分 hidden                 [B,Lt,D]
  physical_head                      [B,Lt,398]
```

代码的总上下文限制是 `Lp+Lt <= max_length=512`，并非“最多生成 512 个 token”。底层本地模型还必须能容纳这段上下文。目标含 EOS、不含 BOS；训练输出位置与 target_ids 对齐。prefix 本身不作为语言输出标签计算 CE，但通过 causal 注意力接收后续损失的梯度。

CE 忽略 PAD，不为缺失标注填 0。结构标签用硬 CE；物理值用基于**未量化 GT**、物理单位 σ=0.75 的 Gaussian soft targets。全部 398 类参与 log-softmax，预测到错误类型也受罚。规划物理损失则在各自 64-bin 范围内 softmax 求期望，再解码成连续控制点和轨迹。

总损失：

```text
Stage 1: L_grounding
Stage 2: L_grounding + L_token + L_planner_physics
L_planner_physics =
  1.0 × ego trajectory SmoothL1
  1.0 × agent trajectory SmoothL1
  1.0 × road centerline/width SmoothL1
  0.2 × measured Doppler consistency
  0.1 × state/future kinematic consistency
  0.001 × trajectory smoothness
smoothness内部 = acceleration² + 0.1 × jerk² 的时间积分均值
```

Stage 2 加载 Stage 1 Grounder 权重，Grounder 仍可训练，未调用 detach 或整体冻结。仅 Qwen 基础权重冻结。默认风险条件来自 predicted risk，`gt_risk_mix=0`；若显式设非零，训练期按 epoch 衰减 GT 混入概率，验证不混 GT。

证据：[planner.py](../../planner.py) 182–290、346–352、501–596；[tokenizer.py](../../tokenizer.py) 93–127；[pipeline.py](../../pipeline.py) 223–241、326–365、402。

## 6. SHORT/LONG 教师与推理的区别

训练 adaptive 教师：若已知的 1s/2s 碰撞概率 ≥0.5，或 dmin≤0.5m，或 areq≥3m/s²，使用 SHORT；已知安全所需监督完整才用 LONG，否则模式未知，严格 SFT 预检会拒绝。3s 风险参与连续条件和 critical agent 排序，但不是 `_is_hazard` 中碰撞分支的直接触发项。SHORT 教师选择满足 per-agent 危险条件的前 2 个 agent，而非总是硬塞 2 个。

推理时没有 Python 风险阈值切换两套模型。相同 Qwen 的第一个输出 token 从 SHORT/LONG 两个允许项中选择；后续通过 `<AGENT>` / `<END_AGENTS>` 决定 agent 个数。`always_short` / `always_long` 是显式消融配置。

生成是 greedy argmax + grammar mask。所有数值仍由模型预测；grammar 仅限制当前位置允许的类型、标签、最大 agent 数和正道路宽度。当前逐样本生成，每次重新调用前缀与已有 token 序列，`use_cache=False`，没有 KV-cache 加速。

已有k个输出token时，输入物理IDs为 `[1,k+1]`（BOS加已有序列），Qwen输入为 `[1,Lp+k+1,D]`，物理logits为 `[1,k+1,398]`，取最后一个位置的398维向量选择新token。这与训练的批量 `[B,Lt,398]` 相区别。

证据：[planner.py](../../planner.py) 113–179、354–410；[hf_planner.py](../../hf_planner.py) 61、140–156。

## 7. 语法、精确长度与输出数组

当前 `road_controls=4`、`trajectory_controls=4`。

```text
LONG:
<LONG>
<ROAD>                    4 × (x,y) + width                      10 tokens
<AGENTS>
  <AGENT>                 x,y,vr,vx,vy            × Na           6Na tokens
<END_AGENTS>
<AGENT_FUTURE>
  <TRAJECTORY>            4 × (x,y)                × Na           9Na tokens
<EGO_FUTURE>
<TRAJECTORY>              3 × (x,y)                               7 tokens
<EOS>
总长度 Lt = 23 + 15Na = 23,38,53,68,83,98,113,128,143

SHORT:
<SHORT>
<CRITICAL_DYNAMICS>
  <AGENT>                 x,y,vr,vx,vy            × Nk           6Nk tokens
<END_AGENTS>
<IMMEDIATE_ACTION>
<TRAJECTORY>              3 × (x,y)                               7 tokens
<EOS>
总长度 Lt = 12 + 6Nk = 12,18,24
```

这些长度包含生成 EOS，不包含输入 BOS，也不包含 prefix。SHORT 不输出道路或 agent future；LONG 同时输出道路和所有选定 agent 的 future。两个模式都输出 ego 控制点，其中第一点固定 `[0,0]`，只生成后 3 点的 6 个标量。

所有几何均在当前 ego 坐标系，x 向前、y 向左，位置单位 m；vx/vy 是当前 ego 轴上表达的绝对地速，vr 为与雷达观察相关的径向相对速度，单位 m/s。输出 local_id=0..Na−1 是本次生成的局部假设序号，不是数据集永久 track ID。

默认未来时刻为 `[0.5,1.0,1.5,2.0,2.5,3.0]`s，但实际推理 decode 读取样本 `future_times_s`，不强行替换为默认栅格。

| 解析项 | LONG | SHORT |
|---|---|---|
| road centerline_controls | `[4,2]` | None |
| road width_m | 标量 | None |
| road control_parameters | `[0,1/3,2/3,1]`，归一化弧长 | 无 |
| agent_states | Na 个 `{local_id,x,y,vr,vx,vy}` | Nk 个同结构对象 |
| agent_control_points | `[Na,4,2]` | 空列表 |
| agent_trajectories | `[Na,6,2]`（默认时刻） | 空列表 |
| ego_control_points | `[4,2]`（首点严格为 0,0） | `[4,2]`（首点严格为 0,0） |
| control_times_s | `[0,1,2,3]`（默认时刻） | `[0,1/3,2/3,1]`（默认时刻） |
| ego_trajectory | `[6,2]`（默认时刻） | `[2,2]`，0.5s/1.0s |

曲线使用自然三次样条，不是简单的折线插值，也不把控制点当等间隔未定时 waypoint。`decode_spline` 接收 `[...,K,2]` 控制点及明确 knot/query 时刻，输出 `[...,T,2]`。不允许超出控制时间区间外推。道路 decode 返回控制点与归一化弧长参数，不直接生成固定长度稠密道路点列。

证据：[planner.py](../../planner.py) 87–110、412–498；[curves.py](../../curves.py) 14–81；[model.py](../../model.py) 3–5。

## 8. 可训练参数与训练硬件口径

| 模块 | 参数形状 / 数量 | Stage 2 状态 |
|---|---|---|
| Grounder | 由 Grounder 配置决定 | 可训练，Stage 1 初始化 |
| 原生 Qwen | 本地 config/weights 决定 | 冻结 |
| LoRA | r=8，alpha=16，dropout=0.05，bias=none | 可训练 |
| radar_projector | weight `[D,64]` + bias `[D]`，65D | 可训练，float32 |
| risk_projector | `[D,5]` + `[D]` + `[D,D]` + `[D]`，D²+7D | 可训练，float32 |
| physical_embedding | `[398,D]`，398D | 可训练，float32，PAD 行梯度屏蔽 |
| physical_head | weight `[398,D]`，无 bias，398D | 可训练，float32 |

非 LoRA planner 适配器参数合计 `D²+868D`。LoRA target modules 是实际存在的 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` 的交集；各矩阵 LoRA 参数数为 `r×(in_features+out_features)`。没有实际本地 Qwen 配置与权重时，不把具体层数、LoRA 参数总量或显存写成已验证的实际值。

按上述官方标准配置推导，`kv_dim=(D/16)×2=256`，七个投影每层 LoRA 参数数为 `r×(9D+2kv_dim+3I)`，36 层合计 `36×8×(9×2048+2×256+3×11008)=14,966,784`。非 LoRA 适配器为 `2048²+868×2048=5,971,968`，所以 **planner 可训练参数预计为 20,938,752（约 20.94M）**，再加独立 Grounder 参数才是整个 Stage 2 可训练量。该数量是由官方架构和当前 adapter/LoRA 定义推导，不是本地完整 3B 的实载 `numel` 测量。

`lora=false` 仍然冻结 Qwen，只训练适配器，不会切换为全量微调。checkpoint 保存 Grounder、LoRA 与适配器/optimizer 状态；冻结 Qwen 仍须从原本地目录加载并核对文件指纹。

8 卡启动器示例为每卡 B=1、梯度累积 4、world_size=8，完整窗口有效 batch=32；每卡各持完整副本，不是模型切成 8 份。示例 5 epochs、lr=3e-4、AdamW weight_decay=1e-4、grad clip=5、BF16/gradient checkpointing；这些是运行初值，不是测得的最优超参。当前完整 3B 权重与 8 卡显存/吞吐尚未提供实测证据。

证据：[hf_planner.py](../../hf_planner.py) 60–93、158–182；[pipeline.py](../../pipeline.py) 311–317、363–365、389–429、461–466；[DISTRIBUTED.md](../../DISTRIBUTED.md) 55–78。

## 9. 核验与不可混淆项

- 在独立 `radar_vla/.conda/bin/python` 中实际运行 `PhysicalTokenizer()` 和 `_layout()`：398 类词表、LONG 23–143 与 SHORT 12–24 的精确长度均与上文一致。
- 已存在 HF 集成测试使用真正的 `Qwen2ForCausalLM`，但它是 `hidden_size=16, layers=1, intermediate_size=32` 的随机小型模型；它验证接口、LoRA 梯度、checkpoint/BF16/DDP 功能，不代表完整 Qwen2.5-3B 的形状与效果验证。见 [test_radar_vla_hf.py](../../tests/test_radar_vla_hf.py) 7–35。
- 当前图若把 ego 直接画到 Qwen、把物理 head 预测直接画到 risk head 或 Qwen，须按第 2 节改正；可保留 ego→physical/risk head、radar tokens→Qwen、risk vector→risk projector→Qwen、instruction→native tokenizer→Qwen。
- 无完整权重时 D 保留为符号是对真实实现的准确表示。待配置真实目录后，可读取其 config 和 embedding shape 补入具体值；不能把配置中的 64 填到 Qwen 图上。
