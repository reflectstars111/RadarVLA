# RadarVLA

依据 [数据记录结构](../docs/data_record.md) 与 [RadarVLA 方案](../docs/RadarVLA_risk_adaptive_plan.md) 实现的独立研究代码库。当前版本 **0.2.0 / v2**，覆盖完整帧导入、物理与风险标签、两阶段训练、本地 Qwen 结构化规划、离线评价、无标注预测及配对消融实验。模块与执行证据见 [v2 交付记录](../docs/RADARVLA_PIPELINE_V2.md)。

单帧输入 **`[C,R,A]=[2,256,107]`**，默认 4 帧历史；训练目标环境 **8×L40S，每卡 44GB**，语言模型 **本地 Qwen2.5-3B**。源码、环境和产物与医学分割项目独立。

```text
完整 frame_t：Radar / Camera / LiDAR / ego / agents / map / instruction
  → 实际文件解码、标定校验、时间同步、场景隔离、当前 ego 坐标转换
  → 原生 RA 历史 + cell 位置/视线 + 传感器位姿与速度
  → 无新增干预 ego rollout × 未来 tracking → 风险标签与覆盖掩膜
  → Stage 1：联合 Power/Doppler backbone + 时间注意力 + physical/KRS heads
  → Stage 2：[Radar tokens; continuous risk token; instruction] → 同一个 Qwen
  → LONG：道路样条/宽度 → 可变数量 agents → agent 样条 → ego 样条
    SHORT：风险关键目标状态 → 短时域 ego 样条
  → 确定性样条解码 → 结构化预测 / 物理指标 / 指令约束指标
```

Camera/LiDAR 有实际读取、标定与归档接口，用于离线标注和教师资产。按照方案的 Radar 主线，它们不进入在线模型；地图与未来 GT 是监督与评价信息。在线模型读取 Radar、ego 和 instruction，不读取 agents、地图、未来轨迹或 GT risk；oracle 是显式标记的实验分支。

## 环境、安装与端到端检查

独立环境已创建在 `.conda-radar-vla`，安装与迁移见 [ENVIRONMENT.md](ENVIRONMENT.md)。也可安装为独立 Python 包：

```bash
.conda-radar-vla/bin/python -m pip install -e ./radar_vla --no-deps
.conda-radar-vla/bin/radar-vla --help
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .conda-radar-vla/bin/python -m radar_vla smoke \
  --output /ssd1/data/RadarVLA/runs/my_v2_check
```

`smoke` 生成可真实解码的完整帧格式合成样本，使用原生 256×107、4 帧历史完成：导入→风险缓存→共同样本集→grounding→SFT→评价→移除标注后的预测。仅该检查显式选择随机小模型 `backend=tiny`；正式配置默认为 HF，缺少本地权重会报错，不会退回小模型。合成数据验证代码链路，不代表真实驾驶效果。

## 完整 frame_t 契约

原始入口为 JSONL，每行 `schema_version="radar_frame_v2"`。可生成完整可运行格式样例：

```bash
.conda-radar-vla/bin/python -m radar_vla synthetic-frames --output /data/radar/format_example
```

路径相对于原始 JSONL 所在目录，或为绝对路径。字段规范见 [DATA_FORMAT.md](DATA_FORMAT.md)。

| 分支 | 实际支持与要求 |
|---|---|
| 帧 | 唯一 sample_id、scene_id、train/val/test、秒级时间戳、world/ego 坐标系；同 scene 不跨 split |
| camera | front_rgb、可选 left/right 等名称；实际 RGB 解码、3×3 内参、外参、时间戳 |
| lidar | NPY 点云 `[N,D≥3]`，前 3 维 xyz；转换到当前 ego，保留强度等附加列 |
| radar | 实际 Power、folded/unfolded Doppler、有效性掩膜、标定距离/角度；读取 raw_points/raw_cube |
| ego | 4×4 pose、完整有符号 vx/vy、ax/ay、yaw_rate、车体尺寸；未来轨迹只作教师 |
| agents | 稳定 id、bbox3d 中心/尺寸/yaw、velocity、heading、带时间戳的未来轨迹及 validity |
| map | lane centerlines、关联的左右 boundaries、traffic elements、可选明确 route；生成真实走廊监督 |
| language | 原始 instruction；可选机器可检查的指令约束，仅用于评价 |

距离 m、角度 rad、速度 m/s。`ego.pose` 为 ego→world，`T_ego_sensor` 为 sensor→ego。转换后以当前 ego 为原点，x 前、y 左、z 上。速度是地面绝对速度；Doppler 是相对传感器速度的 LOS 投影，接近为负。道路规划使用平面动力学：ego 姿态保持竖直轴，ego/agent 竖直速度及 ego 竖直加速度需为零；相机、LiDAR、雷达外参仍支持三维旋转。

历史雷达保留原生 RA 栅格，位置、LOS、原点与传感器运动转换到当前 ego 后进入编码器。异步传感器必须提供测量时刻的 ego pose；异步 Radar 还必须提供当时的 ego 速度与 yaw rate。异步测量缺少对应时刻目标监督时，Doppler 监督屏蔽。

只有 folded Doppler 时，必须提供因果 `doppler_prior` 及无模糊速度范围才能解模糊；无法确定的 cell 标记无效。未知 Doppler 与实测 0m/s 有不同有效性编码。raw cube 实现读入和校验；没有 FMCW 波形、天线与 FFT 标定时，不伪造 cube→RA 转换。

`agents=[]` 表示当前有标注且为空；缺少 agents 表示未知。未来阴性风险还要求完整 tracking 覆盖，不能从当前空帧推断安全。未来进入场景的目标参与风险与行动评价，未出现前的状态保持未知。

## 导入、缓存与共同样本集

```bash
.conda-radar-vla/bin/python -m radar_vla prepare-frames \
  --input-jsonl /data/radar/frames.jsonl --output-dir /data/radar/windows --history-frames 4
.conda-radar-vla/bin/python -m radar_vla prepare \
  --manifest /data/radar/windows/manifest.jsonl --output /data/radar/labeled.jsonl
.conda-radar-vla/bin/python -m radar_vla build-cohort \
  --manifest /data/radar/labeled.jsonl --output /data/radar/cohort.jsonl \
  --config radar_vla/configs/l40s_qwen25_3b.json --require-oracle
.conda-radar-vla/bin/python -m radar_vla validate --manifest /data/radar/cohort.jsonl
```

导入按时间和场景构建因果历史，记录历史不足或断流的排除原因。未来轨迹来自显式标注或同场景稳定 ID；不跨 split，不按距离猜身份。`prepare` 缓存整体与逐目标风险，避免每轮重做几何搜索。

`build-cohort` 生成 LONG/SHORT/adaptive 都有监督的共同样本集，并将所有保留 ID、排除 ID 与原因写入 `.cohort.json`。末尾未来不足、地图走廊缺失等不通过填零伪造。Q3 加 `--require-oracle` 保证三条分支的 5 个风险标签都已知；不可行制动的截尾标签会被排除。结果仅代表该共同子集，必须报告排除比例和原因。单独 grounding 可用部分监督的 labeled manifest；配对实验各分支用同一冻结 manifest。

## 训练与结构化输出

正式两阶段、恢复与手动监测命令见 [DISTRIBUTED.md](DISTRIBUTED.md)。配置入口 [configs/l40s_qwen25_3b.json](configs/l40s_qwen25_3b.json)。Stage 1 不加载 Qwen，Stage 2 要求 `--init-grounding` 和本地模型路径。训练验证选优，测试单独执行。DDP 支持 BF16、LoRA、梯度累积、gradient checkpointing 和逐 rank 随机状态恢复；不自动启动长期训练或监控服务。

风险标签使用 CV/CTRV 无新增干预自车轨迹和未来目标框，与实际 ego future 教师分开。Pcol 为 1/2/3 秒累计事件；dmin 为框间最小间距；areq 为完整离散减速网格中的最小可行值。无可行制动力时保留不可行标记并屏蔽回归。几何时间网格默认 0.05 秒，仍是离散近似。

损失包含风险 focal/Huber/单调性、Hungarian 对象状态和轨迹、真实雷达 Doppler 一致性、全时域非均匀时间运动学，以及加速度/jerk 平滑。框内实测 Doppler 与相同功率权重的 LOS 平均投影匹配，考虑传感器偏置和旋转杆臂速度。缺实测回波不冒充有效 Doppler 监督。

道路和未来轨迹采用自然三次样条控制点（默认 4 个），确定性解码为带物理时间的轨迹；ego 起点严格固定为原点。SHORT 默认 1 秒，教师按逐目标反事实风险选关键目标。可变对象数由 `<AGENT>/<END_AGENTS>` 表示；推理对象编号仅为该次输出的局部编号，不声称是持久 tracking ID。

Gaussian soft-target 以未量化真值为中心。控制点拟合保留真实端点，缺端点时屏蔽监督；稀疏点使用明确自然插值，不外推假标签。正式 SFT 默认严格预检。超出坐标范围、对象容量、指令字节预算或序列长度会报错，应调整配置，不静默裁剪。

同一个 Qwen 接收连续 Radar/KRS prefix 与原生 tokenizer 编码的完整指令，生成物理词表 token。基础模型冻结，LoRA、输入投影及物理输入/输出适配器可训练；`lora=false` 是仅训练适配器。v2 模型结构和词表已更新，v1 检查点不兼容，不能直接续训。

## 评价、纯观测推理和 Q1–Q5

```bash
.conda-radar-vla/bin/python -m radar_vla evaluate \
  --checkpoint /data/runs/sft/best.pt --manifest /data/radar/cohort.jsonl \
  --output /data/runs/sft/test_metrics.json --device cuda
.conda-radar-vla/bin/python -m radar_vla prepare-frames \
  --input-jsonl /data/observations/frames.jsonl --output-dir /data/observations/windows --observations-only
.conda-radar-vla/bin/python -m radar_vla predict \
  --checkpoint /data/runs/sft/best.pt --manifest /data/observations/windows/manifest.jsonl \
  --output /data/predictions.jsonl --device cuda
```

纯观测不要求 agents/map/future。指标包括风险 AUROC/AP/Brier/ECE、状态和未来误差、实测 Doppler 误差、ego ADE/完整时域 FDE、样条稠密框碰撞、安全间距、局部恒速 TTC、制动时刻、jerk、明确路线的进度/完成比例、输出长度及推理耗时。指标附有效样本数或覆盖率；SHORT 未规划的后续不能计作成功。指令遵循使用 `language.constraints` 明确条件；无可核查标签时为 null。

```bash
.conda-radar-vla/bin/python -m radar_vla plan-experiments \
  --config /data/configs/qwen.json --manifest /data/radar/cohort.jsonl \
  --output /data/runs/paired --plan-file /data/runs/paired_plan.json --seeds 42 43 44
# 上一步只写计划；以下命令才会训练。
.conda-radar-vla/bin/python -m radar_vla run-experiments --plan-path /data/runs/paired_plan.json
.conda-radar-vla/bin/python -m radar_vla doppler-sweep \
  --checkpoint /data/runs/sft/best.pt --manifest /data/radar/cohort.jsonl \
  --sample-id example_id --range-interval 15 20 --azimuth-interval -0.1 0.1 \
  --velocities -2 -5 -8 --output /data/runs/doppler_sweep.json
```

矩阵含 Q1/Q2 Power/Doppler×单帧/多帧及无显式 ego 运动条件，Q3 无 Risk Token/predicted/oracle，Q5 always-long/always-short/adaptive。相同种子、样本、预算和 Stage 1 初始化配对，计划与数据均有内容指纹。无 ego 分支保留标定与位姿补偿；无 Risk Token 分支保留相同风险辅助监督，只去除语言条件 token。Q4 改指定 ROI 的观测 Doppler，Power、位置和指令固定；这是输入敏感性试验，不是重模拟后的驾驶场景。

## 验证范围

本代码覆盖方案必需的离线监督主线。方案第 13 节明确可选的闭环 RFT，以及 CARLA 联机闭环，没有冒充已完成。真实 Radar 数据、完整 Qwen2.5-3B 权重与 8 卡服务器尚未提供，实际效果、全尺寸模型显存和吞吐需要在那里验证。当前使用合成格式数据、真实小型 Qwen 和双进程 CPU DDP 验证实现；不将功能测试写成真实训练结论。
