# RadarVLA 初版 pipeline

独立于本仓库的医学分割任务，依据 [data_record.md](../docs/data_record.md) 和 [RadarVLA 方案](../docs/RadarVLA_risk_adaptive_plan.md) 实现。代码、训练配置、Conda 环境与运行输出独立。真实雷达数据和 Qwen2.5-3B 权重尚未提供，因此交付的是可运行的研究链路与服务器启动配置。

单帧雷达输入为 **`[C,R,A]=[2,256,107]`**；初始历史窗口为可配置的 4 帧。没有沿用医学实验的 512×512 设置。正式目标环境为 **8×L40S，每卡约 44GB**。

2026-10-05 验证：独立环境专项 **51 项通过**，全仓库 **229 项通过**，原生输入端到端 smoke 完成。执行证据与待验证项见[交付记录](../docs/RADARVLA_PIPELINE_V1.md)。

```text
JSONL manifest + 历史 Power/Doppler NPY
  → 字段/物理坐标/scene split 校验
  → CV/CTRV 无新增干预 rollout + 未来 agent GT
  → Pcol(1/2/3s)、dmin、areq 标签与缺失掩膜
  → 共用 RA CNN + polar/time PE + temporal query tokens
  → Stage 1：agent 状态/未来 + KRS grounding
  → Stage 2：[Radar tokens; predicted KRS token; instruction]
  → 一个 Qwen2.5-3B + LoRA → SHORT/LONG + 物理离散 token
  → 结构化 agent/ego waypoints JSON
```

## 已实现的范围

| 模块 | 入口 | 行为 |
|---|---|---|
| 数据契约 | `data.py` | 严格校验、scene 隔离、历史张量、标注掩膜、标签预计算 |
| 风险标签 | `geometry.py` | CV/CTRV、不倒车的制动 rollout、旋转矩形距离、完整减速网格搜索 |
| Radar grounding | `model.py` / `losses.py` | 双通道共用 backbone、极坐标与时间编码、跨帧 query 压缩、ego 条件风险头、对象查询及 Hungarian 监督 |
| 物理 token | `tokenizer.py` | signed-log 坐标量化、Gaussian soft targets、确定性线性插值工具 |
| 小模型规划 | `planner.py` | 随机初始化 causal Transformer，供 CPU 集成测试 |
| 本地 Qwen 规划 | `hf_planner.py` | 原生 tokenizer、连续 Radar/KRS prefix、LoRA、BF16、gradient checkpointing、小型物理词表适配头 |
| 实验入口 | `pipeline.py` / `__main__.py` | 两阶段训练、epoch 断点恢复、验证选优、显式测试、预测 JSON、数据/源码/基础模型指纹 |
| 多卡启动 | `launch_l40s.sh` | torchrun DDP、梯度累积、每 rank 随机状态、只由 rank0 保存产物 |

不自动下载权重。Qwen 基础权重冻结，LoRA 和 Radar/KRS/物理词表适配器可训练；检查点保存可训练语言适配参数，基础模型仍从原本地路径读取。`lora=false` 表示冻结基础模型、只训练适配器，不表示全参数微调。

## 独立环境与一条命令验证

环境安装见 [ENVIRONMENT.md](ENVIRONMENT.md)，固定依赖见 [requirements.txt](requirements.txt)。在项目根目录执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  .conda-radar-vla/bin/python -m radar_vla smoke \
  --output /ssd1/data/RadarVLA/runs/my_pipeline_smoke
```

输出目录须为空或不存在。该命令生成独立 train/val/test 合成场景，使用原生 `[2,256,107]`、4 帧历史，完成 Stage 1、Stage 2、测试指标与结构化预测。语言侧使用随机初始化的小模型，**无需 Qwen 权重**；它验证接口与数值链路，不证明预测质量。

主要产物：`data/manifest.jsonl`、`grounding/{best,last}.pt`、`sft/{best,last}.pt`、各阶段 `protocol.json` / `metrics.jsonl` / `status.json`、`test_metrics.json`、`predictions.jsonl`、`smoke_summary.json`。

## 数据接入契约

manifest 为 JSONL，每行描述当前帧和它的雷达历史。`sensors.camera`、`sensors.lidar`、`map` 可保留，但首版训练不消费相机/LiDAR，也没有实现地图监督。

```json
{
  "schema_version": "radar_vla_v1",
  "sample_id": "scene001_frame005",
  "scene_id": "scene001",
  "split": "train",
  "timestamp_s": 12.5,
  "sensors": {
    "radar": {
      "power": "radar/scene001_frame005_power.npy",
      "unfolded_doppler": "radar/scene001_frame005_doppler.npy",
      "range_m": [0.5, 1.0],
      "azimuth_rad": [-0.2, 0.0, 0.2],
      "time_offsets_s": [-0.3, -0.2, -0.1, 0.0]
    }
  },
  "ego": {
    "velocity": [10.0, 0.0], "acceleration": [0.0, 0.0],
    "yaw_rate": 0.0, "box_size": [4.5, 1.8],
    "future_xy": [[5,0],[10,0],[15,0],[20,0],[25,0],[30,0]],
    "future_valid": [true,true,true,true,true,true]
  },
  "agents": [],
  "future_times_s": [0.5,1.0,1.5,2.0,2.5,3.0],
  "language": {"instruction": "Continue along the lane."}
}
```

示例为便于阅读只列了 2 个 range 和 3 个 azimuth 坐标；真实记录必须填完整的 **256/107 个标定值**，与 NPY 尺寸一致。每个 Power/Doppler NPY 分别为 `[N,256,107]`，加载后合成 `[N,2,256,107]`，batch 为 `[B,N,2,256,107]`。这些坐标必须来自标定，不能用栅格尺寸代替距离或角度。

坐标约定：当前 ego 中心为原点、x 向前、y 向左，所有未来轨迹也在这个固定坐标系中；距离 m、速度 m/s、加速度 m/s²、角度 rad、时间 s。agent `velocity` 是绝对地面速度在当前 ego 轴上的分量；Doppler 是 `(v_agent-v_ego)·line_of_sight`，接近为负。多帧雷达使用各帧原生 RA 栅格，首版没有跨帧 ego pose 显式对齐。

Power 必须是非负线性功率；编码器进行 `log1p` 压缩。只提供 folded Doppler 的数据会被拒绝，需先在外部完成解模糊与单位换算。Power/Doppler 不使用两个独立 backbone。

agent 标注结构：`id`、`position:[x,y]`、`velocity:[vx,vy]`、`heading`、`size:[length,width]`；可加 `future_xy:[H,2]`、`future_yaw:[H]`、`future_valid:[H]`。未来不完整时用布尔掩膜，存储占位坐标仍需为有限数值。首版使用地面平面旋转矩形，原始 `bbox3d` 需在数据转换时投影并记录约定。

`agents=[]` 表示具有可靠标注覆盖的空场景，不能用它表示“没有 tracking 标注”。agent 数量超出配置会报错，避免静默丢弃危险目标。同一个 scene 的所有窗口必须处于同一 split；不能先按帧随机划分再拼历史。真实数据集原生格式的转换器需取得样例后补齐。

## 离线标签、训练、评价

```bash
.conda-radar-vla/bin/python -m radar_vla validate --manifest /data/radar/manifest.jsonl
.conda-radar-vla/bin/python -m radar_vla prepare \
  --manifest /data/radar/manifest.jsonl --output /data/radar/prepared.jsonl
```

风险标签只使用当前 ego 状态、未来 agent 标注；**不使用真实 ego future 计算风险**。后者只用于 SFT 轨迹教师。风险的三个 collision target 为累计事件标签，网络输出对应概率；`dmin` 是 0–3 秒矩形间距的最小值，默认截断至 80m；`areq` 在 0–8m/s²、步长 0.5 的减速网格上搜索，不假定碰撞关于制动力单调。默认安全间距为 0.5m。

无完整未来且无已观测冲突的风险标签会被屏蔽；无可行减速时记录 `braking_feasible=false`、屏蔽 `areq` 回归，不将 8m/s² 当作真值。几何检测是离散时间近似，可能漏掉采样点之间的碰撞，不能用作车辆安全控制保证。

各命令完整参数：

```bash
.conda-radar-vla/bin/python -m radar_vla train --help
.conda-radar-vla/bin/python -m radar_vla evaluate --help
.conda-radar-vla/bin/python -m radar_vla predict --help
```

正式 Qwen 与 8 卡配置、两阶段启动和恢复示例见 [DISTRIBUTED.md](DISTRIBUTED.md)。Stage 1 不加载 Qwen；Stage 2 要求显式传入 Stage 1 检查点和本地 Qwen2.5-3B 路径。二者使用同一份冻结 manifest。

训练只访问 train/val，按验证总损失保存 best；测试需要单独运行 `evaluate`。恢复时校验数据内容、源码、基础语言模型及训练配置，加载优化器和每 rank 随机状态。改变 world size、batch、累积步数或精度需另开实验。checkpoint 按完整 epoch 保存，中断后重做未完成的 epoch。

## 初版证据边界

- 小模型不是预训练 LLM；本地 HF 后端已提供，但真实 Qwen2.5-3B 权重尚未接入。8 卡脚本不代表显存与吞吐已实测。
- SHORT/LONG 教师格式由离线风险标签启发式生成；未知模式样本不参与 token 监督。SHORT 暂取最近两个 agent 和前两个 ego waypoints，尚无 learned critical-agent selection 或人工推理标注。推理时由同一个模型生成模式，不依据 GT risk 切换模型。
- 无地图目标时输出 `ROAD_UNKNOWN`。物理点使用 signed-log 量化；Gaussian 教师以量化后 bin 中心为参考。提供线性插值工具，尚未训练 spline control points。
- 当前径向监督从标注速度和 ego 速度计算，尚未完成 agent 与实测 Radar cell 的对应关联。首版 ego 条件为 speed、forward acceleration、yaw rate，主要面向前向驾驶；横移/倒车场景需扩展条件。
- 现有评价包含风险 AUROC、average precision、Brier/ECE，以及匹配后的物理状态/未来误差。Stage 2 的显式评价还报告自由生成 ego ADE、完整标注时域的 FDE、预测点覆盖率、格式合法率和输出长度；SHORT 缺少后续时刻不能被计为完整轨迹成功。average precision 采用阶梯 PR 面积；只有单一标签类别时 AUROC 为 null。未知监督不计入指标。
- 预测入口是离线标注 manifest 验证工具，模型 forward 不读取 future GT；尚未实现实时无标注流、CARLA、闭环 RFT、collision rate、route completion 等评测。
- 合成 smoke 输出的低损失或格式合法不能作为驾驶效果、真实恶劣天气鲁棒性或安全结论。
