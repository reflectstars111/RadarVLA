# 8×L40S 瓶颈对照实验

分支：`codex/bottleneck-fixes-20261023`；代码基线：`959e69a`。
新结构均为显式选项，旧 token 模型与单/多车道对照保留。本轮验证了实现与训练链路，
没有新增真实数据训练成绩，也没有测过 Qwen2.5-3B 在八卡服务器上的显存峰值或加速倍数。

## 1. 更新与预检

在现有仓库根目录执行；不要重新创建嵌套源码目录。已有未提交修改应先单独保存。

```bash
git fetch origin
git switch codex/bottleneck-fixes-20261023
git pull --ff-only origin codex/bottleneck-fixes-20261023
/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
bash scripts/radarvla.sh doctor
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

`configs/local_paths.json` 保持服务器现有配置：Python 为
`/home/luxinlan/.conda/envs/lxl_radarvla/bin/python`，数据根为
`/home/luxinlan/data/lxl_data/RadarVLA`，本地 Qwen 为
`collection/weights/Qwen2.5-3B`。换机器时可用 `RADARVLA_PATHS_CONFIG` 指向本机路径 JSON；
不要修改系统全局环境，也不要设置临时 `PYTHONPATH`。

本方案固定使用已经生成的 KRS-v2 moving 数据：

```bash
bash scripts/radarvla.sh validate --full \
  --manifest /home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_krsv2_0p5_1_2/manifest.jsonl
bash scripts/radarvla.sh bottleneck --arm baseline --stage grounding --dry-run
```

正式启动同时检查 `validation.json` 的 `PASS` 状态与 manifest SHA256、初始化权重、
Qwen 本地文件、选中 CUDA 设备数量和原生 BF16 支持。dry-run 只打印命令，
不导入模型、不探测 GPU、不训练；它不替代数据与硬件预检。
原始 Doppler 产品若依赖后续帧，真实延迟实验仍须计入数据等待或改用因果产品。

## 2. 配置与训练顺序

主配置为 `configs/bottleneck_8xl40s.json`；五组模型在 `configs/bottleneck_v1/`。
`manifest` 相对 `data_root`，`output_root` 相对 `run_root`，模型配置路径相对仓库根目录。
默认输出为 `<run_root>/bottleneck_20261023/<arm>/<stage>/`。

| arm | 检测分配 | Risk 输入 | 自车动作 |
| --- | --- | --- | --- |
| baseline | 原 anchor | 雷达 | 原 token 样条 |
| assignment | 局部质量一对一 | 雷达 | 原 token 样条 |
| risk | 原 anchor | 雷达＋预测物理候选 | 原 token 样条 |
| action | 原 anchor | 雷达 | LLM query＋连续加速度 |
| combined | 局部质量一对一 | 雷达＋预测物理候选 | LLM query＋连续加速度 |

所有配置启用相同 HF KV cache，保留相同多车道描述格式、骨干和数据协议。
`max_agents=8` 仅限制语言描述长度；检测与 Risk 融合没有 top-8 裁剪。
当前数据是 moving-vehicle 协议，不据此宣称已检测所有静止车辆、行人和自行车。

| stage | epoch | 每卡 batch × 累积 | 8 卡全局 batch | 学习率 | 训练范围 |
| --- | --- | --- | --- | --- | --- |
| grounding | 20 | 1 × 1 | 8 | 2e-4 | Grounder |
| risk-readout | 10 | 1 × 1 | 8 | 2e-4 | 仅 Risk query/attention/head 与新融合模块 |
| sft | 5 | 1 × 4 | 32 | 5e-5 | 冻结 Grounder，训练规划 adapters/head 与 LoRA |

BF16、seed=42、每 rank 两个 worker、按 optimizer update 调度；各阶段前 5/2/1 个 epoch
平坦学习率后 cosine 衰减至 .1 倍。SFT 每轮固定 32 个验证样本自由生成。
这是新的八卡基线，不能将其与历史单卡结果当成只改一个变量的严格对照。

先完成两个 Grounder，以下命令顺序运行，各自占用八张卡，不要同时启动：

```bash
bash scripts/radarvla.sh bottleneck --arm baseline --stage grounding
bash scripts/radarvla.sh bottleneck --arm assignment --stage grounding
```

再完成 Risk 读出对照。baseline 与 risk 从同一个 baseline `epoch_020.pt` 开始；
combined 从 assignment `epoch_020.pt` 开始。相同 schema 权重必须完整；
旧雷达 Risk → 物理融合是唯一允许的显式迁移，不静默丢弃旧键或补齐损坏 checkpoint。

```bash
bash scripts/radarvla.sh bottleneck --arm baseline --stage risk-readout
bash scripts/radarvla.sh bottleneck --arm risk --stage risk-readout
bash scripts/radarvla.sh bottleneck --arm combined --stage risk-readout
```

然后运行 SFT：

```bash
bash scripts/radarvla.sh bottleneck --arm baseline --stage sft
bash scripts/radarvla.sh bottleneck --arm action --stage sft
bash scripts/radarvla.sh bottleneck --arm assignment --stage sft
bash scripts/radarvla.sh bottleneck --arm risk --stage sft
bash scripts/radarvla.sh bottleneck --arm combined --stage sft
```

默认 baseline/action 共用 baseline Grounder `epoch_020.pt`，assignment 使用其自身
Grounder `epoch_020.pt`；risk/combined 使用对应读出阶段 `epoch_010.pt`。
这样 baseline/action 直接比较动作结构，baseline/assignment 的 grounding 直接比较检测分配。
**Risk 是否有效，应首先比较 baseline/risk 的同样 10 轮读出结果**，不能只将 risk 的额外
10 轮训练与未经读出训练的 baseline SFT 相比后归因于结构。
若评估 Risk 对规划的独立收益，增加经过相同读出轮数的 token 对照：

```bash
bash scripts/radarvla.sh bottleneck --arm baseline --stage sft \
  --init-grounding /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/baseline/risk-readout/epoch_010.pt \
  --output /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/baseline_readout_sft
```

combined 是可用组合候选，不能单靠它对 baseline 的差值归因三项独立贡献。
第一轮以固定 epoch 比较，不挑选各自最有利的 `best.pt`。可在任意命令追加 `--dry-run`；
中断后用原命令加 `--resume`。恢复必须保持 manifest、配置、初始化来源和训练协议不变。
`--gpus 0,1` 等子集允许诊断，但会改变有效 batch，不能混入八卡对照。
改学习率、epoch 或 batch 应复制 profile 后传 `--profile`，使用新输出目录，不能伪装续训。
自动初始化路径按该 profile 的源阶段最终 epoch 生成；显式 `--init-grounding` 始终优先。

启动时的 `CAMPAIGN` JSON 会打印 profile SHA、GPU、配置、初始化路径与有效 batch，建议保存终端日志。
训练输出保存完整 `protocol.json`、`initialization.json`、每轮实际样本顺序哈希（逐 rank）、
分配回退数、optimizer 学习率序列、显存峰值与验证结果。比较分配两组之前先检查
共同初始权重哈希和实际样本顺序一致；相同 seed 本身不保证公平比较。

## 3. 评估与输出

下面以 action 为例；其余组更换路径即可。`evaluate` 使用训练冻结的同一 manifest
及 test split。所有模型比较相同时间点、有效 mask 与共同目标集合，并单列 episode 统计。

```bash
/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m radar_vla evaluate \
  --checkpoint /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/action/sft/epoch_005.pt \
  --manifest /home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_krsv2_0p5_1_2/manifest.jsonl \
  --output /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/action/test_metrics.json \
  --device cuda --batch-size 1
/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m radar_vla predict \
  --checkpoint /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/action/sft/epoch_005.pt \
  --manifest /home/luxinlan/data/lxl_data/RadarVLA/prepared/taskwise_moving_krsv2_0p5_1_2/manifest.jsonl \
  --output /home/luxinlan/data/lxl_data/RadarVLA/model_runs/bottleneck_20261023/action/predictions.jsonl \
  --device cuda --batch-size 1
```

新动作的 `generated_tokens=0`：一次 LLM query 输出 mode 与六个连续加速度节点，
再积分得到自车轨迹。默认 A≤8 m/s²、J≤20 m/s³，LONG 为 3s、SHORT 原生时域为 1s，
SHORT 额外释放末端加速度并接受限停车。初始加速度缺测时显式使用 CV/CTRV 估计。
动力学约束不等于车辆可驾驶性或碰撞安全；采用过渡也可能被拒绝，应检查 `executed_plan`。
无历史执行计划的普通 predict 使用观测 v/yaw 的 CV/CTRV 延迟基线，真实控制需接入实际执行状态。

加 `--include-description` 会在该批动作全部就绪后产生独立 `description`，包含原有道路
（多车道中心线、边界、相邻关系）与目标描述；ego 数值字段不用于执行。
`description_seconds` 单列，不能替换动作延迟。描述模式与动作模式可不同。
这是同步离线 CLI，最终文件仍在推理完成后写出，尚不是实时异步控制服务。
`generation_seconds` 含动作生成与解码，尚不含 CPU 采用桥接/执行检查与通信控制开销；
实际端到端响应测试须补齐这些部分。

连续动作 SFT 损失为有效位置 SmoothL1＋模式 CE＋.1×描述 CE；有效标签和描述项单列报告。
位置标签差分没有被当作瞬时速度 GT。新动作拒绝 Oracle/GT Risk 混入 action query。
新 checkpoint 与旧 token checkpoint 显式区分，不能跨 schema 静默加载。

**现有离散 token RL 仅支持 baseline/assignment/risk 的 token 模型。** action/combined
目前支持 SFT、predict、evaluate；其 `train-rl` / `evaluate-rl` 会明确拒绝。
连续策略分布、动作密度与概率重评分需要后续实现，不能复用 token log-probability。
旧 token 可独立测试 `configs/response_rl_adoption_v1.json` 的桥接与 latest-ready 调度。

## 4. 验收依据

本机使用用户指定 `.conda`，editable 导入当前 `src/`；测试包含真实微型 Qwen、BF16/LoRA、
两进程 CPU DDP、冻结感知、保存重载、断点恢复、未知 mask 与新旧 schema 隔离。
2026-10-23 完整回归 **683 passed / 102.90 s**；十条八卡训练启动路径的 dry-run 通过，
隔离环境 `pip check` 无冲突、Shell 语法与 `git diff --check` 通过。原始日志与 JUnit 位于本机忽略目录
`artifacts/bottleneck_fixes_20261023/`，不是服务器实测成绩。

检测比较固定 FP 预算的危险目标召回、连续漏检和共同 GT 的轨迹误差；Risk 比较独立危险
事件上的 .5/1/2s 召回/误报、危险 dmin/正 areq；动作比较同点 ADE、停驻/起步、执行有效率、
持续有效变更延迟及误响应。不要仅凭总 loss、语法通过率或推理时间就采纳。
现有短窗危险正例很少，按 episode 划分，阈值在独立校准集冻结，不能把邻帧当独立样本。
更多证据与限制见 [瓶颈分析](BOTTLENECK_REPAIR_V1.md)。
