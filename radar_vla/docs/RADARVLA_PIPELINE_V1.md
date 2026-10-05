# RadarVLA 初版交付与验证（2026-10-05）

实现依据：[数据字段草案](data_record.md)、[风险自适应方案](RadarVLA_risk_adaptive_plan.md)。用户补充：单帧 `[2,256,107]`；未来训练服务器为 8×L40S、每卡 44GB；本地语言模型为 Qwen2.5-3B；需要独立 Conda 环境。真实数据与完整语言模型权重尚未准备。

代码与完整使用说明位于 [radar_vla/README.md](../README.md)。本模块不依赖 `unet_moe`，不加入医学分割的任务队列，不改变既有训练参数。当前独立环境为 `/ssd1/code/Multi-Organ_Foundation_Model/radar_vla/.conda`，输出根目录为 `/ssd1/data/RadarVLA/`。

## 首版交付

- JSONL/NPY 数据接口、实际物理坐标和 Doppler 符号约定、scene split 检查。
- 无新增干预 CV/CTRV rollout、累计冲突/最小间距/最小减速度标签及不完整监督掩膜。
- 共享 Power/Doppler CNN、极坐标/时间编码、跨帧压缩、风险和物理状态预测。
- 双阶段训练、验证选优、断点恢复、显式测试、结构化轨迹预测。
- 同一生成模型接收连续 Radar/KRS tokens；提供 CPU 小模型与本地 Qwen2.5-3B + LoRA 后端。
- 8 卡 DDP 启动脚本、BF16、梯度累积、activation checkpointing；检查点保留每 rank 随机状态。
- 独立 Conda 环境、固定依赖与服务器配置；见[环境文档](../ENVIRONMENT.md)和[多卡文档](../DISTRIBUTED.md)。

## 已执行验证

| 检查 | 结果 |
|---|---|
| 独立环境依赖 | `pip check` 无冲突；Python3.10.20、torch2.5.1+cu124、transformers4.49.0、peft0.14.0 |
| RadarVLA 专项测试 | 新环境 **51 passed**，11.11 秒 |
| 全仓库回归 | 原环境 **229 passed**，21.70 秒；6 条旧环境 DeepSpeed/Pydantic 弃用警告，无失败 |
| 真实 HF 接口 | 本地生成的微型 Qwen2：前向/LoRA梯度、BF16、生成、适配器恢复通过 |
| 分布式 | 实际双进程 CPU/Gloo：grounding 与 HF/LoRA SFT，含梯度累积与非重入 checkpoint，通过 |
| 端到端 | 独立环境、原生 `[2,256,107]`、4帧，合成数据两阶段+测试+预测全部完成 |
| 脚本与文档 | shell 语法、启动命令参数、相对链接及 `git diff --check` 通过 |

端到端产物：`/ssd1/data/RadarVLA/runs/pipeline_smoke_20261005_v1/`，含 `smoke_summary.json`、`test_metrics.json`、`predictions.jsonl` 和两阶段检查点。小模型仅训练一轮，预测没有科学效果意义；例如此次输出均为 SHORT，完整时域 FDE 为 null，评价同时记录仅 1/3 的预测点覆盖率，没有把缺少后续轨迹当作成功。

复现专项检查：

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  radar_vla/.conda/bin/python -m pytest -q radar_vla/tests
```

## 后续真实训练所需

准备真实数据样例、标定的 range/azimuth、时间戳和 tracking/future 标签，再编写数据集专属转换器；按 manifest 校验后离线生成风险标签。配置真实 Qwen2.5-3B 权重路径，先执行单卡小规模验证，再使用 8 卡脚本。

当前没有真实 3B 权重/8 卡 NCCL/L40S 峰值显存实测，没有提交长期训练。相机/LiDAR teacher、地图目标、实测 Radar cell 与 agent 对应、CARLA 闭环与 RFT 尚未实现，具体初版限制已在主说明列出。
