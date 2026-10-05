# RadarVLA

Doppler-grounded, risk-adaptive Radar-Language-Action modeling. 当前版本 **v2 / 0.2.0**。

原生 Radar 输入 `[2,256,107]`，默认 4 帧历史；同一 Qwen2.5-3B 根据连续风险条件生成长/短时域结构化规划。目标训练环境为 **8×L40S，每卡 44GB**。

代码覆盖完整 `frame_t` 导入、标定与时序处理、反事实风险标签、物理 grounding、Qwen SFT、样条轨迹生成、离线评价、无标注推理和 Q1–Q5 配对实验入口。

- [完整使用说明](radar_vla/README.md)
- [数据字段与物理坐标规范](radar_vla/DATA_FORMAT.md)
- [环境安装](radar_vla/ENVIRONMENT.md)
- [8 卡训练、恢复与手动监测](radar_vla/DISTRIBUTED.md)
- [原始方案](docs/RadarVLA_risk_adaptive_plan.md) · [数据结构](docs/data_record.md)
- [v2 实现与验证记录](docs/RADARVLA_PIPELINE_V2.md)

## 安装

远端早期提交包含 Conda 环境文件。以下稀疏检出只下载代码工作树，并根据固定依赖重新创建本机环境；保留原有 Git 历史。

```bash
git clone --filter=blob:none --sparse https://github.com/reflectstars111/RadarVLA.git
cd RadarVLA
git sparse-checkout set radar_vla tests docs
conda env create --prefix "$PWD/.conda-radar-vla" -f radar_vla/environment.yml
.conda-radar-vla/bin/python -m pip install -e ./radar_vla --no-deps
.conda-radar-vla/bin/python -m pip check
```

若已有可用独立环境，可直接用其解释器安装本包。源码不依赖原医学分割仓库。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .conda-radar-vla/bin/python -m pytest -q tests/test_radar_vla*.py

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .conda-radar-vla/bin/python -m radar_vla smoke --output ./runs/v2_check
```

专项 **127 项通过**，含实际小型 Qwen、LoRA、BF16、双进程 CPU DDP，以及完整原始帧→两阶段训练→无标注推理。`smoke` 显式采用合成样本与随机小语言模型验证接口；正式 HF 配置缺本地 Qwen 权重会报错。

真实 Radar 数据、完整 Qwen2.5-3B 和 8 卡 L40S 的效果、显存与吞吐尚未实测。方案中可选的闭环 RFT/CARLA 联机评测未包含在当前离线实现中。
