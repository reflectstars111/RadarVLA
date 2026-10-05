# RadarVLA 独立环境

在 RadarVLA 仓库根目录执行。环境固定 Python 3.10、PyTorch 2.5.1、Transformers 4.49.0、PEFT 0.14.0、Pillow 11.2.1；直接依赖见 [requirements.txt](requirements.txt)。模型权重单独提供，不包含在环境中。

## 推荐安装

```bash
conda env create --prefix "$PWD/.conda-radar-vla" -f radar_vla/environment.yml
.conda-radar-vla/bin/python -m pip install -e ./radar_vla --no-deps
.conda-radar-vla/bin/python -m pip check
.conda-radar-vla/bin/python -m radar_vla --help
```

或创建命名环境：

```bash
conda env create -f radar_vla/environment.yml
conda activate radar_vla
python -m pip install -e ./radar_vla --no-deps
export RADAR_VLA_PYTHON="$(command -v python)"
```

启动脚本默认解释器为仓库 `.conda-radar-vla/bin/python`。其他环境使用 `RADAR_VLA_PYTHON` 或 `RADAR_VLA_ENV` 指定路径。远端早期版本跟踪过完整 Conda 前缀；这种前缀含机器相关路径，建议按 [根 README](../README.md) 稀疏检出源码后重建环境。不要把已经存在的非空前缀作为新环境创建目标。

## 验证

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .conda-radar-vla/bin/python -m pytest -q tests/test_radar_vla*.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  .conda-radar-vla/bin/python -m radar_vla smoke --output ./runs/my_v2_check
```

实际 Radar 单帧 `[2,256,107]`，历史输入 `[T,2,256,107]`。合成检查不占用训练 GPU；输出目录必须为空或不存在。

迁移前已在独立 Python 3.10.20 环境完成 127 项测试和原生尺寸端到端检查，`pip check` 无冲突。真实小型 Qwen 完成 CPU 前向、LoRA、BF16 与双进程 DDP 验证。完整 Qwen2.5-3B 权重和 8 卡实际显存尚未验证；部署说明见 [DISTRIBUTED.md](DISTRIBUTED.md)。
