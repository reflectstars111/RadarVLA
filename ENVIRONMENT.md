# RadarVLA 独立环境

环境、源码、配置、测试和文档统一位于本目录。当前服务器的环境路径为：

```text
/ssd1/code/yqq/radar_vla/.conda
```

本目录同时是 Git 根目录和 Python 包源码目录。当前 Radar 环境固定 Python 3.10、PyTorch 2.5.1 / CUDA 12.4、Transformers 4.49.0、PEFT 0.14.0；直接依赖见 [requirements.txt](requirements.txt)。Qwen 权重需单独准备。

## 从本目录安装

```bash
cd /ssd1/code/yqq/radar_vla
/home/user/anaconda3/bin/conda create --prefix "$PWD/.conda" --override-channels -c conda-forge python=3.10 pip -y
.conda/bin/python -m pip install -r requirements.txt
.conda/bin/python -m pip install -e . --no-deps
```

也可从同一目录使用完整环境定义：

```bash
conda env create --prefix "$PWD/.conda" -f environment.yml
conda activate "$PWD/.conda"
python -m pip install -e . --no-deps
```

安装为 editable 包后，可以在任意工作目录使用本环境的 Python 运行 `python -m radar_vla`，无需设置 `PYTHONPATH`。已有完整依赖时，可离线重新安装源码映射：

```bash
.conda/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .
```

启动器 `launch_l40s.sh` 默认使用与脚本同目录下的 `.conda`，也可通过 RADAR_VLA_ENV / RADAR_VLA_PYTHON 显式覆盖；它与实验队列均从本项目根目录启动子进程。

## 验证和测试

在 RadarVLA 目录内：

```bash
.conda/bin/python -m radar_vla --help
.conda/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .conda/bin/python -m pytest -q tests
```

从其他工作目录检查包映射与命令入口：

```bash
/ssd1/code/yqq/radar_vla/.conda/bin/python -I -c 'import radar_vla; print(radar_vla.__file__)'
/ssd1/code/yqq/radar_vla/.conda/bin/radar-vla --help
```

## 迁移说明

2026-10-06 的独立环境分离记录保留在 `artifacts/reorganization_20261006/`。2026-10-13 将源码、Git 元数据和环境统一至 `/ssd1/code/yqq/radar_vla`：保留原提交历史，移除嵌套源码副本；使用 Conda 的前缀替换机制修复环境内的文本和二进制前缀、命令解释器路径，再重新安装 editable 包。环境、权重和本地产物由 `.gitignore` 排除。

本次迁移备份和清单位于 `artifacts/workspace_consolidation_20261013/`。历史验证记录中的旧路径仅用于追溯，当下操作使用本文档路径。

真实单帧 Radar 为 `[2,256,107]`，四帧输入为 `[4,2,256,107]`。目标服务器的完整 Qwen2.5-3B 与 8 卡 L40S 容量、吞吐仍需实测。Windows 上应重新创建环境，不能直接复制 Linux `.conda` 使用；正式多卡启动器为 Bash/Linux 入口。
