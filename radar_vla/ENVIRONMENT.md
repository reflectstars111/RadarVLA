# RadarVLA 独立环境

环境、源码、配置、测试和文档统一位于本目录。当前服务器的环境路径为：

```text
/ssd1/code/Multi-Organ_Foundation_Model/radar_vla/.conda
```

医学项目继续使用上一级的 `.conda`，两者独立。当前 Radar 环境固定 Python 3.10、PyTorch 2.5.1 / CUDA 12.4、Transformers 4.49.0、PEFT 0.14.0；直接依赖见 [requirements.txt](requirements.txt)。Qwen 权重需单独准备。

## 从本目录安装

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model/radar_vla
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

安装为 editable 包后，可以在本目录、仓库根目录或其他工作目录运行 `python -m radar_vla`。启动器 `launch_l40s.sh` 默认使用与脚本同目录下的 `.conda`，也可通过 RADAR_VLA_ENV / RADAR_VLA_PYTHON 显式覆盖。

## 验证和测试

在 RadarVLA 目录内：

```bash
.conda/bin/python -m radar_vla --help
.conda/bin/python -m pip check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .conda/bin/python -m pytest -q tests
```

从医学仓库根目录执行专项测试时：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 radar_vla/.conda/bin/python -m pytest -q radar_vla/tests
```

医学测试仍使用原环境和入口：从仓库根目录运行 `.conda/bin/python -m pytest -q tests`。RadarVLA 不再向该目录添加测试或依赖。

## 迁移说明

2026-10-06 将原独立 Radar 环境通过 Conda clone 迁入本目录，保留安装的依赖版本，并重新安装 editable 包和命令入口。迁移完成后删除仓库根目录的旧 Radar 环境，不保留指向旧路径的启动依赖。医学环境没有克隆或修改。

真实单帧 Radar 为 `[2,256,107]`，四帧输入为 `[4,2,256,107]`。目标服务器的完整 Qwen2.5-3B 与 8 卡 L40S 容量、吞吐仍需实测。Windows 上应重新创建环境，不能直接复制 Linux `.conda` 使用；正式多卡启动器为 Bash/Linux 入口。
