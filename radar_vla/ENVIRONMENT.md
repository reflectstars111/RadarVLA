# RadarVLA 独立环境

本项目使用独立 Conda 前缀，和原多器官项目的 `.conda` 分开：

```text
/ssd1/code/Multi-Organ_Foundation_Model/.conda-radar-vla
```

环境固定 Python 3.10、PyTorch 2.5.1 / CUDA 12.4、Transformers 4.49.0、PEFT 0.14.0，支持接入 Qwen2.5-3B 系列。具体直接依赖见 [requirements.txt](requirements.txt)。模型权重需另行下载，不包含在 Conda 环境中。

在当前服务器创建环境：

```bash
cd /ssd1/code/Multi-Organ_Foundation_Model
/home/user/anaconda3/bin/conda create --prefix "$PWD/.conda-radar-vla" --override-channels -c conda-forge python=3.10 pip -y
.conda-radar-vla/bin/python -m pip install -r radar_vla/requirements.txt
.conda-radar-vla/bin/python -m pip install -e ./radar_vla --no-deps
```

也可使用 [environment.yml](environment.yml) 在其他机器创建同名环境：

```bash
conda env create -f radar_vla/environment.yml
conda activate radar_vla
python -m pip install -e ./radar_vla --no-deps
```

当前工作区推荐直接使用绝对解释器路径，避免误用原项目环境：

```bash
/ssd1/code/Multi-Organ_Foundation_Model/.conda-radar-vla/bin/python -m radar_vla --help
/ssd1/code/Multi-Organ_Foundation_Model/.conda-radar-vla/bin/python -m pip check
```

CPU 测试无需占用训练 GPU：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .conda-radar-vla/bin/python -m pytest -q tests/test_radar_vla*.py
```

实际 Radar 单帧维度为 `[2, 256, 107]`，历史打包后为 `[T, 2, 256, 107]`。八卡 L40S 训练配置与显存预算由训练配置控制；安装 CUDA 运行库本身不会占用 GPU，也不意味着已验证八卡训练。

合成数据仅用于接口和训练链路检查。生成器拒绝写入非空目录，避免覆盖已有数据。

## 当前工作区安装验证

独立前缀已实际安装，使用本地 Conda 缓存中的 26 个基础包新建 Python 3.10.20，再安装上述 pip 依赖；没有克隆原多器官项目的 pip 软件栈。实际占用约 5.5 GB，原 `.conda` 未修改。

已验证 `pip check` 无依赖冲突；在屏蔽 GPU 的条件下，Torch、Transformers、PEFT 等可以导入，随机初始化的小型 Qwen2 模型完成 CPU 前向。此检查验证库集成，不代表已下载 Qwen2.5-3B 权重或已验证八卡训练。
