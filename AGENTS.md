# RadarVLA 工作目录约定

- 本项目唯一工作目录和 Git 根目录是 `/ssd1/code/yqq/radar_vla`。
- Python 源码直接位于本目录，配置、测试和文档分别位于 `configs/`、`tests/`、`docs/`。不要重新创建 `RadarVLA/` 嵌套仓库或维护第二份源码。
- 使用本目录的 `.conda` 环境。包由根目录 `pyproject.toml` 映射为 `radar_vla`，通过 editable 安装导入；不要依赖临时 `PYTHONPATH`。
- 安装当前源码：`.conda/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .`。
- 运行专项测试：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .conda/bin/python -m pytest -q tests`。
- 启动入口为 `.conda/bin/radar-vla`、`.conda/bin/python -m radar_vla` 和 `./launch_l40s.sh`。训练和实验子进程从本项目根目录启动。
- Conda 环境、数据、权重、构建输出和 `artifacts/` 备份由 `.gitignore` 排除，不能加入版本控制。
- `docs/RADARVLA_PIPELINE_V1.md` 和 `artifacts/` 内的旧路径属于历史记录；当前操作以 `README.md`、`ENVIRONMENT.md`、`DISTRIBUTED.md` 为准。
