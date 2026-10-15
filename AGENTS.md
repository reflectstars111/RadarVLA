# RadarVLA 工作目录约定

- 本项目 Git 根目录是包含此文件与 `pyproject.toml` 的 `RadarVLA/`；不要依据机器特定的绝对路径寻找源码。
- Python 源码位于 `src/`，配置、测试和文档分别位于 `configs/`、`tests/`、`docs/`。不要重新创建嵌套仓库或维护第二份源码。
- 使用 `/home/luxinlan/.conda/envs/lxl_radarvla` 环境。包由根目录 `pyproject.toml` 将 `src/` 映射为 `radar_vla`，通过 editable 安装导入；不要依赖临时 `PYTHONPATH`。
- 安装当前源码：`/home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pip install --no-index --no-deps --no-build-isolation -e .`。
- 运行专项测试：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pytest -q tests`。
- 操作入口为 `scripts/radarvla.sh`，命令见根目录 `RUNBOOK.md`。训练和实验子进程从本项目根目录启动。
- Conda 环境、数据、权重、构建输出和 `artifacts/` 备份由 `.gitignore` 排除，不能加入版本控制。
- 原始采集入口位于 `collection/collector/`，其测试和 notebook 分别位于 `collection/tests/`、`collection/notebooks/`；采集契约、runbook 和验收报告位于 `docs/collection/`。`collection/weights/` 是本地权重，不属于模型源码。
- `docs/RADARVLA_PIPELINE_V1.md` 和 `artifacts/` 内的旧路径属于历史记录；当前操作以 `RUNBOOK.md`、`ENVIRONMENT.md`、`DISTRIBUTED.md` 为准。
