# 原始采集数据清单与三联 BEV

在仓库根目录运行，使用已安装 editable 包的环境：

```bash
PY=/home/luxinlan/.conda/envs/lxl_radarvla/bin/python
DATA=/home/luxinlan/data/lxl_data/RadarVLA

# 清单：统计 raw 索引、prepared 各版本、排除原因及完成标记
"$PY" scripts/inspect_raw_data.py inventory --data-root "$DATA" --output-dir "$DATA/visualizations"

# 均匀抽取 5 帧，输出 LiDAR / Power / unfolded Doppler 三联 PNG
"$PY" scripts/inspect_raw_data.py render --data-root "$DATA" --count 5

# 按场景、天气、split 筛选；也可使用 --episode-id
"$PY" scripts/inspect_raw_data.py render --data-root "$DATA" --category scenario2 --weather rain --split train --count 3

# 指定 job 与 CARLA frame（不是 source index 或 radar_frame_index）
"$PY" scripts/inspect_raw_data.py render --data-root "$DATA" --job-id normal_Town01_19_fog_e49b1e8d61 --carla-frame 5025661

# 显示完整雷达范围，仅保留 reliable Doppler 单元
"$PY" scripts/inspect_raw_data.py render --data-root "$DATA" --count 1 --full-range --reliable-only
```

Notebook：`notebooks/raw_data_bev.ipynb`，选择 lxl_radarvla 环境，依次运行单元。
支持 split / category / weather / episode / job 筛选、滑块、前后帧、显示范围和 PNG 导出。
如果 Jupyter 中没有名为 lxl_radarvla 的 kernel，直接选择解释器 `/home/luxinlan/.conda/envs/lxl_radarvla/bin/python`。

命令也支持 `python -m radar_vla.raw_visualization inventory|render ...`。
这是独立数据检查入口，不依赖训练入口或 checkpoint。

输出默认位于数据根目录 `visualizations/examples/`，交互导出在 `visualizations/interactive/`。
`--output-dir` 可修改位置。图片使用 job + CARLA frame 命名；重复导出同名帧会覆盖图片。
每次批量导出的 `render_report.json` 记录本次图片、source index、实际雷达窗口和可靠单元占比。
`--count` 按筛选结果的索引顺序均匀采样；不保证抽到连续帧。相同 CARLA frame 可能出现在不同 episode，建议同时指定 job。

## 数据与坐标

- 正式原始入口是 `full/dataset_index.jsonl`，不能把 `full/jobs/` 所有任务都算作有效样本。
- 读取原始索引全部 split，包含 prepared 排除样本；准备版本与排除原因在 inventory 报告中。
- raw CARLA 使用 x-forward / y-right，图上右就是车辆右侧。prepared 使用 y-left，不能直接混叠。
- LiDAR 的六列为局部 x/y/z、cosine、actor ID、semantic tag；前三列按记录的安装位姿变换到 ego。
- Power 读取 `folded_ra.npz:power_peak_ra`；速度读取 `unfolded_ra.npz:doppler_unfolded_dense_ra_mps`。
- 雷达保留保存的 RA 单元值，以极坐标单元边界绘制 BEV，不对速度或强度做插值。
- Power 色阶相对该 radar window 的 `global_power_peak`，跨窗口不代表绝对标定强度。
- Doppler 默认显示全部保存数值，可靠性开关仅影响速度底图，不改变 GT 轨迹与几何。
- ego 青色，agent 按 ID 固定配色；未来为 anchor 中 30 × 0.1 s 真实记录，无效点为 NaN 断线。
- 道路使用 `ego_road_corridor_50m_v1`；道路边界从中心线与记录宽度推导，不是独立采集的边界真值。无效宽度处不画边界，不填补缺失道路。
- agent 标签来自已采集的约 50 m 雷达视场，LiDAR 和 radar 可覆盖更大范围。
- 数组按帧惰性加载；首次进入 episode 会扫描 anchor 文件建立字节偏移索引，后续直接定位，不把所有未来轨迹读入内存。
- inventory 检查索引、报告和窗口完成标记，不等于全量数组审计。选中帧会检查文件、数组形状、数值与各模态 frame ID。

## 验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/radarvla_mpl /home/luxinlan/.conda/envs/lxl_radarvla/bin/python -m pytest -q tests/test_raw_visualization.py tests/test_visual_diagnosis.py
```

专项测试覆盖旋转 ego、传感器平移/偏航、正方位角朝右、跨窗口索引、路径迁移、未来缺失断线、道路分段、错帧/缺文件报错、Doppler mask 及交互翻帧/导出。

当前宿主环境注册了 ROS pytest 插件，但缺少 lark；验证命令禁用无关插件自动加载。无需安装或修改系统依赖。
