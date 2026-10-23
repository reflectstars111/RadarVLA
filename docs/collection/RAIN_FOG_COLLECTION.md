# 雨雾 10k 采集

从项目根目录运行 `bash scripts/collect_rain_fog_10k.sh ACTION`。

| ACTION | 行为 |
|---|---|
| `check` | 只读检查配置、地图、磁盘、GPU、端口及 smoke 资格 |
| `status` | 只读显示批次进度、失败原因和结果路径 |
| `smoke` | 真实雨天 S2 pilot、雨雾原始采集、雷达、sample prepare、验证及预览；不启动正式采集 |
| `run` | 要求当前版本 smoke PASS，采集全新正式批次并完成雷达、prepare、validate |
| `resume` | 恢复同一正式批次；输入或代码改变则拒绝续跑 |

运行示例：

```bash
cd /home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA
bash scripts/collect_rain_fog_10k.sh check
bash scripts/collect_rain_fog_10k.sh status
# 以下命令由使用者在确认启动正式采集时运行：
screen -L -Logfile /home/luxinlan/data/lxl_data/RadarVLA/analysis/rain_fog_10k_v1/full_run.log -dmS radarvla_rain_fog bash scripts/collect_rain_fog_10k.sh run
# 中断后：
screen -L -Logfile /home/luxinlan/data/lxl_data/RadarVLA/analysis/rain_fog_10k_v1/full_resume.log -dmS radarvla_rain_fog bash scripts/collect_rain_fog_10k.sh resume
```

脚本使用 pipeline 锁防止重复启动，不训练模型，不合并历史 manifest。

## 目录与配额

数据根目录：`/home/luxinlan/data/lxl_data/RadarVLA`。

- `full/rain_fog_10k_v1/`：正式 episode、雷达产品、accepted anchors、dataset index。
- `prepared/rain_fog_moving_fullbbox_v1/`：manifest、雷达缓存、排除记录、完整验证结果。
- `analysis/rain_fog_10k_v1/`：日志、报告、代码备份及按代码签名区分的 smoke 尝试。每个 smoke 尝试包含原始 episode、pilot、sample prepared 和联合预览。

| 组 | rain | fog | 合计 |
|---|---:|---:|---:|
| normal | 2000 | 2000 | 4000 |
| scenario2 | 1500 | 1500 | 3000 |
| scenario7_9 | 1000 | 1000 | 2000 |
| scenario10 | 500 | 500 | 1000 |

10000 指原始 anchor，不是最终训练条数。原始 train/val/test 配额为 8000/1000/1000；保持 episode 隔离，S2 位于 train。S3、S4 配额为零。S7/8/9 合并计数，不保证组内相等。

批次配置为 `collection/configs/collection_rain_fog_10k_v1.yaml`，沿用历史 rain/fog 全部参数。显卡默认 CARLA=0、radar=0（1 个 worker）；CARLA 退出后才处理雷达，因此两阶段顺序共用空闲 GPU 0；启动前检查它们是否空闲，繁忙时直接停止而不影响其他任务。更改配置需重做对应版本 smoke。候选 seed 每次增加 30 轮，上限 300；到上限仍有缺额则保留数据并报告，不放宽筛选标准。

## 标签与传感器

新 policy：`moving_vehicle_01_fullbbox50_background_v2`。仅当前世界水平速度 ≥0.1 m/s 的车辆进入目标集合，自行车、行人、静止车辆是背景，摩托车沿用车辆类别。车辆世界 BBox 全部顶点的 XY 投影必须严格位于雷达中心的 50 m 圆内；接触或跨界作为背景，整帧保留，且不设 ignore 区。±53° 沿用原 BBox 相交规则。未来停车或驶出范围不截断已保存的真实 future。

检测、Local Risk、planner 目标、Mode 和 task masks 同步重建。新训练数据使用配套 `configs/l40s_qwen25_3b_rain_fog_fullbbox_v2.json`；现有训练默认配置和历史数据保持不变。历史 moving v1 可以读取，但新采集和 prepare 不生成旧标签。

每帧保存前视 RGB 和原始 `semantic_lidar.npy`（x、y、z、cos_incidence、ObjIdx、semantic_tag），LiDAR 量程约 118 m。50 m 规则只限制标签，不裁剪原始点云或雷达信号。C-Shenron 的散射点筛选和稠密解折叠保持原实现。

## prepared 来源和验收

处理顺序是：CARLA 同步 episode → 完整真实未来与场景证据筛选 → C-Shenron 雷达及解折叠 → dataset index → moving/taskwise prepare → 全量 sensor validate → 原始快照成员集合审计。

历史 `full_v3_0a225d38dde3` 从 10000 个 anchor 得到 7751 条；当前 `taskwise_moving_v1` 经任务恢复及标签迁移后为 9841 条。新批次直接使用现有 moving prepare 通路，不调用固定历史样本数的迁移脚本。

`smoke_release.json` 绑定采集、转换源码及配置签名，并绑定原始 pilot、雷达审计、sample manifest 和验证报告的哈希。只有完整烟测 PASS 才允许正式采集。烟测输出包含 `front_rain.png`、`front_fog.png` 和对应联合雷达/地图/真实轨迹预览。

任何失败都保留现场。中断后先检查 `status` 和日志；未完成的原始 episode 不被覆盖或默默重采。完整雷达窗口和 prepared journal 可恢复，输入或代码签名改变则拒绝混用。


## 天气渲染修正

本批次强制 Vulkan / Epic，保存 `rendering.json`，并检查进程确实加载 Vulkan。
本机 CARLA 0.9.10 的 OpenGL / Low 实采出现黑天空、缺失雾效果及部分过曝；相关失败尝试保留在 analysis 中，不作为合格 smoke。
旧引擎的 `-RenderOffScreen` 在 Vulkan 下出现空窗口句柄崩溃，因此脚本使用独立、带认证且禁用 TCP 的 Xvfb 虚拟显示，在其中运行普通窗口。不会连接或修改现有桌面。
Xvfb 从 Ubuntu 软件源下载并解包到本批次 `analysis/.../runtime_xvfb/`，未安装系统包；脚本自动启动并清理自己创建的 CARLA 和虚拟显示进程。
此修正只调整运行时渲染，不更改天气强度、Expert、ScenarioManager tick、传感器格式、雷达算法或 manifest 主体 schema。

参考：[CARLA 0.9.10 渲染说明](https://carla.readthedocs.io/en/0.9.10/adv_rendering_options/)；[CARLA 维护者的 OpenGL 雨雾问题记录](https://github.com/carla-simulator/carla/issues/3545)。

天气条件由 CARLA 实际渲染设置产生；雷达继续由原语义 LiDAR→C-Shenron 模型生成，本批次不额外引入雨雾传播衰减或天气噪声模型。


## 运行环境验收记录

本次检查发现既有 `lxl_cshenron` 环境缺失 NumPy 和部分运行依赖，已按固定版本补齐；依赖一致性检查、C-Shenron 完整模块导入及 GPU 0 上的小规模 CUDA FFT 均通过。版本和结果保存在批次 analysis 目录的 `radar_runtime.json`，用于修复的离线 wheel 保存在 `runtime_wheels/`。
烟测以 72 个原始 anchor 为进度目标；通用采集日志中按正式 10000 配额估算的 ETA 不适用于烟测。原始 episode 仍按现有路线运行和收尾，以保存真实 future 和场景证据。


## 烟测输入选择与复测来源

路口烟测使用 Town01 tiny route 68、seed 0、fog；这条官方路线在历史实采中已有多段真实交互证据，本批次仍重新采集和检查事件。首次 Town03 自动匹配路线虽有注释，却未实际触发，保留为失败尝试。
两个已完成的 normal/fog 与 S2/rain 原始 episode 在核对采集、标签、雷达源码指纹以及每个 job 输入完全一致后复用；仅重采更换路线的路口段。复用来源记录在对应 smoke attempt 的 `capture_reuse.json`，后续雷达、prepare 和 validate 全部重新执行。工作空间别名与真实路径已统一计算签名，并有符号链接路径一致性测试。
正式采集的路线配置、10000 配额和 8:1:1 划分未因烟测路线选择而改变。
