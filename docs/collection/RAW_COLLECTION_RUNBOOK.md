# v5 原始采集与 K-Radar 短测 runbook

## 1 万帧批次入口（2026-10-14）

批次配置为 [`collection/configs/collection_10k.yaml`](../../collection/configs/collection_10k.yaml)，入口为 `collection/collector/batch.py`。脚本自动启动和关闭 CARLA；**不用手动运行 UE4 的 `.sh`**。它执行官方路线采集、future anchor 选择、C-Shenron K-Radar 产品、进度条/ETA 和结果索引，不执行训练 loader 转换。

本次正式批次跳过 smoke 后，先采集 9,000 个 anchor，却因清单中的 Town06 地图未安装而在测试集作业反复产生 0 帧失败；旧批次在雷达启动前已停止。原始数据和失败记录全部保留。CARLA 地图预检现要求清单中的 `.umap` 确实存在。测试集改用已有 Town04 官方 XML 的 route 100–199；训练集使用 route 0–99，两个 split 不共享路线。

恢复阶段独立运行，不再等待 10,000 帧才启动雷达：

- `lxl_radarvla3`：对已选 9,000 帧生成雷达，输出日志 `/home/luxinlan/data/lxl_data/RadarVLA/radar_existing_console.log`。
- `lxl_radarvla_collect`：补齐 Town04 测试集配额，输出日志 `/home/luxinlan/data/lxl_data/RadarVLA/collection_remaining_console.log`。
- `lxl_radarvla_finish`：等待前两阶段成功，然后对新增帧生成雷达并核对最终索引，输出日志 `/home/luxinlan/data/lxl_data/RadarVLA/finish_full_console.log`。

查看状态：

```bash
screen -ls
tail -f /home/luxinlan/data/lxl_data/RadarVLA/radar_existing_console.log
tail -f /home/luxinlan/data/lxl_data/RadarVLA/collection_remaining_console.log
tail -f /home/luxinlan/data/lxl_data/RadarVLA/finish_full_console.log
```

`full/collection_remaining_status.json`、`full/radar_existing_status.json` 和 `full/status.json` 分别记录补采、雷达以及最终原始产品状态。仅在最终 `status` 为 `RAW_RADAR_COMPLETE` 且 `accepted`、`radar_products` 都为 10,000 时，才算原始采集链路完成。`training_ready` 仍为 `UNTESTED`，因为本轮跳过了 smoke 和 loader 验收。已产生的失败作业原样留在各自目录。**不用手动运行 UE4 的 `.sh`**；采集脚本管理 CARLA 服务。

加速对照使用同一真实 episode 的 12 个目标帧：两卡 38.04 秒，四卡 28.04 秒，包含 halo 的 28 帧 DRA、folded RA 和 unfolded RA 逐数组完全一致；四个雷达窗口均通过产品校验。该短测仅证明此负载下约 1.36 倍吞吐改善。

天气变化只作用于 CARLA 场景；当前 C-Shenron 路径没有天气回波模型。稠密 Doppler 是候选估计，`reliable_mask` 才表示时序匹配可靠性；CARLA-assisted Oracle 只用于 smoke 审计。当前 loader 仍不能直接使用 v5 的逐点道路宽度和部分有效 mask。

本 runbook 验收 **CARLA → v5 原始 episode → C-Shenron K-Radar → 稠密 Doppler 候选 → 原始产品校验**。它不执行训练导入。旧 `data_pipeline.py` 入口已经移除，当前采集入口是 `collection/collector/run.py`。原始字段定义见 [DATA_COLLECTION_SPEC.md](DATA_COLLECTION_SPEC.md)。

以下命令在 `/home/metaiot_guest/Workspace/Xinlan_Lu` 运行；一次尝试只使用一个全新输出目录。CARLA 使用 Python 3.7，雷达及报告使用 C-Shenron Python 3.10。若在其他机器运行，先调整 `ROOT`、`CARLA_PY` 和 `RADAR_PY`。

```bash
cd /home/metaiot_guest/Workspace/Xinlan_Lu
ROOT="$PWD"
CARLA_PY=/home/luxinlan/.conda/envs/lxl_carla/bin/python
RADAR_PY=/home/luxinlan/.conda/envs/lxl_cshenron/bin/python
ROUTES="$ROOT/Carla/C-Shenron/leaderboard/data/training_routes_cvpr2021/routes_town02_tiny.xml"
SCENARIOS="$ROOT/Carla/C-Shenron/leaderboard/data/scenarios/no_scenarios.json"
OUT="/tmp/radarvla_v5_smoke_$(date +%Y%m%d_%H%M%S)"
EP="$OUT/episode"
RADAR="$OUT/radar"
export PYTHONPATH="$ROOT/Carla/C-Shenron/team_code:$ROOT/RadarVLA/collection/collector"
export MPLCONFIGDIR="$OUT/matplotlib"
export XDG_RUNTIME_DIR="$OUT/xdg"
export XDG_CACHE_HOME="$OUT/cache"
test ! -e "$EP" && test ! -e "$RADAR"
mkdir -p "$OUT" "$MPLCONFIGDIR" "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR"
```

## 1. 输入预检与 CARLA 服务

```bash
"$CARLA_PY" RadarVLA/collection/collector/run.py \
  --routes "$ROUTES" --route-id 0 --scenarios "$SCENARIOS" \
  --output "$EP" --carla-python "$CARLA_PY" --carla-root "$ROOT/Carla" \
  --host 127.0.0.1 --port 15184 --tm-port 25184 --check-inputs

# 无现有 15184 服务时，在独立终端启动；保留 server.log 和 PID。
"$ROOT/Carla/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" CarlaUE4 \
  -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=15184 \
  > "$OUT/server.log" 2>&1 &
echo $! > "$OUT/server.pid"
ss -ltn '( sport = :15184 )'
PYTHONPATH="$ROOT/Carla/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" \
  "$CARLA_PY" -c 'import carla; c=carla.Client("127.0.0.1",15184); c.set_timeout(5); print(c.get_world().get_map().name)'
```

若服务不能启动，检查 `server.log` 并停止；**不启动本轮采集，也不自动换路线或端口重试**。

## 2. 一次真实采集与 v5 校验

```bash
"$CARLA_PY" RadarVLA/collection/collector/run.py \
  --routes "$ROUTES" --route-id 0 --scenarios "$SCENARIOS" \
  --output "$EP" --carla-python "$CARLA_PY" --carla-root "$ROOT/Carla" \
  --host 127.0.0.1 --port 15184 --tm-port 25184 \
  > "$OUT/collect.log" 2>&1

"$CARLA_PY" RadarVLA/collection/collector/episode.py \
  "$EP" --require-complete | tee "$OUT/episode_validation.json"
```

episode 校验覆盖同步 CARLA frame ID、相机/LiDAR 文件、官方 RoadOption、地图来源和 mask、agent ID 稳定性，以及 future 30 帧引用。`status` 非 `complete` 就停止。没有可见 agent 时，agent future 项为 **UNTESTED**，不能写成 PASS。

## 3. 选择行驶窗口并生成雷达

```bash
# 只选真实连续帧，每帧 Ego 速度 ≥1 m/s 且有完整 30 帧 Ego future。
# 优先选择含可见运动 agent 的窗口；不复制或改写 episode。
START=$("$RADAR_PY" RadarVLA/collection/collector/smoke_check.py \
  select-window "$EP" --count 7)
echo "source_start_index=$START" | tee "$OUT/window.txt"

"$RADAR_PY" -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  "$EP" --output "$RADAR" --start-index "$START" --limit 7 --check-inputs
"$RADAR_PY" -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  "$EP" --output "$RADAR" --start-index "$START" --limit 7 \
  | tee "$OUT/radar_process.json"
"$RADAR_PY" -m e2e_agent_sem_lidar2shenron_package.carla_kradar_episode \
  "$EP" --output "$RADAR" --validate | tee "$OUT/radar_validation.json"
```

`radar_meta.json` 保存 `source_start_index`、逐帧 `source_index` 和 CARLA frame ID。产品校验逐帧核对原始窗口，DRA `64×256×107`、RA `256×107`、三维 Hann、模糊阶数、五类来源、可靠性 mask 和折回误差。

## 4. 预览、报告与退出

```bash
"$RADAR_PY" Carla/AAA_tools/kradar_carla_viewer.py \
  "$EP" "$RADAR" --output "$OUT/preview"
"$RADAR_PY" RadarVLA/collection/collector/smoke_check.py \
  report "$EP" "$RADAR" | tee "$OUT/acceptance.json"

# 仅停止本次由 server.pid 启动的服务；若使用已有服务则不要执行。
kill "$(cat "$OUT/server.pid")"
```

`acceptance.json` 分开报告 `technical_chain`、`agent_future_30` 和 `dynamic_speed_supervision`。速度质量只有经过动态目标的独立支持/误差核对才可判 PASS；存在有限稠密候选值或全局可靠格不构成动态目标速度真值。本 runbook 的简单检查在可靠格存在但未与动态目标关联时报告 **UNTESTED**。

## 离线回归与训练边界

```bash
PYTHONPATH="$ROOT/Carla/C-Shenron/team_code:$ROOT/RadarVLA/collection/collector:$PYTHONPATH" "$RADAR_PY" -m unittest discover \
  -s RadarVLA/collection/tests -p 'test_*.py' -q
```

新 v5 地图提供逐点宽度和部分有效 mask；当前训练 loader 要求单个宽度和 16 点全有效。本流程不压缩或填造道路标签，不生成 `radar_frame_v2` 训练记录，也不宣称训练数据已经就绪。
