# Agent50 数据迁移与使用

当前发布数据：`analysis/grounding_agent50_v1/manifest.jsonl`。
配置：`configs/l40s_qwen25_3b_agent50_v1.json`。
完整验收报告：`analysis/grounding_agent50_v1/REPORT.md`。

## 数据与监督约定

- Radar 输入保持完整 Power、unfolded Doppler、原坐标轴和 mask。
- 检测背景监督及 FP 计分限于当前 Radar 安装位姿对应的 50 m BEV 扇区；正例仍复用采集器的 BBox/FOV 相交规则，边界相交目标不会因中心在域外被删除。
- `grounding_label_schema=agent50_v1`、`detection_class_schema=agent_v1`。所有保留检测正例为 `agent`，索引 0，padding -1。
- 用户追加选择：保留帧，移除行人和自行车检测正例，存入 `ignored_agents`；策略为 `pedestrian_cyclist_regions_v1`。自行车通过原类别与原始 CARLA type_id 识别，摩托车继续保留。原细分类别及 actor 元信息供审计使用。
- 忽略区域为被移除对象的 BBox 内部或中心周围 2 m，与现有匹配容差一致。训练额外忽略最近 anchor cell，避免小框在网格上没有忽略位置。保留正例匹配优先于忽略；评估在正例匹配完成后忽略该区域内未匹配预测，并保留原始预测数。
- Sensor Velocity 由原始四帧状态按 `v_sensor = v_ego + omega × r` 计算，再投影到当前 ego；使用完整安装偏移与姿态。Radar 数组不重算。
- Risk 五维定义、标签、risk_agents 来源和原 SHORT/LONG 标签不重算。过滤后失去所选关键目标的 9 个训练 SHORT 帧标记 `planner_supervision_available=false`；Grounding 保留这些帧，SFT 入口显式跳过并记录 ID，不补选其他目标。

## 版本兼容

新配置使用 objectness 预测 Agent 存在性，不建立冗余单类分类器；分类 loss 为零。旧配置缺少 schema 时继续使用原四类结构。跨 schema checkpoint 加载明确失败，禁止通过 strict=False 绕过。新数据必须与新配置配套；旧权重继续使用旧配置和旧 manifest。

## 复现迁移与验收

在项目 Python 环境中运行，输出必须是新目录：

```bash
python scripts/migrate_agent50.py \
  --source analysis/rebalanced_50m_domain/manifest.jsonl \
  --output-dir analysis/grounding_agent50_reproduction \
  --config configs/l40s_qwen25_3b_agent50_v1.json

python scripts/verify_agent50_dataset.py \
  --directory analysis/grounding_agent50_reproduction
```

迁移入口包含状态修复与行人/自行车忽略阶段。缺少原始数据、实例对应失败或发生不允许的标签变化时阻止发布。`filter_agent50_labels.py` 是已完成基础迁移的独立后处理入口，不需要对发布目录再次运行。

测试入口：

```bash
OMP_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_agent50.py tests/test_radar_vla_detection.py \
  tests/test_detection_annotation_domain.py tests/test_v3_collection_adapter.py \
  tests/test_diagnostic_detection_cache.py tests/test_visual_diagnosis.py \
  tests/test_radar_vla_planner.py tests/test_radar_vla_records.py
```

迁移、验收和上述测试不启动训练；真实样本验收只执行前向、loss 和反向，不执行优化器更新。
