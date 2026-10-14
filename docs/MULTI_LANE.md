# LONG 多车道感知与结构化输出

## 感知与规划

共享雷达 encoder 的空间特征接 `DenseLaneDetector`。每个空间网格单元产生一个车道候选，预测中心线、左边界和右边界的二维采样点，以及存在概率、边界存在概率和左右相邻关系特征。输入仅为雷达观测、标定几何及自车状态；地图只作训练标签。

`lane_samples` 默认 16，表示每条折线的采样点数，不是车道数量。训练保留地图中全部标注车道，包括相邻、对向和横向车道，不按 `route_lane_id` 筛掉其他车道。可变数量在批次内补齐，使用掩码。若标注数超出空间候选数会显式报错，不静默截断。

推理按存在概率筛选，再按中心线几何去重；没有固定输出条数上限，候选容量由输入空间网格决定。相邻关系只在保留候选间计算，要求两侧预测与几何方位一致，避免自引用、悬空引用及相互矛盾的单侧关系。当前支持每条车道的一个左邻居和一个右邻居，不表达前继/后继、合流或分叉拓扑。

学习到的车道几何经过投影，与车道特征一起按存在概率汇聚，回馈给规划器使用的 Radar tokens。语言词表不变；`lanes[]` 来自感知分支，`road` 仍是语言规划器生成的行驶走廊。全部车道几何无需经过自回归 token 生成。SHORT 的输出字段保持不变；启用车道分支仍会增加共享感知开销，此开销纳入部署与 RL 的感知计时。

## 地图标签

在现有 `map.lane_centerlines` 中为每条车道关联左右边界和相邻车道 ID。例如两条同向车道：

```json
{
  "lane_coverage_valid": true,
  "lane_centerlines": [
    {"id": "ego", "points": [[0, 0, 0], [50, 0, 0]],
     "left_boundary_id": "shared", "right_boundary_id": "outer_right",
     "left_neighbor_id": "left", "right_neighbor_id": null},
    {"id": "left", "points": [[0, 3.6, 0], [50, 3.6, 0]],
     "left_boundary_id": "outer_left", "right_boundary_id": "shared",
     "left_neighbor_id": null, "right_neighbor_id": "ego"}
  ],
  "lane_boundaries": [
    {"id": "outer_right", "points": [[0, -1.8, 0], [50, -1.8, 0]]},
    {"id": "shared", "points": [[0, 1.8, 0], [50, 1.8, 0]]},
    {"id": "outer_left", "points": [[0, 5.4, 0], [50, 5.4, 0]]}
  ],
  "route_lane_id": "ego"
}
```

坐标最终统一到当前自车坐标系，单位米。中心线点序表示车道行驶方向；左右相对于该方向，因此对向车道的左右不等同于自车的左右。

- 缺少 `map` 或 `lane_centerlines` 表示车道监督未知；明确的 `lane_centerlines: []` 表示已观测为空。
- 默认认为列表覆盖该局部标注范围。部分地图不能代表完整覆盖时，设置 `lane_coverage_valid: false`，屏蔽该样本的车道损失；不要将遗漏车道作为负样本。
- 缺少边界或邻接字段表示未知；边界/邻接字段为 `null` 表示明确不存在。引用未提供的边界或局部范围外的邻居不作为不存在的监督。
- 对满足方向与重叠条件的共享边界，可推导相邻正样本；不能仅因没有共享边界就推导负样本。
- 推理数据集 `supervision=False` 清空车道监督，模型前向不读取地图标签。

## 输出

LONG 的 `plan.schema` 为 `radar_vla_structured_v3`。保留已有 `road`、目标状态及轨迹、自车轨迹等字段，增加：

| 字段 | 含义 |
|---|---|
| `lanes` | 本次感知保留的车道实例列表 |
| `lane_perception.available` | 检查点是否含经过车道监督训练的分支；不表示精度合格 |
| `lane_perception.num_lanes` | 实际输出车道数量 |
| `lane_perception.source` | `radar_encoder` |
| `lane_perception.schema` | `radar_vla_lanes_v1` |
| `lane_perception.coordinate_frame` / `distance_unit` | `current_ego` / `m` |

每个 `lanes[i]` 包含：

| 字段 | 含义 |
|---|---|
| `lane_id` | 本次输出内的局部编号，不是跨帧追踪 ID 或地图 ID |
| `confidence` | 车道存在置信度 |
| `centerline` | 中心线 `[x,y]` 采样点列表 |
| `left_boundary` / `right_boundary` | 边界采样点列表；未可靠识别时为 `null` |
| `left_boundary_confidence` / `right_boundary_confidence` | 边界存在置信度；该侧未经边界监督时为 `null` |
| `left_neighbor_id` / `right_neighbor_id` | 同一 `lanes[]` 中的相邻车道编号，无确定引用时为 `null` |
| `left_neighbor_status` / `right_neighbor_status` | `linked` 有相邻引用；`none` 预测不存在；`unknown` 未确定 |
| `left_neighbor_confidence` / `right_neighbor_confidence` | 已连接关系或无邻居预测的置信度；该侧未经邻接监督时为 `null` |

必须结合 `neighbor_status` 解读空引用，不能将 `unknown` 当成可安全变道的证据。`lanes: []` 且 `available: false` 表示功能不可用；`available: true` 则表示该次没有达到阈值的车道。原 `physically_complete` 仍衡量语言规划字段的完整性，与车道检测召回率无关。LONG token 生成失败时也保留已得到的车道感知。

边界与邻接关系各自按左右侧记录是否经过监督。只训练中心线不会启用随机初始化的边界或邻接输出。

## 训练与兼容

两个正式配置默认 `use_lanes: true`。Stage 1 联合训练目标检测、风险和车道分支，车道损失包括匹配后的几何回归、存在性 focal、边界存在性和有标注的左右邻接关系；Stage 2 继续联合训练并使语言规划使用带车道信息的 Radar tokens。可配置 `lane_confidence`、`lane_boundary_confidence`、`lane_adjacency_confidence` 和 `lane_nms_distance_m`。

旧检查点继续按原架构加载，返回车道感知不可用，不会挂接随机车道头伪造结果。要启用该功能，需要包含车道标注的 Stage 1 训练以及匹配配置的 Stage 2 训练；已有检查点不能因更新代码自动获得车道识别能力。未含车道监督的新训练也不会启用车道输出。

自动测试验证了数据、梯度、结构化输出、旧权重加载和计时链路。它们不代表真实雷达上的车道检测精度，后续仍需用实际标注数据训练与评估。

## 单车道与多车道分支对比

- `experiment/single-lane`：保留提交 `2171036` 的原实现，LONG 只有生成的单走廊 `road`，包含此前的目标检测与响应时间 RL 修复。
- `experiment/multi-lane`：在同一基线上增加本页描述的感知分支及 `lanes[]`。

在唯一工作目录 `/ssd1/code/yqq/radar_vla` 内，通过 `git switch experiment/single-lane` 或 `git switch experiment/multi-lane` 切换，继续使用本目录 `.conda` 的 editable 安装，无需复制源码。切换前保存当前修改，停止使用该目录源码的训练进程，避免进程读取混合版本。

对比时使用相同的场景划分、原始数据、随机种子、训练预算和硬件；两组分别训练 Stage 1 与 Stage 2，使用不同输出目录及检查点。原单车道版本没有车道检测头，因此比较的是两套完整方法；多车道权重不能直接交给单车道版本加载。优先比较共同的自车轨迹误差、碰撞/安全间距、端到端响应时间及计算开销；多车道感知质量另外评估。共享感知开销必须计入两种推理模式的实测耗时。
