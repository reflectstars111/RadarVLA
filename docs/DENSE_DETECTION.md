# Encoder 多类别目标检测

2026-10-13 实现。Stage 1 在共享雷达 CNN 上增加单阶段稠密检测分支，同时预测每个目标的类别、位置、框、速度和未来轨迹。**检测与目标未来轨迹均由 Grounder 完成，训练和推理不加载大模型。**这里采用类似 YOLO 的空间候选、存在性、框回归和 NMS 流程，没有加载图像 YOLO 权重。代码入口是 `model.py`、`detection.py`、`losses.py`。

## 数量与坐标

`B` 始终表示 batch size。实际目标总数是每帧的 `num_agents`，其中车辆数是 `num_vehicles`。二者都来自预测候选的置信度筛选与去重，推理不读取真实 agents，也不需要事先提供数量。

```text
雷达历史 [B,T,2,256,107]
  → 共享 CNN，stride=4
  → 最新帧空间特征 [B,64,64,27]
    + 标定后的当前自车坐标编码
    + 时序 radar tokens 均值与 ego 特征
  → 每格 2 个候选，K=64×27×2=3456
  → objectness / 类别 / 中心与框 / 速度与未来轨迹
  → 置信度筛选 → 按类别旋转框 NMS → 可变数量 detections
```

检测分支保留 CNN 的空间分辨率，分叉点在原有 `8×8` 池化之前。候选中心由标定 RA 单元的当前自车 xy 加可学习偏移得到，包含传感器旋转和平移。输出位置和未来轨迹均为当前自车坐标系，x 向前、y 向左，单位米。二维框为 length/width/yaw；这是 BEV 框，没有高度预测。速度 vx/vy 是绝对地面速度在当前自车坐标系中的表达，vr 是相对传感器径向速度。

## 类别标注

每个当前 agent 应显式提供 `category`，原始帧导入会将类别保留到 prepared manifest。接受以下规范类与别名：

| 规范类 | 可接受的标注名称 |
|---|---|
| vehicle | vehicle、car、truck、bus、van、motorcycle |
| pedestrian | pedestrian、person |
| cyclist | cyclist、bicycle、bicyclist |
| other | other |
| unknown | unknown 或缺失类别 |

也接受 `class_name`、`class`、`type` 字段；多个字段必须一致。类别字符串忽略首尾空白和大小写。数据集自定义标签或整数 ID 必须先显式映射，未知拼写会报错。当前约定 motorcycle 归 vehicle，自行车归 cyclist；`num_vehicles` 仅统计 vehicle。

分类头的固定四类顺序为 vehicle/pedestrian/cyclist/other。`other` 是有标注的其他目标，`unknown` 表示缺少类别监督或推理分类置信度不足，不是第五个训练类别。缺失类别时仍可监督位置、框、存在性和轨迹，但分类损失屏蔽；不会把无类别目标默认当车辆。

正式两个配置均启用 `data.require_detection_classes=true`，训练/评价预检遇到当前目标缺失类别就报错。需要混合部分标注数据时可显式设为 false，必须报告类别监督覆盖率。空的已标注场景训练负例，完全无当前目标标注的场景屏蔽对象监督。

## 监督和容量

根据当前标定网格中心与真实目标位置做一对一空间 Hungarian 分配。每格默认两个候选；同格多个目标可以分配到不同候选，更多目标可分配到相邻网格。类别和几何的稳定排序处理等距离分配。正候选接受分类交叉熵、中心/速度回归、对数框尺寸回归、周期朝向损失，以及该目标自身的未来位置/速度与物理约束；存在性使用分别归一化正负样本的 focal BCE。

每个候选独立输出未来轨迹：`center + velocity × future_times_s + learned_residual`。NMS 按同一个候选索引提取状态和轨迹，不会借用另一个对象槽的轨迹。未来缺失点继续由 mask 屏蔽。对象损失记录为 `detection_*`，与风险损失共同组成 grounding 总损失；每项只计算一次。Stage 2 继续联合更新 Grounder，因此检测分支也继续训练。

新的检测模型不再建立原有固定对象查询头。训练数据保留全部当前标注，按每个 batch 的实际最大目标数补齐张量并使用 mask；空场景可以有零个监督槽。`model.max_agents` 只用于 `use_detection=false` 的旧查询架构，不限制新的检测与轨迹预测。默认不设 8/64 等结果数截断，所有通过存在性筛选和 NMS 的目标都保留其轨迹。

稠密头仍有空间候选容量：`K=ceil(R/4)×ceil(A/4)×detection_slots_per_cell`，默认输入下为 3456，并非无限容量。训练目标超过 K 会明确报错并要求增大每格候选数，不能静默丢失监督。可选的输出/候选数上限默认 null；如部署方主动设置，截断会显式标记。

`planner.max_agents` 只限制第二阶段 Qwen 的 LONG/SHORT 序列，默认仍为 8，与第一阶段解耦。只有训练/评价第二阶段的语言规划输出时才检查该容量；更长序列还需满足上下文长度。第一阶段可直接在完整 labeled manifest 上训练，不必通过面向 Qwen 监督条件的 `build-cohort`。

## 轨迹头的具体实现

输入是共享 CNN 的最新帧空间特征，加标定坐标编码、时序 tokens 均值和自车特征，经过 `3×3 Conv → GroupNorm → GELU`。运动头使用 `1×1 Conv`，每个候选输出 `3+4H` 个数：`[vr,vx,vy]`、H 个二维位置残差、H 个二维速度残差。H 默认 6，所以每个候选输出 27 个运动量。

未来位置为 `p(t)=p0+v0*t+Δp(t)`，未来速度为 `v(t)=v0+Δv(t)`。p0 来自同候选的检测框中心，v0 来自运动头的当前速度。网络同时预测全部时刻，不是逐点自回归，也没有调用 Qwen。匀速部分提供运动基线，残差允许表示转弯、加减速。

一对一匹配后的目标提供 future_xy 监督，缺失时刻由 mask 屏蔽；另用相邻位置与速度的梯形积分残差、加速度与 jerk 平滑约束运动一致性。当前每个目标输出一条确定轨迹，没有多模态分支、轨迹概率或不确定性估计。默认查询 `[0.5,1,1.5,2,2.5,3]s`，实际时刻来自 manifest。

## 推理字段与配置

现有 `predict` 命令自动输出以下新增字段：

| 字段 | 含义 |
|---|---|
| detection_status | 新模型为 available；无检测头旧模型为 unavailable_legacy |
| num_agents | NMS 后所有类别目标总数，含 unknown |
| num_vehicles | NMS 后 vehicle 数量 |
| class_counts | vehicle、pedestrian、cyclist、other、unknown 分别计数 |
| detections | 可变长度列表，长度与 num_agents 相等 |
| detection_truncated | 候选预筛或最终输出达到容量而有目标被截断 |
| detection_thresholds | 本次使用的存在性、分类和 NMS 阈值 |

每个 detection 包含 `local_id`、`category`、`confidence`（存在性）、`class_confidence`、`class_probabilities`、`position_m`、`size_m`、`heading_rad`、`radial_velocity_mps`、`velocity_mps`、`future_xy`、`future_velocity_mps` 和 `future_times_s`。`local_id` 只是本次结果中的局部编号，不是跨帧跟踪 ID。

| model 配置 | 默认值 | 作用 |
|---|---:|---|
| use_detection | true | 启用检测头 |
| detection_slots_per_cell | 2 | 每个空间网格候选数 |
| detection_confidence | 0.5 | 存在性 sigmoid 阈值 |
| detection_class_confidence | 0.5 | 最大类别概率阈值；不足则 unknown |
| detection_nms_iou | 0.3 | 同类别旋转 BEV 框去重阈值 |
| detection_max_candidates | null | 默认处理全部过阈值候选；可显式设置部署上限 |
| detection_max_objects | null | 默认保留全部 NMS 结果；可显式设置部署上限 |
| detection_match_distance_m | 2.0 | 评价中心匹配距离门限，单位米 |

不同已知类别之间不互相 NMS；unknown 可以与已知类别框去重。默认阈值是实现初值，应在验证集上调校。截断结果的数量不是完整场景数量，使用方应检查 `detection_truncated`。

新模型 JSONL 顶层 `object_probabilities`、`agent_state`、`agent_future` 与 `detections` 来自同一批筛选结果，数量与顺序一一对应；`agent_state` 为 `[x,y,vr,vx,vy]`。模型 forward 内部同名张量是全部 K 个候选，用于训练和原有原始匹配指标。无检测头的旧模型才保留旧固定查询输出。

`plan` 内的 agent 来自第二阶段 Qwen 生成，与检测结果是独立预测，不能按下标混用。Qwen 当前继续接收 radar tokens/risk/instruction，尚未把检测对象序列作为额外 prefix。只有 Stage 1 checkpoint 时 `plan=null`，每个检测目标的 `future_xy` 已经由 Grounder 给出。

## 评价与检查点

`evaluate` 的 `detection` 项对筛选后的预测与真实中心做带距离门限的一对一匹配，报告 TP/FP/FN、precision/recall/F1、目标数量 MAE/完全正确率、车辆数量 MAE/完全正确率、各类别指标、匹配位置误差以及匹配目标未来 ADE/FDE。车辆数量指标只在全部当前目标类别已知的帧计算，另报分母；完全无标注帧不参与检测指标。轨迹误差只覆盖匹配成功的对象，必须结合漏检率及覆盖数查看。这些是固定阈值中心匹配指标，不是 IoU AP/mAP。

检测头需要带类别和几何标签的数据训练。旧 v2 checkpoint 缺少检测参数时保持旧架构，数量和 detections 为 null，不会挂接随机权重后给出数量。新检测训练应使用新输出目录从 Stage 1 开始，再以该 Stage 1 checkpoint 初始化 Stage 2；旧训练的配置/源码指纹与新结构不同，不能直接 resume。

自动测试覆盖多类别标签、缺失类别 mask、全部检测头梯度、同格目标、旋转 NMS、坐标变换、变长及全空 batch、12 个目标全部监督、80 个检测结果无默认截断、显式截断标记、旧模型兼容、双阶段训练和无标注推理。拥挤场景 Stage 1 的训练/评价/预测测试禁止创建 planner，以验证不依赖大模型。这验证软件行为；真实道路上的数量准确率和定位误差仍需实际训练及独立测试集评价。
