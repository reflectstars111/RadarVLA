# 统一训练分配与推理解码的有效候选范围

已在独立分支 `codex/candidate-domain-20261021` 实现并提交：`3b90c785be32adb5350f1be959a80036d2be6107`。原项目目录、原骨干分支和已有四组优化训练均不修改。

服务器源码：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-candidate-domain-20261021`

独立Python：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/candidate_contract_20261021/venv/bin/python`

## 行为改变

- 新策略 `model.detection_candidate_policy=supervised_sector_v1` 使用 checkpoint 固定的50m、±53°扇区，根据最新雷达标定姿态和anchor中心，以FP32生成同一份有效候选mask。模型前向不读取GT框或标注可用性来生成mask。
- 训练Hungarian分配只在有效anchor中进行；所有GT继续保留，包括中心在范围外但框跨界的目标，通过域内anchor的中心偏移学习完整框和运动。有效anchor不足会明确报错，不丢目标、不增加8目标截断。
- 推理解码在置信度筛选、候选数量上限和NMS之前应用同一mask，避免域外无监督高分候选压制域内预测。
- 原始GT匹配的物理指标也使用有效候选；阈值/NMS后的检测计分规则保持原样。运动对照优先看解码后共同检出目标，避免把原始匹配口径改变误认为模型提升。
- 雷达输入仍保留约118m；检测头仍输出3456个原始候选，本次真实样本中有效候选为1458个。Risk结构、损失权重、Doppler预处理和数据标签不修改。

## 兼容与回退

缺省策略仍为 `legacy_v1`，旧配置和checkpoint继续使用原行为。新checkpoint保存 `_candidate_sector`；跨策略、跨范围加载明确失败，即使 `strict=False` 也不允许静默混用。新策略从头训练，不能只给旧权重换一个配置。

新增CNN、HG、T-RODNet三个独立配置，文件名均以 `_candidate_sector.json` 结尾；原配置不覆盖。回退只需使用原骨干worktree、原Python和原配置/权重，不需要reset或清理原仓库。

## 验证

- 108项相关测试通过：候选契约、检测监督域、稠密检测头、模型、训练流水线、物理指标及两种时空骨干。
- 先复现旧行为失败：边界GT仍被分配给域外anchor；再验证新行为通过。覆盖域外高分候选在NMS/数量上限前被排除、空目标、零有效候选、超过8个目标、有效容量不足、错误标注覆盖范围、旋转安装位姿及checkpoint策略冲突。
- CNN/HG/T-RODNet分别在真实空场景、边界目标和12目标样本上完成前向、原生loss、反向；数值与梯度有限，域外候选的直接objectness梯度为0；Risk输出仍为5维。没有执行优化器更新。
- 三种骨干的已训练旧checkpoint，在旧/新环境中使用旧配置时，全部输出张量逐位一致。
- 三个独立配置均完成启动命令dry-run，没有启动dry-run优化器。

证据文件：`acceptance.json`、`implementation.json`、`implementation.patch`、`launch_dry_runs.json`。完整接口约定位于源码 `docs/CANDIDATE_SECTOR.md`。

## 新训练对照

已为HG、T-RODNet安排各一组新契约正式训练，与已完成的旧策略同骨干batch=4实验配对：同一moving-v1 manifest、seed42、lr=1e-4、真实batch4、累积2、BF16、5 epochs，从头初始化。固定第5轮为主要比较，保留所有轮次与最终完整验证诊断。

服务器任务目录：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/candidate_training_20261021`

服务器权重目录：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/candidate_training_20261021`

资源策略沿用用户最新要求：优先空闲卡并行、HG分配器预算4 GiB/T-RODNet 6 GiB、启动多留2 GiB、空闲低于2 GiB仅退出本worker，CPU线程1、data workers=0、nice19、ionice idle；其他用户计算进程出现本身不触发暂停。

新任务 `status.json` 记录PID/卡号；`REPORT.md` 与 `comparison.json` 自动汇总同预算旧/新策略的检测、Risk与共同检出运动误差。训练效果尚未得出；不能把源码修正或候选过滤本身当成模型精度已经提高，更不能据此宣称Planner或闭环驾驶改善。
