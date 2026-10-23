# 本轮验证记录

## 诊断入口与数据协议

在现有 `lxl_radarvla` 环境、CPU单线程执行：

```bash
python scripts/diagnose_risk_rollout.py --self-test
python -m pytest -q tests/test_risk_rollout.py tests/test_local_risk.py tests/test_taskwise_supervision.py
```

结果：自检通过，50项测试通过。验证不完整Oracle排除、短时域覆盖单列、共同时间点配对、Local Risk mask与几何协议等。

真实数据核查：train+val共8319帧metadata；46帧分层几何重算；21帧实际Dataset→collate；全部通过。模型结果复现：三个Oracle分片共6个代表样本，原始预测逐token一致、Risk差值为0。

## 训练控制与回归

```bash
python -m pytest -q tests/test_risk_training_controls.py tests/test_risk_rollout.py tests/test_training_quality.py tests/test_radar_vla_pipeline.py
```

结果：31项通过（与上组部分重叠）。新增梯度测试在旧实现上先出现 `unexpected keyword argument 'detach_risk_condition'`，实现后通过。

随后补强冻结eval模式和detach数值不变的检查，专项7项再次通过：

```bash
python -m pytest -q tests/test_risk_training_controls.py
```

- 真实SFT优化步骤更新Planner，Grounder所有参数/缓冲区与初始化逐张量一致，并始终处于eval。
- detach前后eval前向loss数值逐位一致；只改变反传路径。
- detach后规划损失到risk_head梯度为0；物理监督到risk_head仍非0；规划损失到Radar encoder仍非0。
- 参数合法性、SFT阶段限制、互斥控制和恢复协议检查通过。
- `python -m radar_vla train --help`确认两个新CLI开关；相关脚本编译检查和修改行的`git diff --check`通过。

## 实验范围

Oracle完整rollout1407帧，低速几何回放1039帧。后续训练为3组同初始化、同预算1epoch对照；新候选只对全val Risk和首模式token做完整评价，轨迹使用训练期固定32帧审计。它不替代候选完整轨迹、闭环或实采雷达验证。

没有评价test指标，没有更改数据标签、物理词表、阈值或原始检查点。模型权重留在服务器的独立`model_runs/risk_controls_20261019/`目录。

## 累计风险一致性补丁与完整轨迹对照

```bash
python -m pytest -q tests/test_risk_state_projection.py tests/test_risk_training_controls.py tests/test_risk_rollout.py tests/test_radar_vla_planner.py tests/test_radar_vla_hf.py tests/test_radar_vla_pipeline.py tests/test_training_quality.py
```

81 passed in 23.33s

新增补丁测试在应用前因缺少API导入失败（projection_test_before_fix.log），应用后通过。包含独立SciPy参考、梯度检查、Tiny与真实HF前缀入口、旧默认行为、训练控制和本地微型Qwen回归；未下载模型。CLI帮助确认predict/evaluate均提供显式开关；相关修改的git diff --check通过。

完整1461帧SFT验证集：158帧输入改变后重新原生greedy生成完整轨迹，1303帧输入逐值不变并复用冻结原结果。代表帧验证原模型逐token复现及恒等投影输出不变，实际Qwen输入hook确认仅缩放一次。全部累计输入满足顺序约束，dmin/areq逐位不变；没有使用GT变换预测，没有评价test。
