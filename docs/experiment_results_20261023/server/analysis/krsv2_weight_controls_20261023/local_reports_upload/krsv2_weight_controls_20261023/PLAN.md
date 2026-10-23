# KRS-v2：A/B并行单变量权重对照

用户已授权：先启动A组，并在有可用GPU时同时启动B组。两组均从头训练20轮；基线C0已经完成。

| 组别 | 唯一算法变化 | 启动GPU | 运行占比 |
| --- | --- | ---: | --- |
| C0：已有基线 | 原系数 | 原GPU1，已结束 | 原设定 |
| A_dmin025 | 原生distance权重1→0.25 | GPU1 | 原生节奏 |
| B_areq3 | 原生brake权重1→3 | GPU4 | 每步后增加等比例休息，约50%运行占比 |

B启动时GPU4已有约22%利用率，但显存余量约43GiB。为给其他任务留计算余量，B使用50%step duty；这是墙钟时间的资源安排，不改变数据顺序、优化器更新次数或按更新计数的学习率调度，不代表驱动层GPU计算配额。

## 不变项

T-RODNet、seed42、batch4、梯度累积2、20轮、LR2e-4、前5轮恒定后按更新余弦衰减至0.1倍、BF16。标签为Pcol0.5/1/2秒、dmin/areq均2秒；Agent未来轨迹仍保留到3秒。两组使用相同manifest与训练/验证划分，不加载基线训练后权重，不改变采样器、Focal、归一化尺度、检测/运动损失、NMS或候选范围，不把A/B两个系数叠加。

## 源码与provenance

隔离worktree：
`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-risk-weight-controls-20261023`

分支：`codex/risk-weight-controls-20261023`；提交：`78364875130f5bad00a6ac9ee98c300ffe4d7515`。

相对基线的package源码仅修改pipeline.py：新增loss_weights配置校验、训练时传入原生loss、checkpoint重载恢复系数。模型结构和参数初始化代码不变；旧配置不增加新protocol字段，单位系数保持原loss/梯度行为。配置中的loss_weights随checkpoint保存，修改系数恢复训练会被拒绝；额外execution.json还锁定外部记录脚本。

原生系数调用：A使用`distance=0.25`，B使用`brake=3.0`。前三个冲突项仍按原有效元素一起归约，不将五个系数归一化为和1。

主源码与已完成基线均不修改，使用独立editable环境：
`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/krsv2_weight_controls_20261023/venv`

## 输出与监控

数据根目录下：

- `model_runs/krsv2_weight_controls_20261023/A_dmin025/train_seed42_20ep/`
- `model_runs/krsv2_weight_controls_20261023/B_areq3/train_seed42_20ep/`
- `analysis/krsv2_weight_controls_20261023/{A_dmin025,B_areq3}/`保存计划、PID、日志、原始/加权损失、加权后特征梯度与完整val结果。

各组独立6GiB显存上限，余量低于8GiB仅退出自身worker；CPU线程1、dataworkers0、nice19、ionice idle。不因其他用户进程存在而直接暂停或操作其进程。

`STATUS.json`是最后查询快照；`refresh_status.py`读取两组原生status与逐轮metrics刷新它。若仍在fingerprint阶段，不能把worker存活当成已更新参数。20轮未完成前不宣称对照胜负。

## 启动验收

- 外部loss/observer契约14项测试通过，包括单位系数、单项系数与记录梯度缩放、RNG和参数梯度一致性。
- 现有pipeline/KRS兼容回归19项通过。
- 原生checkpoint恢复、重载与错误系数拒绝等2项新增测试通过。
- A、B真实batch4前向、反向及验证记录均通过，覆盖危险、空目标及超过8目标；预检无优化器更新，峰值显存约2.16GiB。
- B的节奏控制检查通过：batch数与顺序保持，不丢弃或重复样本。节奏控制不进入loss计算。

## 比较规则

固定epoch20为主要比较；epoch12、16是预先固定的次要观察点，保留完整曲线。比较各窗口AP/Brier、危险检测召回、误报、危险dmin误差、areq>0/areq=0子集误差及运动质量。

各组val total loss使用各自系数，不能直接以总loss数值大小宣布哪组更好。ADE/FDE须同时报告匹配数量并检查共同匹配目标，不能因漏检难目标而假装轨迹改善。测试集不参与选权重；三seed和100轮训练仍不自动启动。单seed CUDA反向非确定性和短时正例episode稀缺的限制继续保留。
