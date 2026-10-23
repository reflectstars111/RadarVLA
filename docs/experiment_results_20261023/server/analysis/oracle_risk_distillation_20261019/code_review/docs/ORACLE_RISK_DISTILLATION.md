# Oracle 模式蒸馏：风险头受控训练

目标是让**预测Risk驱动固定Planner时的模式判断**接近GT Risk条件，而非把未来真值接入推理。原5epoch检查点为固定起点；GT风险的模式效果不代表Oracle轨迹更好，因此只蒸馏BOS后的SHORT/LONG分布。

两组均冻结Radar/Ego编码器、检测器及Planner所有参数（包括LoRA与连续前缀映射），只训练`risk_query`、`risk_attention`、`risk_head`。缓存冻结的8个Radar token和1个共享Ego token，调用Grounder自身的`risk_from_context`，不另建网络。逐帧确认缓存路径与完整传感器路径的Risk输出逐位相同。

| 设置 | physical_only | physical_plus_kl |
| --- | --- | --- |
| 初始化 | 原SFT 5epoch best_planning | 相同 |
| 物理风险损失 | 原loss_grounding的风险部分 | 相同 |
| 模式KL权重 | 0 | 1 |
| 训练 | 3epoch，batch8，lr3e-4，AdamW decay0.01 | 相同 |
| 温度 / seed | 1 / 42 | 相同 |
| 正例权重、阈值、保序后处理 | 不改变，projection=none | 相同 |

KL方向为`KL(stopgrad(Oracle)||Student)`，只在五维Risk标签全部有效的train行计算。Student仅接收缓存的Radar/Ego特征、指令及预测Risk；GT标签仅是监督与教师输入。val只用于报告，test不加载。固定最终epoch，不按val选择epoch或调温度/权重；保留每轮结果。

冻结Planner不能用`no_grad`包住Student前向，否则风险头也得不到梯度。Student仍正常反传；教师分布预先在`no_grad`下生成。专项测试用真实微型Qwen验证这一路径，并确认Planner权重完全不变。

脚本入口为`scripts/distill_risk_modes.py`。从项目根目录、既有`lxl_radarvla`环境执行；启动前显式检查空闲GPU。`--output`保存可审阅结果，`--model-output`保存特征缓存和检查点，不能将后者提交Git或同步到本地小磁盘。

```bash
python scripts/distill_risk_modes.py \
  --checkpoint <original_sft/best_planning.pt> \
  --manifest <manifest.jsonl> --baseline <e5_complete_validation> \
  --output <analysis/oracle_mode_experiment> \
  --model-output <model_runs/oracle_mode_experiment> \
  --epochs 3 --batch-size 8 --lr 0.0003 --seed 42 \
  --temperature 1 --mode-weight 1
```

这份脚本针对当前Local/Ego-context检查点。缓存仅处理train/val的SFT可用样本；val中原始Risk和模式与冻结原报告逐帧核对，Oracle模式混淆矩阵也与前轮报告核对。已有缓存按源码、manifest、检查点摘要及样本顺序检查后复用。manifest会重新校验；本轮不重复扫描所有历史原始传感器文件的摘要，两组比较使用同一份实际缓存。

主指标为完整Risk子集的SHORT召回/误报、与Oracle的分层模式一致率和KL，同时报告全部SFT val及各维AP/Brier、距离误差与正制动量误差。总体一致率容易被大量LONG样本抬高，不能单独用作成功标准。模式接近但物理量明显退化也不能称为成功。

每组训练结束后对冻结的Planner/非风险Grounder所有参数及缓冲区计算完整摘要，要求前后完全相同。输出`refined_inference.pt`保留原Planner适配器状态，可由现有`load_models/predict`加载；它是推理专用检查点，风险头训练轮数记录在`risk_refinement`，不能当作原SFT训练的resume文件。

本轮不自动替换原模型。完整自由轨迹、闭环效果与模式评价是不同层次，报告必须说明实际完成的范围。

### Pcol3教师覆盖对照

主实验的完整五维教师要求全部风险标签有效。训练集477个Pcol3正例中，191个缺少其他维度，仍参与物理监督，却没有完整教师的模式KL。补充脚本 `scripts/refine_risk_pcol3_teacher.py` 仅将固定原模型风险向量中的Pcol3替换成有效GT，其余四维维持原预测，让全部Pcol3有效样本参与KL；未知制动量不补作真实零值。Student输入及物理loss保持不变，固定原模型起点、训练顺序、3epoch、batch8、lr0.0003、seed42、KL权重1和温度1。

这是混合风险教师，不是完整GT风险。与主KL组相比同时改变教师内容和有效样本覆盖，不能单独归因于覆盖率。单独记录教师质量，再在相同1407帧完整标签子集与全部1461帧验证集评价学生；不把验证集当训练数据，也不自动提升检查点。全模型冻结摘要及导出Planner状态保真检查继续执行。
