# 50m输入裁剪配对训练

2026-10-19启动。目的：检验只保留50m以内雷达输入，是否改善危险车辆检测和五维Risk估计。

## 已完成的准备

- 4项裁剪对齐/标签保持测试通过（1.55秒）。
- 真实样本CPU前向、反向测试通过，梯度有限。
- 输入由`[1,4,2,256,107]`变为`[1,4,2,109,107]`，不重采样。
- 相同随机初始化已在smoke中逐字节核对。
- 仅增加实验脚本与测试，没有修改生产模型、损失、Dataset或原始数据。

## 固定实验设置

两组：`full118`和`crop50`。完整Grounding训练6847帧、验证1472帧，各5个epoch；相同权重初始化、相同批次顺序、学习率0.0003、AdamW、有效batch=8、BF16、裁剪梯度范数5。两组均使用当前正确的50m背景监督掩码。裁剪仅改变模型输入，不删除边界目标，不重新计算标签或每目标雷达监督关联。

一张卡、一个进程，两组共享一次原始batch读取并依次计算。CPU计算线程1、数据加载worker=0、nice=19、ionice=idle。启动要求空闲显存≥32GiB且GPU利用率≤10%；运行时显存占用设12%上限，余量低于8GiB时自停本进程。

固定第5个epoch为主要比较，同时保存每个epoch的完整验证结果。关注全体检测召回、Pcol3危险来源车辆召回、运动误差及五维Risk指标。补充原36个Risk漏检样本、两组共同检出的危险车辆运动误差。当前阶段不实例化LLM，不进行SFT或rollout。

## 启动记录

服务器`whuserver`，进程`2723804`，启动时选择GPU 3，空闲45458MiB、利用率0%。GPU UUID：`GPU-25f5647b-e094-f2d5-37ea-71ed14b27c02`。

远端结果目录：`/home/luxinlan/data/lxl_data/RadarVLA/analysis/range_control_20261019`。

远端权重目录：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/range_control_20261019`。不下载权重至本机。

代码位于远端项目的`scripts/train_range_control.py`、`scripts/launch_range_control.py`、`scripts/summarize_range_control.py`；测试为`tests/test_range_control.py`。本目录`code_review/`保留对应审阅副本。

## 解释限制

裁剪同时改变归一化统计、池化布局及检测候选数量（3456→1512）；它是整个裁剪方案的对照，不能单独证明某种机制。单个配对种子的结果只作为后续多种子/组件拆解的依据。风险概率阈值召回不等于LLM模式召回。

新实验没有沿用旧模型权重；它与历史两卡实验的采样与归约细节不同，主要结论必须来自本轮两组之间的比较。
