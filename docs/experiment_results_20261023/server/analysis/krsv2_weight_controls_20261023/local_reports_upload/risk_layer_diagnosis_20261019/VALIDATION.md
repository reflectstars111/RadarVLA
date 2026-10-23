# 专项验证

诊断只新增 scripts/tests 下的工具，未修改神经网络、Dataset、标签定义或训练损失。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q tests/test_risk_layer_diagnosis.py
```

最终运行：**4 passed in 0.99s**，日志见 `tests.log`。早期三项检查为 **3 passed in 1.16s**，不累加。检查内容：

- 五维及单调项的逐帧分解值、反向梯度与 canonical FP32 Risk loss 一致。
- 缺失 areq 的 NaN 占位不会产生梯度；各维梯度只作用于对应原始输出。
- 旋转GT框归属正确，零梯度的余弦关系记作未知。
- 使用真实 Dataset 的 Tensor 返回类型检查 Doppler 归属及径向速度误差。
- 原始表征配对训练子集完整保留危险正例，负例抽样固定seed42且无重复。

首次真实小样本检查因诊断工具的 Tensor/NumPy 转换失败，已保存 `smoke/` 日志和修复前脚本；未开始模型训练。修复后 `smoke_verified/` 对4帧完成原缓存逐位核对和输入/梯度回放。后续新增最近目标运动探针及阶段进度记录，全量运行逐帧重新验证。

梯度诊断使用原 BF16 前向值，在 FP32 中分解物理损失，以排除低精度损失算术的干扰。统计对应原始batch-one物理监督的自然样本平均，不包含Planner和检测任务梯度，也不包含Adam优化器预条件效应。

原始回波检查使用GT当前框归属，属于特权诊断。位置、速度及未来GT未作为原Grounder前向输入。为减少共享存储I/O，原始表征配对探针训练子集为全部477个Pcol3正例加固定1000个负例，val仍为全部1461帧。完整6728帧的缓存特征探针及梯度统计单独报告。早期全量原始扫描已只中断本任务，日志与源码保存在attempts/full_raw_scan，不复用其未完成特征。

特征探针独立拟合，只在对应train上标准化、训练和选阈值；val仅评价。运动探针读取当前最近目标，不代表所有危险来源目标均可恢复。本轮延用既有离线雷达缓存，未重新证明其预处理的在线因果性；既有报告记录了部分Doppler解折叠产品的双侧时间依赖，本轮不据此宣称零延迟在线效果。

最终精确复现帧数、模型权重与缓冲区不变、源码/原检查点/manifest摘要不变、train/val episode隔离由 `completion_checks.json` 记录。没有使用test评价，没有修改默认模型，没有进行完整轨迹rollout。

检测头补充：`scripts/audit_detector_risk.py` 对同一99帧逐样本重放，要求Risk值与每个GT目标的检测匹配标志均与原完整验证报告完全一致；实际通过。GT正例anchor分配采用原排序及空间Hungarian定义，预测评价采用原2m一对一匹配、0.5置信度和0.3旋转NMS。检测loss和风险loss分别反传共享CNN及时序模块，全部权重/缓冲区保持不变。诊断脚本的解码/匹配遵循真实调用链，未另写仅模拟实现的单元测试。

原始8×8与CNN2×2探针的空间摘要不同，故另做raw2×2及非线性读取对照，结果保存 `pooling_controls.json`。该对照排除了直接把跨尺度AP差异归因于CNN信息丢失的解释。最近运动目标分层另保存 `motion_subset_probes.json`。

独立CPU预审只读相同缓存，原始99帧来源证据和完整训练集token探针均要求在主任务中再次精确复现；最终整理脚本检查一致性。共享风险参数梯度拆分的CPU重放与主CUDA重放分开报告，不将硬件数值差异混用为单一梯度结论。
