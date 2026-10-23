# RadarVLA 超参数优化：官方项目配方调研

调研日期：2026-10-21。只采用官方仓库配置、实际训练代码及 PyTorch 官方文档。配置不是论文最优结果的保证，也不是完整超参数搜索记录。此次没有启动新训练，没有修改服务器主代码目录。

## 1. 核验范围与版本

已通过 GitHub API 解析提交，并下载以下文件至本目录 `upstream/`。链接固定提交，避免分支后续更新改变证据。

| 项目 | 固定源码提交 | 核验内容 |
| --- | --- | --- |
| T-RODNet | `39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9` | 专用配置、训练脚本 |
| RODNet | `44fd61fa7b433b920469bd8455a54d5676e57fd8` | HG1配置、训练脚本 |
| OpenPCDet | `233f849829b6ac19afb8af8837a0246890908755` | ONCE PointPillar配置、优化器与OneCycle实现 |
| MMDetection3D | `fe25f7a51d36e3702f961e198894580d83c4387b` | nuScenes CenterPoint配置及其继承的cyclic-20e计划 |

## 2. 实际训练配方

| 项目与具体配置 | 真实batch含义 | 优化器与学习率 | 轮次与调度 | 对我们的参考意义 |
| --- | --- | --- | --- | --- |
| T-RODNet，config_T_Rodnet_win16 | batch=4；脚本每batch更新，无梯度累积；输入序列16帧 | Adam，1e-4，weight_decay=0 | 100轮，CosineAnnealingLR，默认eta_min=0，每轮末step | 最接近当前骨干；支持以4作为合理真实batch起点、围绕1e-4搜索，并测试更长余弦训练 |
| RODNet，HG1 win16 | batch=4；标准DataLoader路径每batch更新；16帧 | Adam，1e-5，默认无权重衰减 | 50轮，StepLR每5轮乘0.1 | 小batch很常见；HG与Transformer未必共用最优学习率；不建议直接照搬极快衰减 |
| OpenPCDet，ONCE PointPillar | 每GPU=4，全局batch还取决于GPU数 | adam_onecycle；配置LR=0.003为峰值；封装使用解耦weight decay=0.01 | 80轮；前40%更新从3e-4升至3e-3，再降至3e-8；动量同步变化 | 借鉴完整更新级调度、明确峰值/初始值；不能把0.003直接设成我们的固定LR |
| MMDetection3D，nuScenes CenterPoint voxel01 | 每GPU=4；参考总batch=8×4=32 | AdamW，起始LR=1e-4，weight_decay=0.01，clip_norm=35 | 20轮；前8轮升至1e-3，后12轮降至1e-8；按iteration换算；使用CBGS数据包装 | batch32不是单卡32；20轮也不是与我们等量的更新或样本；默认自动缩放LR关闭 |

证据：

- [T-RODNet配置](https://github.com/Zhuanglong2/T-RODNet/blob/39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9/configs/config_T_Rodnet_win16.py)、[实际训练脚本](https://github.com/Zhuanglong2/T-RODNet/blob/39a2444c2bdf83f56645fb9ea74da1ca2f2eeab9/tools/train.py)。注意配置中的 `lr_step=5` 不生效：StepLR已注释。`win_size=16`是输入序列长度，网络构造的`win_size=4`是注意力窗口参数，不能混同。
- [RODNet HG1配置](https://github.com/yizhou-wang/RODNet/blob/44fd61fa7b433b920469bd8455a54d5676e57fd8/configs/config_rodnet_hg1_win16.py)、[训练脚本](https://github.com/yizhou-wang/RODNet/blob/44fd61fa7b433b920469bd8455a54d5676e57fd8/tools/train.py)。本表针对HG1，不能以其他v2配置替代。
- [OpenPCDet ONCE PointPillar](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/cfgs/once_models/pointpillar.yaml)、[优化器构造](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/train_utils/optimization/__init__.py)、[OneCycle实现](https://github.com/open-mmlab/OpenPCDet/blob/233f849829b6ac19afb8af8837a0246890908755/tools/train_utils/optimization/learning_schedules_fastai.py)。`LR_WARMUP=False`不取消OneCycle自身的升温阶段。
- [MMDetection3D CenterPoint配置](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/centerpoint/centerpoint_voxel01_second_secfpn_8xb4-cyclic-20e_nus-3d.py)、[继承的cyclic-20e](https://github.com/open-mmlab/mmdetection3d/blob/fe25f7a51d36e3702f961e198894580d83c4387b/configs/_base_/schedules/cyclic-20e.py)。后者注释声称从0升温，但实际优化器初始化1e-4，应以可执行配置解释。CBGS改变每轮采样构成和长度。

## 3. 哪些经验可以直接采用

1. **batch4可以作为基准。** 四个参考配置都出现每设备或脚本batch4。增加batch要作为实验变量，不预设16/32一定更好。
2. **学习率必须连同调度解释。** T-RODNet为1e-4起始的余弦，OpenPCDet为3e-3峰值的OneCycle；这两个数不能直接比较大小。新的计划应记录每次optimizer update对应的LR。
3. **较长训练值得验证，但不直接复制100轮。** 原始窗口、数据规模、损失和采样不同。我们已有20轮较5轮更好的Risk和共同目标运动指标，但部分几何Risk及总验证loss会退化。
4. **优化器差异单独做对照。** 原T用Adam且无weight decay；我们用AdamW且weight_decay=1e-4、clip_norm=5。第一轮不同时换Adam、改衰减、调batch并加轮次，否则难归因。
5. **重复实验先于宣布小幅收益。** PyTorch说明固定seed不等于所有CUDA算子都确定；应记录随机状态、后端设置和环境，必要时启用确定性算法检查。见[官方可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)。这提供排查方法，不证明我们当前差异已定位。

## 4. 与RadarVLA的关键差异

- 我们仅移植骨干，没有移植官方整套检测器。官方T-RODNet训练分支使用MSE置信图损失，RODNet HG使用BCE；我们保留Hungarian匹配、focal objectness、几何/运动与五维Risk联合监督。原损失数值和最优学习率不可直接迁移。
- 我们输入4帧Power/Doppler，官方配置输入16帧；两通道的物理定义也不能仅因通道数一致而视为相同。此次保持四帧、可靠性mask和离线双向解折叠数据不变。
- 原官方T脚本每batch更新，我们batch4累积2后有效batch8。即便真实batch相同，更新频率也不相同。
- 我们检测objectness按一个真实batch内的匹配目标数归一化，其他分支按有效标签数计算均值。梯度累积不能严格模拟更大真实batch的归一化，因此必须分别记录两者。
- 训练集6847、验证1472；真实batch4累积2约856次更新/轮，20轮17120次更新。改有效batch32而仍训20轮，会明显减少更新数；解释时必须同时报告样本遍历次数与更新次数。
- 验证集危险车辆实例仅35个，不宜仅凭其中1—2个检出差异选模型，也不能将验证集多次调参后的最好值当成独立test结论。

## 5. 建议采用的有限搜索计划（尚未执行）

以T-RODNet为主，HG保留既有对照；固定moving-v1 manifest、数据划分、候选域、损失、阈值/NMS、BF16设置。保留目前20轮flat-cosine模型作为历史参考，重新训练的同配置作为本轮基准。

### 阶段0：复现与资源测量

先检查同seed初始化、固定batch前向/梯度、若干更新后的模型/优化器/RNG状态。检查确定性模式是否遇到不支持算子，不为通过检查而静默改变骨干或池化算法。如不能逐位复现，明确记录限制，所有最终候选做多seed，不能声称已证明调度因果收益。

在可用GPU上测batch4/8/16的峰值显存与吞吐，再确定可运行范围；不能为追求16而占用别人任务所需余量。此次调研本身不调用GPU。

### 阶段1：学习率与调度，先固定batch

- 真实batch4、累积2、AdamW/weight_decay=1e-4、20轮。
- 第一批：起始/平台LR为5e-5、1e-4、2e-4，各用已有flat-cosine计划，隔离学习率因素。
- 额外一个最接近原T思路的对照：LR=1e-4，从训练开始持续余弦衰减至1e-6，20轮；保持我们AdamW不变。该组比较调度形状，不宣称完整复现原论文。
- 不把5轮成绩直接当最终淘汰依据；已有实验显示部分Risk和运动收益出现在更后期。只因非有限loss/OOM等技术失败提前终止，或事先明确统一中期淘汰规则。

### 阶段2：真实batch与有效batch分开

- 用阶段1较好配方，对照真实batch4/8/16，累积4/2/1，固定有效batch16；batch16仅在资源实测允许时启用。此阶段比较真实batch效应。
- 再固定可运行的真实batch，比较有效batch8/16/32。先保持LR不变，不自动套线性缩放；若大batch退化，再单列LR补偿实验。
- 固定轮次比较代表固定数据遍历预算；需要判断是否仅因更新变少时，增加一个匹配更新次数的对照，并单独报告其额外样本预算。两种口径不能混为同一公平预算。

### 阶段3：延长训练和复验

- 对前1—2个候选从头按完整40轮计划训练，并与匹配的20轮计划比较。20/40轮余弦轨迹前段不同，不能把20轮权重直接接上再宣称是同一40轮配方。
- 40轮仍持续改善才考虑更长计划；不直接启动100轮全网格。
- 最终候选用3个固定seed报告均值及波动。共享同一检查点分叉可用于调度对照，但必须增加明确的分叉协议、完整模型/优化器/RNG状态和来源记录，不能篡改旧resume协议。

### 判定与验收

- 检测：固定阈值Precision/Recall/FP、危险目标检出，同时查看PR随阈值变化，避免把校准变化误当表征提升。
- 运动：速度、径向速度、未来ADE；既报告每模型检出集合，也报告共同匹配目标集合及其规模。
- Risk：1/2/3秒AP与Brier，dmin/areq误差；不以单一总loss或Pcol3 AP替代其他维度。
- 报告非支配候选及取舍，避免事后拼出人为综合分数；最终标准在test评估前确定。测试集只用于最终锁定配方后的评估。

## 6. 执行位置和保护边界

- 主源码：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/RadarVLA`，本轮未修改。
- 数据：`/home/luxinlan/data/lxl_data/RadarVLA`。
- 已准备隔离worktree：`/home/luxinlan/PublicWorkspace/Xinlan_Lu/.worktrees/RadarVLA-hparam-20261021`，分支`codex/hparam-20261021`，基于`48241754c6cdc3d838cc3a4a399340cae0884306`。
- 独立Python：`/home/luxinlan/data/lxl_data/RadarVLA/model_runs/hparam_20261021/venv/bin/python`。
- 研究记录同步至数据目录`analysis/hparam_20261021/RESEARCH.md`；上游代码快照只作外部研究材料，不进入主源码仓库。
- 后续运行保持CPU线程1、data workers=0、nice=19、ionice=idle；不因存在其他用户进程就自动停止，也不预先假定共享GPU没有性能影响。训练前按当时负载和实际峰值预算选择资源，仅管理本worker。

结论：优先借鉴T-RODNet的batch4、1e-4量级和余弦训练；先解决复现与比较口径，再依次搜索学习率/调度、batch、训练长度。当前没有证据支持直接采用单卡batch32或100轮作为默认最优方案。
