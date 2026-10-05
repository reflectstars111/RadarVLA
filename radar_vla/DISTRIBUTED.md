# RadarVLA：8 × L40S / Qwen2.5-3B 运行配置

本目录属于独立的 RadarVLA 任务。以下相对路径命令均在本目录运行。分布式入口不读取医学分割实验队列，不复用其检查点。配置目标是 **8 张 L40S，每卡约 44 GiB 可用显存**。这里给出启动配置和验证流程；尚未进行真实 Qwen 权重、真实雷达数据和 8 卡显存验证，配置不是已经测得的吞吐或容量结论。

## 数据与模型约定

- 单帧输入保持原生 **`[2, 256, 107]`**，两通道依次为线性 Power 和 unfolded Doppler；不把雷达图 resize 成医学图像的 512 × 512。
- 初始配置使用 **4 帧历史**，单样本张量为 `[4, 2, 256, 107]`。4 帧是可修改的初始假设，需要和实际记录、采样频率及 `time_offsets_s` 一致。Power、Doppler 的 NPY 文件分别存 `[N, 256, 107]`。
- `range_m`、`azimuth_rad` 和 `time_offsets_s` 必须来自实际标定。不能仅凭栅格大小推断物理坐标。
- 默认最多 8 个 agent、6 个未来评价时刻、4 个样条控制点；改变标签容量/时间范围时，同时调整 model/planner 配置及 manifest。评价时间以 `future_times_s` 为准。
- Stage 1 训练 Radar encoder 和 physical/risk grounding，不加载 Qwen。
- Stage 2 使用本地 **Qwen2.5-3B** 权重和 LoRA，接受连续 Radar prefix、连续 predicted KRS token 和语言指令。不会按风险阈值切换两套模型。
- SHORT/LONG 教师由反事实风险生成：近时域冲突、危险间距或所需制动力超过阈值时使用 SHORT。推理模式由同一生成模型决定，不读取 GT risk。严格 SFT 预检模式、地图、ego 与 agent future；使用 `build-cohort` 处理缺失监督并保留完整排除报告。
- 这是离线研究训练入口；格式完整和 token loss 下降不代表轨迹安全，也不替代闭环驾驶验证。

配置文件：[configs/l40s_qwen25_3b.json](configs/l40s_qwen25_3b.json)。`data.history_frames`、物理归一化尺度、训练轮数和学习率都是初始设置，应在训练集/验证集上确定；不要据测试集反复调参。

## 独立环境与本地权重

推荐独立环境 `radar_vla/.conda`。运行前按主使用文档安装 PyTorch、Transformers、PEFT 等依赖，并用实际路径设置 `QWEN_MODEL_PATH`。启动器开启 Hugging Face 离线模式，不负责下载权重或安装包。

```bash
export RADAR_VLA_ENV=/ssd1/code/Multi-Organ_Foundation_Model/radar_vla/.conda
export QWEN_MODEL_PATH=/absolute/path/to/Qwen2.5-3B
export MANIFEST=/absolute/path/to/prepared_radar_manifest.jsonl

"$RADAR_VLA_ENV/bin/python" -m radar_vla validate --manifest "$MANIFEST" --max-agents 8
```

从其他工作目录调用启动脚本时，使用脚本绝对路径；`MANIFEST`、`OUTPUT` 等相对路径相对于调用时的目录解析。`RADAR_VLA_PYTHON` 可以直接指定 Python 可执行文件并覆盖默认环境选择。

## 先检查命令，再做单卡短验证

`--dry-run` 仅打印启动命令，不导入 CUDA、不读模型权重、不启动训练。

```bash
STAGE=grounding OUTPUT=/absolute/path/to/runs/grounding_check \
  NPROC_PER_NODE=1 ACCUMULATION_STEPS=1 STOP_AFTER_EPOCH=1 \
  bash ./launch_l40s.sh --dry-run
```

确认路径和环境后，去掉 `--dry-run` 可做单卡单轮验证。大数据集的一轮仍可能耗时，应先准备保持 scene split 独立的小型代表性 manifest。建议为该验证使用单独的输出目录，验证输入维度、标签掩膜、checkpoint 读写与实际显存峰值。

Stage 2 也要先用本地 Qwen 做单卡验证：

```bash
STAGE=sft OUTPUT=/absolute/path/to/runs/sft_check \
  INIT_GROUNDING=/absolute/path/to/runs/grounding_check/best.pt \
  NPROC_PER_NODE=1 ACCUMULATION_STEPS=1 STOP_AFTER_EPOCH=1 \
  bash ./launch_l40s.sh --dry-run
```

先核对命令，再去掉 `--dry-run`。上述单卡设置每个完整优化窗口只有 1 个样本，用于联通验证；不应与正式有效 batch 32 的结果混为一谈。

## 正式双阶段启动

Stage 1 示例：

```bash
STAGE=grounding OUTPUT=/absolute/path/to/runs/radar_grounding \
  NPROC_PER_NODE=8 BATCH_SIZE=1 ACCUMULATION_STEPS=4 \
  EPOCHS=5 LR=0.0003 \
  bash ./launch_l40s.sh
```

Stage 1 完成并检查验证集结果后，Stage 2 示例：

```bash
STAGE=sft OUTPUT=/absolute/path/to/runs/radar_qwen_sft \
  INIT_GROUNDING=/absolute/path/to/runs/radar_grounding/best.pt \
  NPROC_PER_NODE=8 BATCH_SIZE=1 ACCUMULATION_STEPS=4 \
  EPOCHS=5 LR=0.0003 \
  bash ./launch_l40s.sh
```

`BATCH_SIZE` 是**每个 GPU 进程**的 microbatch，完整梯度累积窗口的全局 batch 为 `8 × 1 × 4 = 32`。最后一个不完整窗口及 distributed sampler 的取舍以运行记录为准；它们不能被默认为恰好 32 个独立样本。`EPOCHS=5`、`LR=0.0003` 是初始运行值，没有宣称它们已优化。

DDP 在每张卡各持有一份模型副本，不会自动把 8 张卡显存合并为一块。LoRA、BF16、gradient checkpointing 和较小 microbatch 用于降低单卡开销，仍需实测。若显存不足，优先减小每卡 batch、历史帧数或文本/生成长度，并记录协议变化；增加梯度累积只能恢复 batch，不能恢复被缩短的历史信息。

如需手动后台运行，可将单次启动交给本机任务管理器或在上述命令外加 `nohup` 并重定向日志；本项目不会额外创建持续在线监测服务。

## 恢复与手动监测

使用相同配置和输出目录恢复，8 卡实验默认保持同样的 rank 数与 batch/accumulation 设置：

```bash
STAGE=sft OUTPUT=/absolute/path/to/runs/radar_qwen_sft RESUME=1 \
  NPROC_PER_NODE=8 BATCH_SIZE=1 ACCUMULATION_STEPS=4 \
  bash ./launch_l40s.sh
```

恢复 SFT 仍需 `QWEN_MODEL_PATH` 指向原本地基础模型；不会重新选择 Stage 1 初始化检查点。恢复的实际粒度以 checkpoint 记录为准。

可手动查看硬件和自己重定向的训练日志：

```bash
watch -n 10 nvidia-smi

tail -n 30 -F /absolute/path/to/radar_training.log
```

当前代码提供 v2 完整离线监督链路和启动配置，没有提交长期 8 卡训练；完整 Qwen2.5-3B 的实际显存与吞吐需在目标服务器验证。
