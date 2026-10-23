# 阶段0：真实样本资源测量

日期：2026-10-21。GPU0 NVIDIA L40S；PyTorch2.5.1+cu124。全部测量已结束。

输入为真实4帧Power/Doppler，尺寸`[B,4,2,256,107]`。资源样本包括空目标、训练manifest中目标数最多的样本（14个）及seed43固定抽样。B4/8/16均包含空目标和14目标样本；三个batch是嵌套集合。模型为完整Grounder，不加载语言模型。

每组在临时模型上执行4次前向、原始联合loss、反向、梯度裁剪和AdamW更新；第1次为预热，之后3次统计中位数。整体使用BF16，未采用FP32插值干预。所有loss和梯度有限，检测候选均为3456。

| 真实batch | 峰值allocated | 峰值reserved | 前向中位数 | 前向+反向+更新中位数 | 模型计算吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 2.44 GiB | 2.94 GiB | 53.22 ms | 100.63 ms | 39.75样本/s |
| 8 | 4.39 GiB | 5.05 GiB | 76.37 ms | 152.31 ms | 52.52样本/s |
| 16 | 8.30 GiB | 9.43 GiB | 127.70 ms | 268.74 ms | 59.54样本/s |

这些是**缓存输入的模型计算测量**：不包含DataLoader、文件读取、完整验证或2秒探针间隔。仅3次稳态计时，且为共享服务器当时负载；不能直接换算正式训练墙钟时间，也不能据此断言batch16收敛更好。

结论：batch16具有资源可行性，可以进入预定对照。相对batch8，模型吞吐仅再提高约13%，其收益需结合显存和训练效果判断；不预设最大batch最优。

后续建议的起始allocator预算：batch4为6GiB、batch8为9GiB、batch16为14GiB；这些是按本次峰值reserved加余量提出的预算，不是已验证的整个训练过程硬上限。正式运行前还需检查验证峰值，并动态根据共享卡实际余量准入；本次没有把这些预算配置到训练worker。

保护措施：单GPU串行，CPU线程1、workers0、nice19、ionice idle；replay allocator cap12GiB，资源测量cap18GiB；启动要求预算外至少3GiB余量，运行时余量低于2GiB仅退出本worker。每次模型操作后休息2秒，不操作其他用户进程。系统nice不构成GPU算力隔离，本次短探针结果不代表长时间共享训练不会影响吞吐。

原始数据：[batch4](stage0/bench_b4.json)、[batch8](stage0/bench_b8.json)、[batch16](stage0/bench_b16.json)。样本ID、目标数、输入哈希见[prepare.json](stage0/prepare.json)；源码、GPU UUID和探针哈希见[STAGE0_PROTOCOL.json](stage0/STAGE0_PROTOCOL.json)。
