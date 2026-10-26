# 2026-10-25 配额清理与续训

只删除了一个可重建的未完成打包文件：

`/home/luxinlan/.cache/radarvla-rain-fog/formal_spool/processing/scenario7_9_Town01_68_rain_9ea3ff8122/packages/scenario7_9_Town01_68_rain_9ea3ff8122.raw.tar.partial`

- 文件大小：14,891,769,856字节，约13.87GiB。
- 清理前确认无采集、打包、训练进程；遍历临时tar内12,317个条目，对应原始路径均存在，文件大小一致；tar尾部确认为不完整。
- 打包输入jobs、prepared、acceptance.json、dataset_index.jsonl、代码、环境、checkpoint全部保留。
- 工作盘账户配额占用从102396MiB降至88195MiB，已低于100GiB硬配额；仍超过80GiB软配额，检查时宽限期约6天。
- 不对NAS扫描、删除或重传；未清理其他用户文件。

续训沿用原始源码、配置、execution指纹和输出目录，仅恢复risk_sft。原STATUS已另存备份，不重跑已完成10项实验。

恢复监督进程2962324；新worker2971826；GPU0；audit位于原控制目录risk_sft/attempt_02。从完整的第2轮last.pt恢复，重跑第3轮；中断的半轮没有独立checkpoint。

09:30状态：worker已启动，正在真实样本preflight，尚未据此断言已进行优化器更新。以服务器RECOVERY_20261025.json、attempt_02/training.log和输出status.json为后续准据。

清理收据：同目录cleanup_result.json；服务器另存于`/storage/ssd/metaiot_guest/luxinlan_radarvla_cache/maintenance_20261025/`。
