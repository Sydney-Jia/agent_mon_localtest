# AgentMon 文件后端启动观测

适用于 vLLM v0.30.0 的 `FileSystemTierManager`，在下一次**原本就启用该 secondary tier** 的服务启动时接入。开关只安装观察器，不创建缓存、不改后端、不提交任务。Bench120 HBM-only 实验不启用此项。

对独立解包中的观察器或项目源码，将下面两个目录放入 **Engine 进程可见** 的 `PYTHONPATH`，并提供该次 Run 的证据目录。容器内目录须按实际只读代码挂载和可写 evidence 挂载填写。

```bash
export AGENTMON_VLLM_030_FS_ENABLE=1
export AGENTMON_VLLM_030_RUN_ID="$RUN_ID"
export AGENTMON_VLLM_030_EVENTS="$EVIDENCE_DIR/fs"
export PYTHONPATH="$APP/hooks/v030:$APP/src${PYTHONPATH:+:$PYTHONPATH}"
# 上述环境随原有批准的 Engine 启动命令生效
```

`AGENTMON_VLLM_030_ENABLE` 是独立的原有 .030 观察开关；单独设置 FS_ENABLE 可只安装文件观察器。如果部署已有 sitecustomize，引导程序需在原有启动链中显式调用 `from agentmon.adapters.inference.vllm_030.hook import install; install()`。不要替换现有初始化。

| 事件 | 与作业的关联及口径 |
|---|---|
| `V030_TARGET_INSTALLED` | 同一进程代际必须见到固定 io、thread_pool、manager 三个 SHA 及代码验证。拒绝、缺项或未启用路径分别记录，不能用安装事件证明发生过 I/O。 |
| `FS_LOOKUP` | 记录请求、tier_scope、原 OffloadKey 的 SHA-256 和真实 HIT/MISS/RETRY。异步查询的 RETRY 需保留；MISS 没有命中副本。 |
| `FS_JOB_PLAN / FS_JOB_SLOTS` | 真实 TransferJob 的 job_id、请求、方向、按序 keys 的摘要、实际 FileMapper 路径的摘要、主机槽 ID 和偏移、文件偏移 0、槽宽及 I/O 模式。每 128 槽一行，first 为该行起点，ordinal 为全作业序号。路径摘要使用操作系统路径编码字节的 SHA-256，张量不出进程。 |
| `FS_ENQUEUE_RETURN / FS_TASK_END` | 同一 tier_scope + job_id 关联调度线程与工作线程。记录入队调用起止、任务起止和状态，耗时为含观察开销的 Host monotonic 时钟。 |
| `FS_FILE_OPERATION` | Python 路径中记录原来的 exists 判定、open、write/readv 的实返字节、replace 发布、errno、文件身份和实际槽偏移。已有文件分支写入量为零；短读与异常分别保留。C 扩展路径保留计划与整批结果，逐文件系统调用和跳写结局为未知。 |
| `FS_FINISHED_JOB` | 原生 poll 返回的 success、部分成功的 Key 摘要、后端任务时间和 poll 时刻。通过原生 Run、PID、epoch、tier_scope、job_id 回连。 |

验收时核对 Writer 连续性、drop/error、生成的槽数、每个作业的入队/任务完成/poll，以及每次读写的实际结果。`os.replace` 表示文件已发布；文件系统吞吐与物理盘写放大分别计量。FS TransferJob 没有直接提供上层 HBM 搬运 Job ID 或 slot lease，因此字段留空。跨层串联需上层提供同一逻辑传输身份，也不能把完成到 poll 的时间解释为请求等待。

源码取自 [vLLM v0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0/vllm/v1/kv_offload/tiering/fs)，与 [PyPI 0.30.0 AArch64 wheel](https://pypi.org/project/vllm/0.30.0/#files) 中对应三份文件逐字节一致。SHA 在 vllm_030/bindings.json 中。本地测试实际执行这些文件的方法和双队列线程读写临时目录（Windows 缺少 readv 时仅为测试提供 readinto 等价调用），覆盖跳写、成功回读、部分失败、RETRY 和篡改阻断。C 扩展内部和线上设备尚未验收。
