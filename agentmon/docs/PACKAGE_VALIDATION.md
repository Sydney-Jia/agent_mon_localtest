# 三个成员包的验收范围

| 验证对象 | 本地实测 |
|---|---|
| 发布边界 | 单独解包 online 即可导入文件后端 Hook、运行采集 CLI 帮助。此时找不到实验 CLI 或报告生成器。再加 offline 后从运行目录生成 report.txt。全部源码模块和资源在三份 ZIP 中各归属一包。 |
| HBM 查询 | 调用真实观测函数，以测试对象构造三个组的查询。首组 HIT 对应 Block 9、Lease 7，第二组 MISS 无 Lease，第三组保留为未访问。 |
| Writer 与封存 | 实际创建分片文件、写五条事件，核验连续序号和封存摘要；修改一个字节后校验失败。两个引擎版本复用同一 JSONL Writer，分别检查落盘事件与 health 中的版本身份。 |
| 文件后端 | 用保留上游 SHA 的 vLLM 0.30 Python 文件实现和临时文件执行写入、已有文件跳写、回读、部分失败和 Lookup RETRY。C 扩展路径仅验批次观测边界。Windows 测试用 readinto 代替缺失的 POSIX readv。 |
| 系统采样 | 用样本验证 Host 和 Ascend 指标解析、Prometheus 样本与事件编码。真实 NPU 读数和设备 DMA 需要设备环境。 |

本次 Windows 专项回归通过 20 项。另对原 Bench120 开发环境运行查询、池队列、计算边界和 Worker 块表四组回归，共通过 28 项。9 个对应 Hook 与构建文件按 SHA-256 和原实验源码逐一核对。后四组依赖实验输入，成员包仅携带采集实现。

仓库全量验收仍有平台与依赖限制。Windows 不提供 fcntl、os.sysconf 和 POSIX 0600 文件权限语义；联合验收环境缺少 mini-swe-agent。独立 kv_ana 在短路径下为 330 项通过、3 项失败，涉及本地参考资料及编码。真实 Engine、模型请求与八卡传输未在此次分包验证中运行。

本地可复跑专项：

```bash
python -m pytest -q agentmon/tests/unit/test_distribution_split.py agentmon/tests/unit/test_v030_filesystem_tier.py agentmon/tests/unit/test_ascend_smi.py agentmon/tests/unit/test_host_sampling.py agentmon/tests/unit/test_prometheus.py agentmon/tests/unit/test_events.py
```

Linux 专项另测独立在线包启动一个 Python 子进程、落盘、安装离线包生成报告，以及三包合装后打开完整 CLI。对应流程见仓库的 AgentMon collection packages 工作流；执行状态以具体提交的 Actions 记录为准。
