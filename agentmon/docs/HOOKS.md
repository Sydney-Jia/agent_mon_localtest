# AgentMon 的 Hook 如何采集数据

假设一次请求检查三个缓存组。第一个组返回有效块，第二个组查不到，vLLM 随即退出。只看联合查询的最终结果，会丢失两条信息：第一组已经命中，第三组根本没有被访问。AgentMon 在原分支返回的位置读出已访问列表和首个缺失组，保留这三个不同的结果。随后用同一次请求的最终选用记录回答，Scheduler 实际用了多少历史。

## 两种接入方式

| | Bench120 HBM-only | vLLM 0.30 可选观察器 |
|---|---|---|
| 何时接入 | Prepare 针对审核过的源码生成带观测调用的文件，新 Engine 加载 | Python 启动加载 `sitecustomize.py`，按开关安装 import hook |
| 如何定位 | `importance_build.overlay` 组合 query、pool、compute、worker 四个构建适配器；各自核对原方法的锚点或 AST | `Finder` 拦截已登记的模块名，`Loader` 先执行原模块，再由 `Installation` 安装观察器；已导入的目标也会接受检查 |
| 如何确认加载 | Prepare 记录实际生成文件的 SHA。Admission 对照 Scheduler、Worker 的加载方法、PID、Epoch 和 Writer 事件 | bindings.json 固定受支持源码 SHA；对所用方法核对加载代码与源文件编译结果。事件区分安装成功、目标拒绝和真实业务调用 |
| 谁写日志 | 生成的边界函数调用已有 Native State Writer | 各进程的 EventSink 用有界队列、后台写线程和分片 JSONL 落盘，另写 health 文件 |

## 一条 HBM 请求怎样串起来

1. **到达。** Scheduler 读取引擎生成的完整块哈希向量。按请求版本记录哈希分块、覆盖范围、组规则和 Pool 身份。原策略提前停止 Lookup 时，向量仍能描述后续完整块的身份。最后不足一块的 Token 单独对待。
2. **查找。** `BlockPool.get_cached_block` 沿原有顺序查组。探针记录实际访问的组、HIT 对应的 Block/Lease、首次 MISS 和后续未访问的组。MISS 没有命中副本，Lease 不适用。Key 摘要标识缓存键，不能作为 Tensor 数值校验和。
3. **选用与执行。** 通过 Request、Query 和 Decision ID 对照候选边界与最终选用。Scheduler 发出的批次再用 Batch ID/SHA、Rank、Worker Epoch 和 Forward 身份对回抽样块表与执行时间。Host 方法返回、设备流计时和调用方收到首 Token 是三个不同的时间点。
4. **释放与回收。** 从保留 null 槽后的队列顺序开始，记录分配请求和已发生的队列编辑，核对实际头、尾及剩余数量。引用归零后，仍在索引中的版本可能再被请求命中；索引删除和同槽新 Lease 能进一步说明旧版本何时让位。

复核可从 `importance_query.py`、`importance_pool.py` 和 `importance_compute.py` 的 observe 调用入手；对应的 `*_build.py` 说明这些调用被插在原程序的哪一处。运行后的 `KV_IMPORTANCE_COVERAGE.json` 汇总查询与选择能否对应、队列首次分歧及抽样批次能否跨 Rank 配对。

## 文件搬运怎样观测

启用文件观察器且原引擎使用相应后端时，沿 `FileSystemTierManager` 的真实提交记录 Job、Key 和槽位。任务进入线程池后，Python 文件路径上的观察器记录存在性检查、实际 write/readv 和发布，再对回任务结束与调用方取回完成结果。已有文件跳过写入时，Job 已提交而实际写入量可以为零。C 扩展路径的可见性止于可证明的批次边界。仅凭文件名或时间接近，无法把文件作业归给某个 HBM Lease。

源码入口为 `hooks/v030/sitecustomize.py` 和 `src/agentmon/adapters/inference/vllm_030/{hook,instrument,filesystem,events}.py`。启动参数与限制见 [文件后端采集说明](../hooks/v030/FILESYSTEM.md)。

## 怎么判断这段数据可用

先固定 Run、进程、Epoch 和观察截止点，再核对 Writer 序号、accepted、written、丢弃、错误与轮转覆盖。可见相同身份的重复归档应核对内容后去重。观察窗口未封存时，未见重访属于右截尾；方法安装成功但业务未触发、采样未覆盖和 Writer 丢事件应分别记录。硬件采样描述节点或设备负载，单个 KV 的重算与恢复成本还需要匹配执行路径及批次。
