# 在线采集与 Hook

在 Linux 上安装本 ZIP 后运行 `agentmon-capture run -- <原命令>`。运行目录中生成 manifest.json 和 events.jsonl。用 `agentmon-capture doctor --help` 查看环境检查参数。原命令的退出状态会返还给调用方。

启用引擎 Hook 要将本 ZIP 解压到 Engine 可见路径。0.30 文件后端的配置见 hooks/v030/FILESYSTEM.md，Bench120 插桩与事件边界见 docs/HOOKS.md。0.18 / 0.27.1 启动入口在 hooks/legacy；同一 Engine 只选择适配其版本的入口。默认安装不会把 sitecustomize 注入系统 Python。硬件旁路采样在 tools/importance_hardware.py，依赖 Linux、相应工具与现有采样进程协议。
