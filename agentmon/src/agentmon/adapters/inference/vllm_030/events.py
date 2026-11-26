"""Version identity for this adapter's per-process event stream."""
from agentmon.storage.hook_jsonl import HookEventSink
from . import BUILD_ID


class EventSink(HookEventSink):
    build_id = BUILD_ID
    diagnostic_prefix = "AGENTMON_030"
