"""mini-SWE-agent P1 instrumentation adapters."""

from .adapter import MINI_SWE_P1_CLASSES, MiniSweAgentFrameworkAdapter, adapter, configure_p1_command
from .agent import (
    InstrumentedAgentMixin,
    InstrumentedDefaultAgent,
    InstrumentedInteractiveAgent,
    make_instrumented_agent_class,
)
from .environment import (
    InstrumentedEnvironment,
    InstrumentedLocalEnvironment,
    InstrumentedMiniSweLocalEnvironment,
)
from .model import InstrumentedLitellmModel, InstrumentedModel
from .context_compression import MiniSweContextCompressor


# Runtime controls are provided by the optional experiment distribution.
def __getattr__(name):
    if name in {"RuntimeControlledLocalEnvironment", "RuntimeControlledModel",
                "RuntimeControlledToolResult", "RuntimeControlRejected"}:
        from . import runtime_control
        return getattr(runtime_control, name)
    raise AttributeError(name)


__all__ = [
    "InstrumentedAgentMixin",
    "InstrumentedDefaultAgent",
    "InstrumentedEnvironment",
    "InstrumentedInteractiveAgent",
    "InstrumentedLitellmModel",
    "InstrumentedLocalEnvironment",
    "InstrumentedMiniSweLocalEnvironment",
    "InstrumentedModel",
    "MINI_SWE_P1_CLASSES",
    "MiniSweAgentFrameworkAdapter",
    "MiniSweContextCompressor",
    "RuntimeControlledLocalEnvironment",
    "RuntimeControlledModel",
    "RuntimeControlledToolResult",
    "RuntimeControlRejected",
    "adapter",
    "configure_p1_command",
    "make_instrumented_agent_class",
]
