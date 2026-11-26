"""Compatibility namespace for mini-SWE-agent integrations."""
from agentmon.adapters.agent_frameworks import mini_swe as _impl

__all__ = _impl.__all__


def __getattr__(name):
    return getattr(_impl, name)
