"""Opt-in profile bridge; no model-specific request parameters are introduced.

mini-SWE's query accepts **kwargs, while the existing instrumentation only
injects correlation headers when extra_headers is explicit. Promote the
profile's headers into query kwargs before entering that unchanged wrapper.
All API parameters (including extra_body) remain mini-SWE/LiteLLM's concern.
"""
from __future__ import annotations

from agentmon.adapters.inference.prefix_reuse import affinity_headers

from .model import InstrumentedLitellmModel


def profile_request_kwargs(config_kwargs, call_kwargs, *, session_id=None, dp_size=0):
    result = dict(call_kwargs)
    headers = dict(config_kwargs.get("extra_headers") or {})
    for name, value in (result.get("extra_headers") or {}).items():
        for configured in tuple(headers):
            if configured != name and configured.lower() == name.lower():
                del headers[configured]
        headers[name] = value
    result["extra_headers"] = affinity_headers(headers, session_id, dp_size)
    return result


class ProfiledLitellmModel(InstrumentedLitellmModel):
    @property
    def model_name(self):
        return self.config.model_name

    def query(self, messages, **kwargs):
        dp_size = self._agentmon_compressor().cfg.vllm.dp_affinity_size
        session_id = kwargs.pop("agentmon_session_id", None)
        if session_id is None and dp_size > 1:
            context = self._agentmon_sink().context
            if context.run_id != "unattached":
                session_id = context.run_id
                if context.agent_id:
                    session_id += "/" + context.agent_id
        request_kwargs = profile_request_kwargs(
            self.config.model_kwargs, kwargs, session_id=session_id, dp_size=dp_size
        )
        return super().query(messages, **request_kwargs)
