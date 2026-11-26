from __future__ import annotations

import os
import sys


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


if _truthy(os.environ.get('AGENTMON_OBSERVER_OFF_ATTEST_ENABLE')):
    try:
        from agentmon.adapters.inference.vllm_0271.off_attestation import start
        start()
    except Exception:
        print('agentmon off startup attestation unavailable',file=sys.stderr)


if _truthy(os.environ.get("AGENTMON_VLLM_0271_TRACE_ENABLE")):
    try:
        from agentmon.adapters.inference.vllm_0271.hook import install_from_env

        install_from_env(startup=True)
    except Exception as exc:
        # Never include request data or mutate the model on bootstrap failure.
        print(f"agentmon vllm 0.27.1 bootstrap rejected: {type(exc).__name__}", file=sys.stderr)
elif _truthy(os.environ.get("AGENTMON_VLLM_P2_ENABLE")):
    try:
        from agentmon.adapters.inference.vllm_018.hook import install_from_env

        install_from_env()
    except Exception as exc:
        print(f"agentmon vllm p2 hook bootstrap failed: {exc}", file=sys.stderr)


if _truthy(os.environ.get("AGENTMON_STATE_PLACEMENT_ENABLE")):
    try:
        from agentmon.adapters.inference.vllm_0271.placement import install_from_env as install_placement
        install_placement(startup=True)
    except Exception as exc:
        print(f"agentmon placement bootstrap rejected: {type(exc).__name__}", file=sys.stderr)
