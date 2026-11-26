"""Opt-in startup hook. Read README before adding this directory to PYTHONPATH."""
import os
if os.getenv('AGENTMON_VLLM_030_ENABLE')=='1' or os.getenv('AGENTMON_VLLM_030_FS_ENABLE')=='1':
    try:
        from agentmon.adapters.inference.vllm_030.hook import install
        install()
    except Exception:
        # Do not leak input/env data or prevent model startup. Missing startup
        # evidence remains a failed observation gate, not silent acceptance.
        import sys
        print('AGENTMON_V030_STARTUP_FAILED; observation not verified',file=sys.stderr)
