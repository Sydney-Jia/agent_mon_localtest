from pathlib import Path
import unittest

from agentmon.config import load_config
from agentmon.adapters.agent_frameworks.mini_swe.profiled_model import profile_request_kwargs
from agentmon.adapters.agent_frameworks.mini_swe.model import inject_request_headers
from agentmon.adapters.inference.prefix_reuse import DP_RANK_HEADER, affinity_headers
from agentmon.adapters.inference.vllm_018.version import is_supported_vllm_version


ROOT = Path(__file__).resolve().parents[2]


class V41ProfileTests(unittest.TestCase):
    def test_target_config_is_independent_and_conservative(self):
        cfg = load_config(ROOT / "examples/config.deepseek_v41.toml", env={"AGENTMON_REPO": str(ROOT)})
        self.assertEqual(cfg.vllm.api_base, "http://127.0.0.1:8800/v1")
        self.assertEqual(cfg.model.max_model_len, 262144)
        self.assertEqual(cfg.model.max_num_seqs, 1)
        self.assertEqual(cfg.vllm.request_timeout_seconds, 2.0)
        self.assertEqual(cfg.vllm.prefix_caching, "unknown")
        self.assertEqual(cfg.vllm.dp_affinity_size, 0)
        self.assertFalse(cfg.context_compression.enabled)
        self.assertFalse(cfg.runtime_optimization.enabled)
        self.assertEqual(cfg.finalization.p2_events_path, "")
        self.assertIn("VLLM::Worker_TP", cfg.vllm.process_match)
        self.assertFalse(any("qwen" in x.lower() for x in cfg.vllm.process_match))

    def test_header_promotion_does_not_mutate_profile_or_add_thinking(self):
        profile = {"extra_headers": {"x-custom": "old"}, "extra_body": {"chat_template_kwargs": {"thinking": False}}}
        call = {"extra_headers": {"x-custom": "new"}, "timeout": 30}
        result = profile_request_kwargs(profile, call)
        self.assertEqual(result["extra_headers"]["x-custom"], "new")
        self.assertEqual(profile["extra_headers"]["x-custom"], "old")
        self.assertNotIn("extra_body", result)
        self.assertNotIn("extra_body", profile_request_kwargs({}, {}))

    def test_existing_header_injector_accepts_opt_in_profile(self):
        def query(messages, **kwargs): pass
        result = profile_request_kwargs({}, {})
        self.assertTrue(inject_request_headers(result, "run", "call", target=query))
        self.assertEqual(result["extra_headers"]["X-AgentMon-Model-Call-Id"], "call")
        untouched = {}
        self.assertFalse(inject_request_headers(untouched, "run", "call", target=query))
        self.assertEqual(untouched, {})

    def test_no_p2_gate_bypass(self):
        self.assertTrue(is_supported_vllm_version("0.18.0"))
        self.assertFalse(is_supported_vllm_version("0.22.1"))

    def test_session_affinity_is_stable_and_respects_explicit_route(self):
        ranks = {affinity_headers({}, f"session-{i}", 2)[DP_RANK_HEADER] for i in range(16)}
        self.assertEqual(ranks, {"0", "1"})
        first = profile_request_kwargs({}, {}, session_id="session-1", dp_size=2)
        retry = profile_request_kwargs({}, {}, session_id="session-1", dp_size=2)
        self.assertEqual(first, retry)
        self.assertNotIn(DP_RANK_HEADER, affinity_headers({}, "unattached", 2))
        self.assertNotIn(DP_RANK_HEADER, affinity_headers({}, "session-1", 1))
        explicit = profile_request_kwargs(
            {"extra_headers": {DP_RANK_HEADER: "0"}},
            {"extra_headers": {"x-data-parallel-rank": "1"}},
            session_id="session-1",
            dp_size=2,
        )
        self.assertEqual(explicit["extra_headers"], {"x-data-parallel-rank": "1"})


if __name__ == "__main__":
    unittest.main()
