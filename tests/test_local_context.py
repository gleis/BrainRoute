"""Context-window handling for local models.

Two failure modes are covered here, and they pull in opposite directions:

- Under-reporting a window escalates a long prompt to the cloud that a local
  model could have handled - a privacy leak caused by a bad default.
- Over-reporting it, or failing to ask the runtime for the window, gets the
  prompt silently truncated - a quality loss with no visible symptom.
"""

import unittest
from unittest.mock import patch

from brainroute import catalog
from brainroute.classifier import TaskProfile
from brainroute.execution import required_context
from brainroute.policy import apply_policy
from brainroute.pricing import apply_pricing
from brainroute.providers.ollama import (
    NUM_CTX_BUCKETS,
    OLLAMA_DEFAULT_NUM_CTX,
    _payload,
    resolve_num_ctx,
)
from brainroute.settings import override


TAGS = {"models": [
    {"name": "qwen2.5:7b", "details": {"family": "qwen2", "parameter_size": "7.6B"}},
    {"name": "llama3.3:70b", "details": {"family": "llama", "parameter_size": "70.6B"}},
]}

SHOW = {
    "qwen2.5:7b": {"model_info": {"general.architecture": "qwen2", "qwen2.context_length": 32768}},
    "llama3.3:70b": {"model_info": {"general.architecture": "llama", "llama.context_length": 131072}},
}


def discover(show=SHOW, detect=True):
    def fake_post(url, payload, timeout):
        if payload["model"] not in show:
            raise RuntimeError("not found")
        return show[payload["model"]]

    with override({"catalog": {"detect_local_context": detect}}), \
         patch("brainroute.catalog._read_json", return_value=TAGS), \
         patch("brainroute.catalog._post_json", side_effect=fake_post):
        return {m["model"]: m for m in catalog.discover_ollama("http://localhost:11434")}


class ContextDetectionTests(unittest.TestCase):
    def test_real_windows_are_read_from_the_runtime(self):
        found = discover()
        self.assertEqual(found["qwen2.5:7b"]["max_context"], 32768)
        self.assertEqual(found["llama3.3:70b"]["max_context"], 131072)
        self.assertEqual(found["qwen2.5:7b"]["context_source"], "detected")

    def test_detection_can_be_turned_off(self):
        found = discover(detect=False)
        self.assertEqual(found["qwen2.5:7b"]["context_source"], "estimated")

    def test_a_failed_probe_leaves_the_estimate_in_place(self):
        found = discover(show={})
        # The family table still beats the old flat 8192 guess.
        self.assertEqual(found["qwen2.5:7b"]["max_context"], 32768)
        self.assertEqual(found["qwen2.5:7b"]["context_source"], "estimated")

    def test_the_family_table_covers_common_local_models(self):
        self.assertEqual(catalog._context_guess("llama3.3:70b"), 131072)
        self.assertEqual(catalog._context_guess("qwen2.5-coder:14b"), 32768)
        self.assertEqual(catalog._context_guess("mistral:7b"), 32768)

    def test_longer_family_names_win_over_shorter_prefixes(self):
        # "llama3.3" must not be matched by the shorter "llama3" entry.
        self.assertNotEqual(
            catalog._context_guess("llama3.3:70b"), catalog._context_guess("llama3:8b")
        )

    def test_an_explicit_size_in_the_tag_wins(self):
        self.assertEqual(catalog._context_guess("custom-model:64k"), 64 * 1024)

    def test_unknown_models_fall_back_conservatively(self):
        # Guessing low escalates; guessing high truncates. Low is the safer error.
        self.assertEqual(catalog._context_guess("some-private-model:latest"), 8192)

    def test_malformed_metadata_does_not_break_discovery(self):
        broken = {"qwen2.5:7b": {"model_info": {"qwen2.context_length": "not-a-number"}}}
        self.assertEqual(discover(show=broken)["qwen2.5:7b"]["context_source"], "estimated")


class NumCtxTests(unittest.TestCase):
    MODEL = {"id": "local", "provider": "ollama", "model": "qwen2.5:7b", "max_context": 32768}

    def test_short_prompts_do_not_request_a_larger_window(self):
        self.assertIsNone(resolve_num_ctx(self.MODEL, 1_000))
        self.assertNotIn("options", _payload(self.MODEL, [], stream=False, context_tokens=1_000))

    def test_a_long_prompt_requests_a_bucket_that_holds_it(self):
        num_ctx = resolve_num_ctx(self.MODEL, 9_000)
        self.assertEqual(num_ctx, 16384)
        self.assertGreaterEqual(num_ctx, 9_000)

    def test_buckets_avoid_allocating_a_models_whole_ceiling(self):
        big = {**self.MODEL, "max_context": 131072}
        # A 5k prompt on a 128k model must not ask for 128k of KV cache.
        self.assertLess(resolve_num_ctx(big, 5_000), big["max_context"])

    def test_the_request_never_exceeds_the_models_window(self):
        for tokens in (40_000, 200_000):
            self.assertLessEqual(resolve_num_ctx(self.MODEL, tokens), self.MODEL["max_context"])

    def test_buckets_are_ascending(self):
        self.assertEqual(list(NUM_CTX_BUCKETS), sorted(NUM_CTX_BUCKETS))
        self.assertEqual(NUM_CTX_BUCKETS[0], OLLAMA_DEFAULT_NUM_CTX)

    def test_the_option_reaches_the_request_payload(self):
        payload = _payload(self.MODEL, [], stream=False, context_tokens=9_000)
        self.assertEqual(payload["options"]["num_ctx"], 16384)

    def test_required_context_leaves_room_for_a_reply(self):
        messages = [{"role": "user", "content": "x" * 4000}]
        self.assertGreater(required_context(messages), 1000)


class LongPromptRoutingTests(unittest.TestCase):
    """The end-to-end behaviour the fix exists for."""

    def _models(self):
        local = [apply_pricing({**m, "enabled": True}) for m in discover().values()]
        cloud = apply_pricing({
            "id": "premium-reasoner", "local": False, "enabled": True,
            "model": "premium-reasoner", "max_context": 200000, "quality_score": 0.95,
        })
        return local + [cloud]

    def _allowed(self, tokens):
        task = TaskProfile("reasoning", "high", "normal", "normal", (), context_tokens=tokens)
        # Budgets are generous here so the context window is the only thing
        # these assertions can be measuring.
        with override({"policy": {"max_estimated_cost_usd": 100.0}}), \
             patch("brainroute.policy.snapshot", return_value={}), \
             patch("brainroute.policy.spend_since", return_value=0):
            return [m["id"] for m in apply_policy(self._models(), task, "x").allowed]

    def test_a_nine_thousand_token_prompt_stays_eligible_for_local(self):
        # This is the regression: with a guessed 8192 window every local model
        # was rejected and the prompt escalated to the cloud.
        self.assertIn("ollama-qwen2-5-7b", self._allowed(9_000))

    def test_a_prompt_beyond_the_smaller_model_uses_the_larger_one(self):
        allowed = self._allowed(60_000)
        self.assertNotIn("ollama-qwen2-5-7b", allowed)
        self.assertIn("ollama-llama3-3-70b", allowed)

    def test_a_prompt_beyond_every_local_model_still_escalates(self):
        self.assertEqual(self._allowed(150_000), ["premium-reasoner"])


if __name__ == "__main__":
    unittest.main()
