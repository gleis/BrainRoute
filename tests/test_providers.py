import unittest
from unittest.mock import patch

from brainroute.providers import (
    ProviderError,
    flatten,
    generate,
    stream_generate,
    to_messages,
)
from brainroute.providers.http import RETRYABLE_STATUS, with_retries


class MessageTests(unittest.TestCase):
    def test_a_plain_string_becomes_a_user_message(self):
        self.assertEqual(to_messages("hi"), [{"role": "user", "content": "hi"}])

    def test_multi_turn_structure_is_preserved(self):
        conversation = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        self.assertEqual(to_messages(conversation), conversation)

    def test_unknown_roles_are_coerced_to_user(self):
        self.assertEqual(to_messages([{"role": "hacker", "content": "x"}])[0]["role"], "user")

    def test_openai_content_parts_are_flattened(self):
        messages = to_messages([
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
        ])
        self.assertEqual(messages[0]["content"], "ab")

    def test_empty_input_still_yields_a_message(self):
        self.assertEqual(len(to_messages([])), 1)

    def test_flatten_keeps_roles_for_classification(self):
        self.assertIn("system: be terse", flatten(to_messages([
            {"role": "system", "content": "be terse"}, {"role": "user", "content": "go"},
        ])))


class RetryTests(unittest.TestCase):
    def test_retryable_failures_are_retried(self):
        calls = {"count": 0}

        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise ProviderError("503", retryable=True)
            return "ok"

        with patch("brainroute.providers.http.time.sleep"):
            self.assertEqual(with_retries(flaky, retries=3), "ok")
        self.assertEqual(calls["count"], 3)

    def test_permanent_failures_are_not_retried(self):
        calls = {"count": 0}

        def unauthorised():
            calls["count"] += 1
            raise ProviderError("401", status=401, retryable=False)

        with patch("brainroute.providers.http.time.sleep"):
            with self.assertRaises(ProviderError):
                with_retries(unauthorised, retries=3)
        self.assertEqual(calls["count"], 1)

    def test_retries_are_bounded(self):
        calls = {"count": 0}

        def always_fails():
            calls["count"] += 1
            raise ProviderError("503", retryable=True)

        with patch("brainroute.providers.http.time.sleep"):
            with self.assertRaises(ProviderError):
                with_retries(always_fails, retries=2)
        self.assertEqual(calls["count"], 3)

    def test_rate_limits_and_server_errors_are_classified_retryable(self):
        self.assertIn(429, RETRYABLE_STATUS)
        self.assertIn(503, RETRYABLE_STATUS)
        self.assertNotIn(401, RETRYABLE_STATUS)
        self.assertNotIn(400, RETRYABLE_STATUS)


class OpenAICompatibleTests(unittest.TestCase):
    """One adapter for any endpoint speaking /v1/chat/completions."""

    MODEL = {
        "id": "gateway-model",
        "provider": "openai_compatible",
        "provider_label": "my-gateway",
        "model": "vendor/some-model",
        "endpoint": "https://gateway.example.com/v1/chat/completions",
        "api_key_env": "MY_GATEWAY_KEY",
    }

    def test_a_configured_endpoint_and_key_are_used(self):
        captured = {}

        def fake_post(url, payload, provider, headers=None, timeout=60):
            captured.update(url=url, payload=payload, headers=headers or {})
            return {"choices": [{"message": {"content": "hello"}}]}

        with patch.dict("os.environ", {"MY_GATEWAY_KEY": "sk-test"}), \
             patch("brainroute.providers.openai_compatible.post_json", side_effect=fake_post):
            self.assertEqual(generate(self.MODEL, "hi"), "hello")

        self.assertEqual(captured["url"], self.MODEL["endpoint"])
        self.assertEqual(captured["payload"]["model"], "vendor/some-model")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer sk-test")

    def test_a_missing_endpoint_is_a_clear_permanent_error(self):
        model = {**self.MODEL, "endpoint": ""}
        with patch.dict("os.environ", {"MY_GATEWAY_KEY": "sk-test"}):
            with self.assertRaises(ProviderError) as caught:
                generate(model, "hi")
        self.assertIn("endpoint", str(caught.exception))
        self.assertFalse(caught.exception.retryable)

    def test_a_missing_credential_names_the_variable_and_is_not_retryable(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderError) as caught:
                generate(self.MODEL, "hi")
        self.assertIn("MY_GATEWAY_KEY", str(caught.exception))
        self.assertFalse(caught.exception.retryable)

    def test_an_endpoint_needing_no_auth_is_supported(self):
        # Self-hosted runtimes commonly have no credential at all.
        model = {**self.MODEL, "requires_api_key": False}

        def fake_post(url, payload, provider, headers=None, timeout=60):
            assert "Authorization" not in (headers or {})
            return {"choices": [{"message": {"content": "ok"}}]}

        with patch.dict("os.environ", {}, clear=True), \
             patch("brainroute.providers.openai_compatible.post_json", side_effect=fake_post):
            self.assertEqual(generate(model, "hi"), "ok")

    def test_an_error_body_becomes_a_provider_error(self):
        with patch.dict("os.environ", {"MY_GATEWAY_KEY": "k"}), \
             patch("brainroute.providers.openai_compatible.post_json",
                   return_value={"error": {"message": "model overloaded"}}):
            with self.assertRaises(ProviderError) as caught:
                generate(self.MODEL, "hi")
        self.assertIn("model overloaded", str(caught.exception))
        self.assertIn("my-gateway", str(caught.exception))

    def test_streaming_parses_sse_deltas(self):
        lines = [
            ": keepalive",
            'data: {"choices":[{"delta":{"content":"he"}}]}',
            'data: {"choices":[{"delta":{"content":"llo"}}]}',
            "data: [DONE]",
        ]
        with patch.dict("os.environ", {"MY_GATEWAY_KEY": "k"}), \
             patch("brainroute.providers.openai_compatible.stream_lines", return_value=iter(lines)):
            self.assertEqual("".join(stream_generate(self.MODEL, "hi")), "hello")

    def test_extra_body_reaches_the_request(self):
        captured = {}

        def fake_post(url, payload, provider, headers=None, timeout=60):
            captured.update(payload)
            return {"choices": [{"message": {"content": "x"}}]}

        model = {**self.MODEL, "extra_body": {"temperature": 0.1, "top_p": 0.9}}
        with patch.dict("os.environ", {"MY_GATEWAY_KEY": "k"}), \
             patch("brainroute.providers.openai_compatible.post_json", side_effect=fake_post):
            generate(model, "hi")
        self.assertEqual(captured["temperature"], 0.1)


class DispatchTests(unittest.TestCase):
    def test_unknown_provider_raises_a_typed_error(self):
        with self.assertRaises(ProviderError):
            generate({"id": "x", "provider": "nope"}, "hi")

    def test_missing_openai_key_is_not_retryable(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(ProviderError) as caught:
                generate({"id": "x", "provider": "openai", "model": "gpt-4o"}, "hi")
        self.assertFalse(caught.exception.retryable)


if __name__ == "__main__":
    unittest.main()
