import unittest
from unittest.mock import patch

from modelroute.config import load_config
from modelroute.execution import execute_with_fallback
from modelroute.providers import ProviderError
from modelroute.router import route


class FallbackTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.decision = route("Plan the architecture for a model router", self.config)
        # Run records are global routing signal. Capturing them here keeps
        # simulated provider outages from teaching the real router that these
        # models are unhealthy.
        recorder = patch("modelroute.execution.record_run")
        self.addCleanup(recorder.stop)
        self.record_run = recorder.start()

    def test_every_attempt_is_recorded_as_routing_evidence(self):
        with patch("modelroute.execution.generate", side_effect=ProviderError("down")):
            execute_with_fallback(self.decision.selected.model, self.decision, "prompt")
        recorded = [call.args[0] for call in self.record_run.call_args_list]
        self.assertEqual(len(recorded), 1 + len(self.decision.fallback_chain))
        self.assertTrue(all(run["ok"] is False for run in recorded))
        self.assertTrue(all(run["latency_ms"] >= 0 for run in recorded))

    def test_a_successful_run_records_tokens_and_cost(self):
        with patch("modelroute.execution.generate", return_value="hello world"):
            execute_with_fallback(self.decision.selected.model, self.decision, "prompt")
        run = self.record_run.call_args_list[0].args[0]
        self.assertTrue(run["ok"])
        self.assertGreater(run["input_tokens"], 0)
        self.assertGreaterEqual(run["estimated_cost_usd"], 0)

    def test_the_first_working_model_is_used(self):
        with patch("modelroute.execution.generate", return_value="done") as generate:
            result = execute_with_fallback(
                self.decision.selected.model, self.decision, "prompt"
            )
        self.assertEqual(result.output, "done")
        self.assertIsNone(result.error)
        self.assertIsNone(result.fallback_attempted)
        self.assertEqual(generate.call_count, 1)

    def test_execution_walks_the_whole_fallback_chain(self):
        # Fail every model except the last one in the chain.
        working = self.decision.fallback_chain[-1].model["id"]

        def maybe(model, *args, **kwargs):
            if model["id"] != working:
                raise ProviderError("provider down", retryable=False)
            return "recovered"

        with patch("modelroute.execution.generate", side_effect=maybe):
            result = execute_with_fallback(
                self.decision.selected.model, self.decision, "prompt"
            )
        self.assertEqual(result.output, "recovered")
        self.assertEqual(result.model["id"], working)
        self.assertEqual(result.fallback_attempted, working)
        self.assertTrue(any(not item.ok for item in result.attempts))

    def test_total_failure_reports_every_attempt(self):
        with patch("modelroute.execution.generate", side_effect=ProviderError("down")):
            result = execute_with_fallback(
                self.decision.selected.model, self.decision, "prompt"
            )
        self.assertEqual(result.output, "")
        self.assertIsNotNone(result.error)
        self.assertFalse(result.ok)
        self.assertEqual(len(result.attempts), 1 + len(self.decision.fallback_chain))
        for attempt in result.attempts:
            self.assertFalse(attempt.ok)

    def test_messages_reach_the_provider_unflattened(self):
        conversation = [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hello"},
        ]
        with patch("modelroute.execution.generate", return_value="ok") as generate:
            execute_with_fallback(self.decision.selected.model, self.decision, conversation)
        sent = generate.call_args[0][1]
        self.assertEqual(sent, conversation)


if __name__ == "__main__":
    unittest.main()
