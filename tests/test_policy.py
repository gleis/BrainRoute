import unittest
from unittest.mock import patch

from modelroute.classifier import TaskProfile, classify
from modelroute.policy import apply_policy
from modelroute.reliability import OPEN, ModelReliability
from modelroute.settings import override


BASE_POLICY = {
    "allow_cloud": True,
    "allow_cloud_for_private": False,
    "max_estimated_cost_usd": 1.0,
    "monthly_budget_usd": 10.0,
    "prefer_local": False,
    "context_headroom_tokens": 512,
}

LOCAL = {"id": "local", "local": True, "enabled": True, "max_context": 32000}
CLOUD = {"id": "cloud", "local": False, "enabled": True, "max_context": 128000,
         "model": "gpt-4o", "input_price_per_mtok": 2.5, "output_price_per_mtok": 10.0}


def task(**overrides):
    base = {
        "task_type": "general", "complexity": "medium", "privacy_level": "normal",
        "urgency": "normal", "signals": (), "context_tokens": 100,
    }
    base.update(overrides)
    return TaskProfile(**base)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        settings = patch("modelroute.policy.load_settings", return_value={"policy": dict(BASE_POLICY)})
        spend = patch("modelroute.policy.spend_since", return_value=0)
        health = patch("modelroute.policy.snapshot", return_value={})
        for patcher in (settings, spend, health):
            self.addCleanup(patcher.stop)
        self.settings = settings.start()
        self.spend = spend.start()
        self.health = health.start()

    def _policy(self, **overrides):
        self.settings.return_value = {"policy": {**BASE_POLICY, **overrides}}

    def test_private_policy_rejects_cloud(self):
        result = apply_policy([LOCAL, CLOUD], classify("private patient note"), "private patient note")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertEqual(result.rejected[0]["id"], "cloud")

    def test_cloud_can_be_disabled_outright(self):
        self._policy(allow_cloud=False)
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])

    def test_models_that_cannot_hold_the_prompt_are_rejected(self):
        # 100k tokens fits the cloud model's window but not the local one's.
        result = apply_policy([LOCAL, CLOUD], task(context_tokens=100_000), "x")
        self.assertEqual([item["id"] for item in result.allowed], ["cloud"])
        self.assertIn("context window too small", result.rejected[0]["reason"])

    def test_per_request_budget_rejects_an_expensive_call(self):
        # 20k tokens fits both windows, so cost is the only thing separating them.
        self._policy(max_estimated_cost_usd=0.01)
        result = apply_policy([LOCAL, CLOUD], task(context_tokens=20_000), "x")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertIn("per-request budget", result.rejected[0]["reason"])

    def test_monthly_budget_stops_cloud_spend(self):
        self.spend.return_value = 99.0
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertIn("monthly budget reached", result.rejected[0]["reason"])

    def test_an_open_circuit_removes_a_model_from_consideration(self):
        self.health.return_value = {
            "cloud": ModelReliability("cloud", state=OPEN, detail="circuit open")
        }
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])

    def test_a_total_outage_re_admits_unhealthy_models_rather_than_stranding_the_request(self):
        # If every provider has tripped, returning no route at all is worse
        # than trying a degraded one.
        self.health.return_value = {
            "local": ModelReliability("local", state=OPEN, detail="down"),
            "cloud": ModelReliability("cloud", state=OPEN, detail="down"),
        }
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        self.assertEqual(sorted(item["id"] for item in result.allowed), ["cloud", "local"])

    def test_prefer_local_adds_a_scoring_bonus(self):
        self._policy(prefer_local=True, local_bonus=0.15)
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        allowed = {item["id"]: item for item in result.allowed}
        self.assertEqual(allowed["local"]["policy_score_bonus"], 0.15)
        self.assertNotIn("policy_score_bonus", allowed["cloud"])

    def test_the_local_bonus_is_configurable(self):
        self._policy(prefer_local=True, local_bonus=0.4)
        result = apply_policy([LOCAL, CLOUD], task(), "hello")
        allowed = {item["id"]: item for item in result.allowed}
        self.assertEqual(allowed["local"]["policy_score_bonus"], 0.4)


class CloudOverrideTests(unittest.TestCase):
    """Per-request opt-in to cloud, and the postures that refuse it."""

    def setUp(self):
        spend = patch("modelroute.policy.spend_since", return_value=0)
        health = patch("modelroute.policy.snapshot", return_value={})
        for patcher in (spend, health):
            self.addCleanup(patcher.stop)
            patcher.start()

    def _apply(self, posture, task_profile=None, overrides=None):
        with override({"policy": {"posture": posture}}):
            return apply_policy(
                [LOCAL, CLOUD], task_profile or task(), "hello", overrides=overrides
            )

    def test_local_only_blocks_cloud_by_default(self):
        result = self._apply("local_only")
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertFalse(result.cloud_override_applied)

    def test_local_only_honours_a_per_request_opt_in(self):
        result = self._apply("local_only", overrides={"allow_cloud": True})
        self.assertIn("cloud", [item["id"] for item in result.allowed])
        self.assertTrue(result.cloud_override_applied)

    def test_air_gapped_refuses_the_opt_in(self):
        # A posture a single request can switch off is not a guarantee.
        result = self._apply("air_gapped", overrides={"allow_cloud": True})
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertFalse(result.cloud_override_applied)

    def test_the_opt_in_does_not_extend_to_sensitive_prompts(self):
        # Allowing cloud for a request is a different decision from consenting
        # to send confidential data off the machine.
        result = self._apply(
            "local_only", task_profile=task(privacy_level="high"), overrides={"allow_cloud": True}
        )
        self.assertEqual([item["id"] for item in result.allowed], ["local"])

    def test_open_posture_needs_no_override(self):
        result = self._apply("open", overrides={"allow_cloud": True})
        self.assertIn("cloud", [item["id"] for item in result.allowed])
        self.assertFalse(result.cloud_override_applied)


if __name__ == "__main__":
    unittest.main()
