import unittest
from unittest.mock import patch

from modelroute.config import load_config
from modelroute.router import NoRouteAvailable, route
from modelroute.settings import override
from modelroute.simple_yaml import load_simple_yaml


class RouterTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_loads_yaml_config(self):
        data = load_simple_yaml("config/models.yaml")
        self.assertGreaterEqual(len(data["models"]), 3)
        self.assertEqual(data["models"][0]["id"], "local-qwen2-5")

    def test_private_profile_prefers_local_model(self):
        decision = route("Summarize this confidential patient note", self.config, profile_name="private")
        self.assertEqual(decision.selected.model["id"], "local-qwen2-5")
        self.assertEqual(decision.task.privacy_level, "high")

    def test_urgent_profile_prefers_fast_model_when_privacy_is_not_weighted(self):
        with override({"policy": {"posture": "open"}}):
            decision = route("Quickly write a short status update", self.config, profile_name="urgent")
        self.assertEqual(decision.selected.model["id"], "gpt-5.4-mini")

    def test_urgent_work_stays_local_under_the_default_posture(self):
        # The speed gap between a local model and fast cloud is small; under
        # local-first that is not worth leaving the machine for.
        decision = route("Quickly write a short status update", self.config, profile_name="urgent")
        self.assertEqual(decision.selected.model["id"], "local-qwen2-5")

    def test_complex_work_prefers_a_capable_model(self):
        decision = route(
            "Design a complex production architecture for a multi-provider routing layer",
            self.config,
        )
        self.assertEqual(decision.selected.model["id"], "premium-reasoner")

    def test_json_shape_contains_fallback(self):
        payload = route("Plan the architecture for a model router", self.config).as_dict()
        self.assertIn("recommended_model", payload)
        self.assertIn("fallback_model", payload)
        self.assertGreaterEqual(len(payload["ranked"]), 2)

    def test_decision_carries_a_request_id_and_margin(self):
        decision = route("Write a haiku", self.config)
        self.assertTrue(decision.request_id)
        self.assertIsInstance(decision.margin, float)
        self.assertIn("explain", decision.as_dict())

    def test_a_supplied_request_id_is_preserved(self):
        decision = route("Write a haiku", self.config, request_id="abc123")
        self.assertEqual(decision.as_dict()["request_id"], "abc123")

    def test_fallback_chain_excludes_the_selected_model(self):
        decision = route("Write a haiku", self.config)
        chain = [item.model["id"] for item in decision.fallback_chain]
        self.assertNotIn(decision.selected.model["id"], chain)

    def test_unknown_profile_is_rejected_with_a_helpful_message(self):
        with self.assertRaises(ValueError) as caught:
            route("hello", self.config, profile_name="nonsense")
        self.assertIn("Available:", str(caught.exception))

    def test_no_admissible_model_raises_a_typed_error(self):
        with patch("modelroute.router.apply_policy") as policy:
            policy.return_value = type(
                "R", (), {"allowed": (), "rejected": ({"id": "x", "reason": "disabled"},),
                          "settings": {}, "monthly_spend_usd": 0.0}
            )()
            with self.assertRaises(NoRouteAvailable):
                route("hello", self.config)

    def test_exploration_is_off_by_default(self):
        first = route("Write a haiku", self.config)
        for _ in range(10):
            self.assertEqual(route("Write a haiku", self.config).selected.model["id"],
                             first.selected.model["id"])
            self.assertFalse(first.exploration)

    def test_exploration_can_promote_a_near_tied_candidate(self):
        settings = {
            "classifier": {"enabled": False, "model_id": ""},
            "routing": {"exploration_rate": 1.0, "exploration_score_margin": 1.0,
                        "max_fallback_depth": 2},
        }
        with patch("modelroute.router.load_settings", return_value=settings), \
             patch("modelroute.router.random.random", return_value=0.0), \
             patch("modelroute.router.random.choice", side_effect=lambda options: options[0]):
            decision = route("Write a haiku", self.config)
        self.assertTrue(decision.exploration)


if __name__ == "__main__":
    unittest.main()
