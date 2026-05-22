import unittest

from brainroute.config import load_config
from brainroute.router import route
from brainroute.simple_yaml import load_simple_yaml


class RouterTests(unittest.TestCase):
    def test_loads_yaml_config(self):
        data = load_simple_yaml("config/models.yaml")
        self.assertGreaterEqual(len(data["models"]), 3)
        self.assertEqual(data["models"][0]["id"], "local-qwen3")

    def test_private_profile_prefers_local_model(self):
        config = load_config()
        decision = route("Summarize this confidential patient note", config, profile_name="private")
        self.assertEqual(decision.selected.model["id"], "local-qwen3")
        self.assertEqual(decision.task.privacy_level, "high")

    def test_urgent_profile_prefers_fast_model(self):
        config = load_config()
        decision = route("Quickly write a short status update", config, profile_name="urgent")
        self.assertEqual(decision.selected.model["id"], "gpt-5.4-mini")

    def test_json_shape_contains_fallback(self):
        config = load_config()
        decision = route("Plan the architecture for a model router", config)
        payload = decision.as_dict()
        self.assertIn("recommended_model", payload)
        self.assertIn("fallback_model", payload)
        self.assertGreaterEqual(len(payload["ranked"]), 2)


if __name__ == "__main__":
    unittest.main()
