import unittest
from unittest.mock import patch

from brainroute.classifier import classify
from brainroute.policy import apply_policy


class PolicyTests(unittest.TestCase):
    @patch("brainroute.policy.spend_since", return_value=0)
    @patch("brainroute.policy.load_settings")
    def test_private_policy_rejects_cloud(self, load_settings, _spend):
        load_settings.return_value = {
            "policy": {
                "allow_cloud": True,
                "allow_cloud_for_private": False,
                "max_estimated_cost_usd": 1,
                "monthly_budget_usd": 10,
            }
        }
        result = apply_policy(
            [{"id": "local", "local": True, "enabled": True}, {"id": "cloud", "local": False, "enabled": True}],
            classify("private patient note"),
            "private patient note",
        )
        self.assertEqual([item["id"] for item in result.allowed], ["local"])
        self.assertEqual(result.rejected[0]["id"], "cloud")


if __name__ == "__main__":
    unittest.main()
