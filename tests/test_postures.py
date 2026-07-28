import unittest
from unittest.mock import patch

from brainroute import catalog, postures
from brainroute.config import load_config
from brainroute.router import route
from brainroute.settings import DEFAULT_SETTINGS, load_settings, override


HARD = "Design a complex production architecture for a multi-provider routing layer"
EASY = "Rename these three variables."
SENSITIVE = "Summarize this confidential patient note."


class PostureDefinitionTests(unittest.TestCase):
    def test_the_default_posture_is_privacy_first(self):
        self.assertEqual(DEFAULT_SETTINGS["policy"]["posture"], "local_first")
        self.assertTrue(DEFAULT_SETTINGS["policy"]["prefer_local"])

    def test_unknown_postures_fall_back_to_the_default(self):
        self.assertEqual(postures.resolve("nonsense"), postures.DEFAULT_POSTURE)
        self.assertEqual(postures.resolve(None), postures.DEFAULT_POSTURE)

    def test_only_air_gapped_refuses_per_request_overrides(self):
        self.assertFalse(postures.cloud_override_allowed("air_gapped"))
        for name in ("open", "local_first", "local_only"):
            self.assertTrue(postures.cloud_override_allowed(name), name)

    def test_no_posture_ever_allows_cloud_for_sensitive_prompts(self):
        for name, definition in postures.POSTURES.items():
            policy = definition["settings"]["policy"]
            self.assertFalse(policy["allow_cloud_for_private"], name)

    def test_presets_are_copies_not_shared_state(self):
        first = postures.preset("open")
        first["policy"]["allow_cloud"] = "mutated"
        self.assertTrue(postures.preset("open")["policy"]["allow_cloud"])

    def test_every_posture_is_described_for_the_ui(self):
        described = {item["id"] for item in postures.describe()}
        self.assertEqual(described, set(postures.POSTURES))


class RemoteDiscoveryTests(unittest.TestCase):
    """Discovery is outbound traffic even though it carries no prompt data."""

    def setUp(self):
        catalog.invalidate()
        self.addCleanup(catalog.invalidate)

    GATEWAY = "https://gateway.example.com/v1/models"

    def _contacted(self, posture, enabled=True):
        urls = []

        def spy(url, timeout, headers=None):
            urls.append(url)
            if url == self.GATEWAY:
                return {"data": []}
            raise RuntimeError("no local runtime in tests")

        scope = {
            "policy": {"posture": posture},
            "catalog": {"remote_sources": [{
                "id": "gateway",
                "enabled": enabled,
                "models_url": self.GATEWAY,
                "chat_url": "https://gateway.example.com/v1/chat/completions",
            }]},
        }
        with override(scope), patch("brainroute.catalog._read_json", side_effect=spy):
            result = catalog.discover_models(force_refresh=True)
        return [url for url in urls if url == self.GATEWAY], result

    def test_remote_catalog_is_fetched_when_enabled(self):
        contacted, _ = self._contacted("local_first")
        self.assertTrue(contacted)

    def test_local_only_still_allows_listing_remote_models(self):
        # Listing is how you find a model to opt in to; it sends no prompt data.
        contacted, _ = self._contacted("local_only")
        self.assertTrue(contacted)

    def test_air_gapped_blocks_the_remote_catalog_even_when_pinned_on(self):
        # Regression: the posture preset alone was overridden by a pinned
        # catalog setting, so air-gapped still phoned home.
        contacted, result = self._contacted("air_gapped", enabled=True)
        self.assertEqual(contacted, [])
        source = next(item for item in result["sources"] if item["id"] == "gateway")
        self.assertFalse(source["ok"])
        self.assertIn("air_gapped", source["detail"])

    def test_nothing_is_fetched_when_the_catalog_is_off(self):
        contacted, _ = self._contacted("open", enabled=False)
        self.assertEqual(contacted, [])


class PostureRoutingTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def _route(self, prompt, posture, profile="balanced", **kwargs):
        with override({"policy": {"posture": posture}}):
            return route(prompt, self.config, profile_name=profile, **kwargs)

    def test_sensitive_prompts_stay_local_under_every_posture(self):
        for name in postures.POSTURES:
            decision = self._route(SENSITIVE, name)
            self.assertTrue(decision.selected.model["local"], name)

    def test_local_first_escalates_hard_work_and_says_so(self):
        decision = self._route(HARD, "local_first")
        self.assertFalse(decision.selected.model["local"])
        self.assertIn("privacy_escalation", [item.code for item in decision.warnings])

    def test_local_first_keeps_easy_work_local_without_warning(self):
        decision = self._route(EASY, "local_first")
        self.assertTrue(decision.selected.model["local"])
        self.assertEqual(decision.warnings, ())

    def test_local_only_keeps_even_hard_work_local(self):
        decision = self._route(HARD, "local_only")
        self.assertTrue(decision.selected.model["local"])

    def test_air_gapped_ignores_a_cloud_opt_in(self):
        decision = self._route(HARD, "air_gapped", overrides={"allow_cloud": True})
        self.assertTrue(decision.selected.model["local"])
        self.assertFalse(decision.policy.cloud_override_applied)

    def test_local_only_warns_when_an_opt_in_is_used(self):
        decision = self._route(HARD, "local_only", overrides={"allow_cloud": True})
        self.assertIn("cloud_override", [item.code for item in decision.warnings])

    def test_open_posture_lets_capability_win(self):
        decision = self._route(HARD, "open")
        self.assertFalse(decision.selected.model["local"])


class CapabilityFloorConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_lowering_the_floor_keeps_hard_work_local(self):
        with override({"routing": {"capability_floor": {"high": 0.60}}}):
            decision = route(HARD, self.config, profile_name="balanced")
        self.assertTrue(decision.selected.model["local"])

    def test_raising_the_floor_escalates_more_work(self):
        # Long enough to classify as medium complexity, with no signals that
        # would push it to high.
        medium = (
            "Write the release notes for this week's payments service update, "
            "covering the new invoice export option, the updated currency "
            "picker, and the two bug fixes that shipped alongside them."
        )
        with override({"routing": {"capability_floor": {"medium": 0.90}}}):
            decision = route(medium, self.config, profile_name="balanced")
        self.assertEqual(decision.task.complexity, "medium")
        self.assertFalse(decision.selected.model["local"])

    def test_the_escalation_warning_can_be_silenced(self):
        with override({"routing": {"warn_on_privacy_escalation": False}}):
            decision = route(HARD, self.config, profile_name="balanced")
        self.assertNotIn("privacy_escalation", [item.code for item in decision.warnings])


class SettingsLayeringTests(unittest.TestCase):
    def test_a_posture_supplies_flags_the_user_has_not_pinned(self):
        with override({"policy": {"posture": "local_only"}}):
            self.assertFalse(load_settings()["policy"]["allow_cloud"])

    def test_an_explicit_flag_survives_a_posture_switch(self):
        # Postures are a starting point, not a mode you are trapped in.
        with override({"policy": {"posture": "local_only", "allow_cloud": True}}):
            self.assertTrue(load_settings()["policy"]["allow_cloud"])

    def test_derived_posture_facts_are_exposed(self):
        with override({"policy": {"posture": "air_gapped"}}):
            policy = load_settings()["policy"]
        self.assertEqual(policy["posture"], "air_gapped")
        self.assertFalse(policy["cloud_override_allowed"])

    def test_overrides_do_not_leak_after_the_block(self):
        with override({"policy": {"posture": "air_gapped"}}):
            pass
        self.assertEqual(load_settings()["policy"]["posture"], postures.DEFAULT_POSTURE)


if __name__ == "__main__":
    unittest.main()
