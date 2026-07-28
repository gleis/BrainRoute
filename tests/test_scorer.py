import unittest
from unittest.mock import patch

from brainroute.classifier import TaskProfile
from brainroute.config import normalize_profile
from brainroute.reliability import ModelReliability
from brainroute.scorer import effective_weights, score_models


def task(**overrides):
    base = {
        "task_type": "general",
        "complexity": "medium",
        "privacy_level": "normal",
        "urgency": "normal",
        "signals": (),
        "context_tokens": 100,
    }
    base.update(overrides)
    return TaskProfile(**base)


SMALL_LOCAL = {
    "id": "small-local", "local": True, "enabled": True,
    "quality_score": 0.60, "speed_score": 0.85, "cost_score": 1.0, "privacy_score": 1.0,
    "strengths": ["reasoning"], "max_context": 32000,
}
FRONTIER = {
    "id": "frontier", "local": False, "enabled": True,
    "quality_score": 0.95, "speed_score": 0.50, "cost_score": 0.30, "privacy_score": 0.35,
    "strengths": ["reasoning"], "max_context": 200000,
}
BALANCED = {"quality": 0.35, "speed": 0.25, "cost": 0.25, "privacy": 0.15, "task_fit": 0.20}


class WeightTests(unittest.TestCase):
    def test_profile_weights_are_normalised_to_one(self):
        weights = normalize_profile(BALANCED)
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_weights_stay_normalised_after_task_tilting(self):
        weights = effective_weights(BALANCED, task(complexity="high"))
        self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_high_complexity_shifts_weight_to_quality(self):
        flat = effective_weights(BALANCED, task(complexity="medium"))
        hard = effective_weights(BALANCED, task(complexity="high"))
        self.assertGreater(hard["quality"], flat["quality"])
        self.assertLess(hard["cost"], flat["cost"])

    def test_urgency_shifts_weight_to_speed(self):
        calm = effective_weights(BALANCED, task(urgency="normal"))
        rushed = effective_weights(BALANCED, task(urgency="high"))
        self.assertGreater(rushed["speed"], calm["speed"])

    def test_empty_profile_falls_back_to_uniform_weights(self):
        weights = normalize_profile({})
        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertEqual(len(set(weights.values())), 1)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("brainroute.scorer.snapshot", return_value={})
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_capability_floor_blocks_a_weak_model_on_a_hard_task(self):
        # On an easy task the cheap local model wins on cost and privacy.
        easy = score_models([SMALL_LOCAL, FRONTIER], BALANCED, task(complexity="low"))
        self.assertEqual(easy[0].model["id"], "small-local")
        # The same trade-off must not survive a genuinely hard task.
        hard = score_models([SMALL_LOCAL, FRONTIER], BALANCED, task(complexity="high"))
        self.assertEqual(hard[0].model["id"], "frontier")
        penalties = [item.name for item in hard[-1].adjustments]
        self.assertIn("capability_floor", penalties)

    def test_sensitive_prompts_penalise_external_providers(self):
        ranked = score_models([SMALL_LOCAL, FRONTIER], BALANCED, task(privacy_level="high"))
        names = {item.model["id"]: [adj.name for adj in item.adjustments] for item in ranked}
        self.assertIn("sensitive_local", names["small-local"])
        self.assertIn("sensitive_external", names["frontier"])
        self.assertEqual(ranked[0].model["id"], "small-local")

    def test_context_pressure_penalises_a_nearly_full_window(self):
        ranked = score_models([SMALL_LOCAL], BALANCED, task(context_tokens=30_000))
        self.assertIn("context_pressure", [item.name for item in ranked[0].adjustments])

    def test_scores_are_bounded_and_explainable(self):
        ranked = score_models([SMALL_LOCAL, FRONTIER], BALANCED, task())
        for item in ranked:
            self.assertGreaterEqual(item.score, 0.0)
            self.assertLessEqual(item.score, 1.0)
            explanation = item.explain()
            self.assertAlmostEqual(
                sum(explanation["contributions"].values()), explanation["base_score"], places=3
            )

    def test_disabled_models_are_never_scored(self):
        ranked = score_models([{**SMALL_LOCAL, "enabled": False}, FRONTIER], BALANCED, task())
        self.assertEqual([item.model["id"] for item in ranked], ["frontier"])

    def test_ranking_is_deterministic_for_tied_scores(self):
        twin = {**SMALL_LOCAL, "id": "aaa-twin"}
        ranked = score_models([SMALL_LOCAL, twin], BALANCED, task())
        self.assertEqual([item.model["id"] for item in ranked], ["aaa-twin", "small-local"])


class TelemetryBlendTests(unittest.TestCase):
    def test_measured_latency_overrides_an_optimistic_speed_claim(self):
        slow = ModelReliability(
            model_id="frontier", samples=50, ok_rate=1.0, p95_latency_ms=30_000
        )
        with patch("brainroute.scorer.snapshot", return_value={"frontier": slow}):
            ranked = score_models([FRONTIER], BALANCED, task())
        self.assertLess(ranked[0].factors["speed"], FRONTIER["speed_score"])

    def test_observed_failures_penalise_the_score(self):
        flaky = ModelReliability(model_id="frontier", samples=20, ok_rate=0.5)
        with patch("brainroute.scorer.snapshot", return_value={"frontier": flaky}):
            ranked = score_models([FRONTIER], BALANCED, task())
        self.assertIn("observed_failures", [item.name for item in ranked[0].adjustments])

    def test_negative_feedback_lowers_the_quality_factor(self):
        disliked = ModelReliability(
            model_id="frontier", samples=20, ok_rate=1.0, feedback_score=0.1, feedback_votes=20
        )
        with patch("brainroute.scorer.snapshot", return_value={"frontier": disliked}):
            ranked = score_models([FRONTIER], BALANCED, task())
        self.assertLess(ranked[0].factors["quality"], FRONTIER["quality_score"])


if __name__ == "__main__":
    unittest.main()
