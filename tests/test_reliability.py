import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from modelroute import reliability
from modelroute.reliability import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    blend,
    observed_speed_score,
    snapshot,
)


def runs(consecutive_failures, *, seconds_ago=1, total=10, ok_rate=0.5):
    last_failure = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return {
        "flaky": {
            "runs": total,
            "successes": int(total * ok_rate),
            "ok_rate": ok_rate,
            "p50_latency_ms": 900,
            "p95_latency_ms": 2400,
            "avg_cost_usd": 0.001,
            "cost_total": 0.01,
            "output_tokens_total": 500,
            "consecutive_failures": consecutive_failures,
            "last_failure_at": last_failure.isoformat(),
            "last_run_at": last_failure.isoformat(),
        }
    }


class CircuitBreakerTests(unittest.TestCase):
    def setUp(self):
        reliability.invalidate()
        self.addCleanup(reliability.invalidate)

    def _snapshot(self, stats):
        with patch("modelroute.reliability.model_stats", return_value=stats), \
             patch("modelroute.reliability.feedback_stats", return_value={}):
            return snapshot(force_refresh=True)

    def test_circuit_stays_closed_below_the_failure_threshold(self):
        entry = self._snapshot(runs(2))["flaky"]
        self.assertEqual(entry.state, CLOSED)
        self.assertTrue(entry.available)

    def test_circuit_opens_after_repeated_failures(self):
        entry = self._snapshot(runs(3, seconds_ago=1))["flaky"]
        self.assertEqual(entry.state, OPEN)
        self.assertFalse(entry.available)

    def test_circuit_half_opens_once_the_cooldown_elapses(self):
        entry = self._snapshot(runs(5, seconds_ago=600))["flaky"]
        self.assertEqual(entry.state, HALF_OPEN)
        self.assertTrue(entry.available)

    def test_snapshot_is_cached_between_calls(self):
        with patch("modelroute.reliability.model_stats", return_value={}) as stats, \
             patch("modelroute.reliability.feedback_stats", return_value={}):
            snapshot(force_refresh=True)
            snapshot()
            snapshot()
        self.assertEqual(stats.call_count, 1)


class BlendingTests(unittest.TestCase):
    def test_speed_score_falls_as_latency_rises(self):
        self.assertGreater(observed_speed_score(500), observed_speed_score(5_000))
        self.assertGreater(observed_speed_score(5_000), observed_speed_score(60_000))

    def test_blend_ignores_observation_without_evidence(self):
        self.assertAlmostEqual(blend(0.9, 0.1, confidence=0.0), 0.9)

    def test_blend_moves_toward_observation_with_evidence(self):
        self.assertLess(blend(0.9, 0.1, confidence=1.0), 0.9)


if __name__ == "__main__":
    unittest.main()
