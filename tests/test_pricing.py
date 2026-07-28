import unittest

from brainroute.pricing import apply_pricing, cost_score, estimate_request_cost, lookup_price


class PricingTests(unittest.TestCase):
    def test_cost_score_is_ordered_by_price(self):
        cheap = cost_score({"model": "gpt-4o-mini"})
        mid = cost_score({"model": "claude-sonnet-4"})
        dear = cost_score({"model": "claude-opus-4"})
        self.assertGreater(cheap, mid)
        self.assertGreater(mid, dear)

    def test_local_models_are_free_and_score_top(self):
        self.assertEqual(cost_score({"local": True}), 1.0)

    def test_unknown_pricing_is_mid_market_not_free(self):
        # An unpriced model must not win the cheap profile by default.
        self.assertEqual(cost_score({"model": "mystery-model", "local": False}), 0.5)

    def test_per_token_prices_are_scaled_to_per_million(self):
        price = lookup_price({"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015})
        self.assertEqual(price, (3.0, 15.0))

    def test_apply_pricing_fills_both_representations(self):
        model = apply_pricing({"id": "gpt-4o-mini", "model": "gpt-4o-mini", "local": False})
        self.assertTrue(model["pricing_known"])
        self.assertAlmostEqual(model["input_cost_per_token"], 0.15 / 1_000_000)
        self.assertGreater(model["cost_score"], 0.5)

    def test_apply_pricing_respects_an_explicit_cost_score(self):
        model = apply_pricing({"id": "x", "model": "gpt-4o", "cost_score": 0.11})
        self.assertEqual(model["cost_score"], 0.11)

    def test_request_cost_includes_the_completion(self):
        model = {"model": "gpt-4o-mini"}
        input_only = estimate_request_cost(model, 1_000_000, output_tokens=0)
        with_output = estimate_request_cost(model, 1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(input_only, 0.15)
        self.assertAlmostEqual(with_output, 0.75)


if __name__ == "__main__":
    unittest.main()
