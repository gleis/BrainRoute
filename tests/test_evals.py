import unittest

from brainpicker.config import load_config
from brainpicker.evals import run_evals


class EvalTests(unittest.TestCase):
    def test_routing_evals_pass(self):
        results = run_evals(load_config())
        self.assertTrue(results)
        self.assertTrue(all(result.passed for result in results), results)


if __name__ == "__main__":
    unittest.main()

