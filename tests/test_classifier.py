import unittest
from unittest.mock import patch

from brainroute.classifier import classify_with_model


class ClassifierTests(unittest.TestCase):
    def test_structured_classifier_validates_response(self):
        with patch("brainroute.classifier.call_ollama_json") as call:
            call.return_value = {
                "task_type": "coding",
                "complexity": "high",
                "privacy_level": "high",
                "urgency": "normal",
                "capabilities": ["tool_use"],
                "context_tokens": 1200,
                "confidence": 0.92,
                "reason": "needs code review",
            }
            task = classify_with_model("Review this confidential code.", {"id": "local", "provider": "ollama"})
        self.assertEqual(task.classifier, "model:local")
        self.assertEqual(task.task_type, "coding")
        self.assertEqual(task.capabilities, ("tool_use",))
        self.assertEqual(task.context_tokens, 1200)


if __name__ == "__main__":
    unittest.main()
