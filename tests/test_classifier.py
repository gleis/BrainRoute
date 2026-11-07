import unittest
from unittest.mock import patch

from modelroute import classifier
from modelroute.classifier import classify, classify_prompt, classify_with_model


VALID_RESPONSE = {
    "task_type": "coding",
    "complexity": "high",
    "privacy_level": "high",
    "urgency": "normal",
    "capabilities": ["tool_use"],
    "context_tokens": 1200,
    "confidence": 0.92,
    "reason": "needs code review",
}


class HeuristicTests(unittest.TestCase):
    def test_detects_task_type_from_keywords(self):
        self.assertEqual(classify("Refactor this python function").task_type, "coding")
        self.assertEqual(classify("Summarize the meeting notes").task_type, "summarization")

    def test_unrecognised_prompts_fall_back_to_general(self):
        self.assertEqual(classify("Hello there").task_type, "general")

    def test_multi_word_urgency_phrases_are_detected(self):
        # A bag-of-words match can never see these.
        self.assertEqual(classify("I need this as soon as possible please").urgency, "high")
        self.assertEqual(classify("We are in a real time crunch here").urgency, "high")

    def test_multi_word_privacy_phrases_are_detected(self):
        self.assertEqual(classify("Please do not share this outside the team").privacy_level, "high")

    def test_complexity_tracks_signals_and_length(self):
        self.assertEqual(classify("Say hi").complexity, "low")
        self.assertEqual(classify("Design a distributed architecture").complexity, "high")
        self.assertEqual(classify("x" * 2000).complexity, "high")

    def test_context_tokens_are_estimated(self):
        self.assertGreater(classify("x" * 400).context_tokens, 50)


class StructuredClassifierTests(unittest.TestCase):
    def setUp(self):
        classifier.invalidate()
        self.addCleanup(classifier.invalidate)

    def test_structured_classifier_validates_response(self):
        with patch("modelroute.classifier.call_ollama_json", return_value=VALID_RESPONSE):
            task = classify_with_model(
                "Review this confidential code.", {"id": "local", "provider": "ollama"}
            )
        self.assertEqual(task.classifier, "model:local")
        self.assertEqual(task.task_type, "coding")
        self.assertEqual(task.capabilities, ("tool_use",))
        self.assertEqual(task.context_tokens, 1200)

    def test_invalid_fields_fall_back_to_heuristic_values(self):
        garbage = {**VALID_RESPONSE, "task_type": "astrology", "complexity": 9, "confidence": "x"}
        with patch("modelroute.classifier.call_ollama_json", return_value=garbage):
            task = classify_with_model("Refactor this python module", {"id": "m", "provider": "ollama"})
        self.assertEqual(task.task_type, "coding")
        self.assertIn(task.complexity, {"low", "medium", "high"})
        self.assertEqual(task.confidence, 0.5)

    def test_a_missed_privacy_signal_is_restored_from_the_heuristic(self):
        # Under-detecting privacy is the costlier error, so the heuristic wins.
        lax = {**VALID_RESPONSE, "privacy_level": "normal"}
        with patch("modelroute.classifier.call_ollama_json", return_value=lax):
            task = classify_with_model("Handle this confidential patient record", {"id": "m", "provider": "ollama"})
        self.assertEqual(task.privacy_level, "high")

    def test_repeat_prompts_are_served_from_cache(self):
        with patch("modelroute.classifier.call_ollama_json", return_value=VALID_RESPONSE) as call:
            model = {"id": "local", "provider": "ollama"}
            classify_with_model("same prompt", model)
            classify_with_model("same prompt", model)
        self.assertEqual(call.call_count, 1)

    def test_classifier_failure_degrades_to_the_heuristic(self):
        with patch("modelroute.classifier.call_ollama_json", side_effect=RuntimeError("offline")):
            task = classify_prompt("Refactor this python module", {"id": "m", "provider": "ollama"})
        self.assertEqual(task.classifier, "heuristic")
        self.assertEqual(task.task_type, "coding")
        self.assertIn("router model unavailable", task.detail)


if __name__ == "__main__":
    unittest.main()
