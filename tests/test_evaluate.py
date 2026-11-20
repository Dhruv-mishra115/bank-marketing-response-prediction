import unittest

from bank_marketing.evaluate import evaluate_probabilities


class EvaluationTests(unittest.TestCase):
    def test_metrics_and_threshold(self):
        result = evaluate_probabilities(["no", "yes", "no", "yes"], [0.1, 0.8, 0.2, 0.7], 0.5)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_probabilities(["no"], [1.2])


if __name__ == "__main__":
    unittest.main()
