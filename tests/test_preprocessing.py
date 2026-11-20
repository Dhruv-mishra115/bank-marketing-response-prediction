import unittest

import pandas as pd

from bank_marketing.preprocessing import chronological_split, select_features


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "age": range(10), "job": ["a", "b"] * 5,
            "duration": range(10), "campaign": range(10),
            "y": ["no", "yes"] * 5,
        })

    def test_feature_exclusions_are_kept(self):
        X, y = select_features(self.frame)
        self.assertEqual(list(X.columns), ["age", "job"])
        self.assertEqual(y.tolist(), self.frame["y"].tolist())

    def test_chronological_boundaries_and_sizes(self):
        splits = chronological_split(self.frame)
        self.assertEqual(len(splits.X_train), 6)
        self.assertEqual(len(splits.X_validation), 2)
        self.assertEqual(len(splits.X_test), 2)
        self.assertEqual(splits.X_train.index.tolist(), list(range(6)))
        self.assertEqual(splits.X_validation.index.tolist(), [6, 7])
        self.assertEqual(splits.X_test.index.tolist(), [8, 9])

    def test_invalid_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            chronological_split(self.frame, 0.8, 0.2)


if __name__ == "__main__":
    unittest.main()
