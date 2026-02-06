import unittest

from src.evaluate import average_precision_at_k, ndcg_at_k, precision_at_k, recall_at_k, rolling_splits
from src.ontology import Topic, TopicOntology


class TestOntology(unittest.TestCase):
    def test_canonicalization_matches_synonym(self):
        ontology = TopicOntology([Topic(topic_id="T1", name="heart failure", synonyms=["cardiac failure"])])
        matches = ontology.match_topics("Cardiac failure symptoms", max_topics=1)
        self.assertEqual(matches[0][0], "T1")


class TestMetrics(unittest.TestCase):
    def test_metrics_values(self):
        predicted = ["A", "B", "C", "D"]
        gold = ["B", "D"]
        self.assertAlmostEqual(recall_at_k(predicted, gold, 3), 0.5)
        self.assertAlmostEqual(precision_at_k(predicted, gold, 3), 1 / 3)
        self.assertAlmostEqual(average_precision_at_k(predicted, gold, 4), (1 / 2 + 2 / 4) / 2)
        self.assertGreater(ndcg_at_k(predicted, gold, 4), 0)


class TestSplits(unittest.TestCase):
    def test_rolling_splits(self):
        splits = rolling_splits([2022, 2023, 2024, 2025])
        self.assertEqual(splits[0], ([2022], 2023))
        self.assertEqual(splits[-1], ([2022, 2023, 2024], 2025))


if __name__ == "__main__":
    unittest.main()
