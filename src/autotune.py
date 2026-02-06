import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from src.evaluate import Metrics, compute_metrics, load_year_data, macro_average, rolling_splits
from src.labeling import auto_label_questions, load_gold_labels
from src.ontology import TopicOntology
from src.predictor_interface import SimpleFrequencyPredictor
from src.utils import dump_yaml


@dataclass
class TuneResult:
    config: dict
    metrics: Metrics


class AutoTuner:
    def __init__(self, data_dir: Path, ontology: TopicOntology, seed: int = 42):
        self.data_dir = data_dir
        self.ontology = ontology
        self.seed = seed
        self.predictor = SimpleFrequencyPredictor()

    def _candidate_configs(self) -> List[dict]:
        weights = [0.8, 1.0, 1.2]
        graph = [0.05, 0.1, 0.2]
        configs = []
        for recency, marks, frequency, graph_weight in itertools.product(weights, weights, weights, graph):
            configs.append(
                {
                    "recency_weight": recency,
                    "marks_weight": marks,
                    "frequency_weight": frequency,
                    "graph_weight": graph_weight,
                }
            )
        return configs

    def evaluate_config(self, config: dict, years: List[int], k: int) -> Metrics:
        metrics = []
        gold_labels = load_gold_labels(self.data_dir / "gold_labels.csv")
        for train_years, test_year in rolling_splits(years):
            train_questions = []
            for year in train_years:
                train_questions.extend(load_year_data(self.data_dir, year))
            test_questions = load_year_data(self.data_dir, test_year)
            auto_labels = auto_label_questions(test_questions, self.ontology)
            gold_for_test = {
                q["q_id"]: gold_labels.get(q["q_id"], auto_labels.get(q["q_id"], []))
                for q in test_questions
            }
            predictions = self.predictor.predict(train_questions, self.ontology, config)
            predicted_topics = [pred.topic_id for pred in predictions]
            per_question_metrics = [
                compute_metrics(predicted_topics, gold, k) for gold in gold_for_test.values()
            ]
            metrics.append(macro_average(per_question_metrics))
        return macro_average(metrics)

    def tune(self, years: List[int], k: int, epsilon: float, max_rounds: int) -> Dict[str, TuneResult]:
        best: TuneResult = None
        history: Dict[str, TuneResult] = {}
        configs = self._candidate_configs()
        rounds_without_improvement = 0
        for idx, config in enumerate(configs, start=1):
            result = self.evaluate_config(config, years, k)
            key = f"candidate_{idx}"
            history[key] = TuneResult(config=config, metrics=result)
            if best is None or result.recall_at_k > best.metrics.recall_at_k:
                best = TuneResult(config=config, metrics=result)
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
            if rounds_without_improvement >= max_rounds:
                break
            if best and abs(result.recall_at_k - best.metrics.recall_at_k) < epsilon:
                continue
        if best:
            dump_yaml(best.config, Path("configs/best_config.yaml"))
        return {
            "best": best,
            "history": history,
        }
