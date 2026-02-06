from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from src.ontology import TopicOntology


@dataclass
class PredictionResult:
    topic_id: str
    score: float


class BasePredictor:
    def predict(self, questions: Iterable[dict], ontology: TopicOntology, config: dict) -> List[PredictionResult]:
        raise NotImplementedError


class SimpleFrequencyPredictor(BasePredictor):
    def predict(self, questions: Iterable[dict], ontology: TopicOntology, config: dict) -> List[PredictionResult]:
        weights = {
            "recency": config.get("recency_weight", 1.0),
            "marks": config.get("marks_weight", 1.0),
            "frequency": config.get("frequency_weight", 1.0),
            "graph": config.get("graph_weight", 0.1),
        }
        scores: Dict[str, float] = {}
        for question in questions:
            topic_candidates = []
            if question.get("harrison_tag_ids"):
                ids = question["harrison_tag_ids"].split(";")
                topic_candidates = [i for i in ids if i]
            if not topic_candidates:
                topic_candidates = [topic_id for topic_id, _ in ontology.match_topics(question["raw_text"], 3)]
            if not topic_candidates:
                continue
            mark_weight = 1.0
            if question.get("marks"):
                mark_weight += (question["marks"] / 15) * weights["marks"]
            for topic_id in topic_candidates:
                scores[topic_id] = scores.get(topic_id, 0.0) + weights["frequency"] * mark_weight
                for related in ontology.related_topics(topic_id):
                    scores[related] = scores.get(related, 0.0) + weights["graph"] * mark_weight
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [PredictionResult(topic_id=topic_id, score=score) for topic_id, score in ranked]
