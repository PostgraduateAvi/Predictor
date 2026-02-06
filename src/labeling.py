import csv
from pathlib import Path
from typing import Dict, Iterable, List

from src.ontology import TopicOntology


def load_gold_labels(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    labels: Dict[str, List[str]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.setdefault(row["q_id"], []).append(row["topic_id"])
    return labels


def auto_label_questions(
    questions: Iterable[dict], ontology: TopicOntology, max_topics: int = 3
) -> Dict[str, List[str]]:
    suggestions = {}
    for question in questions:
        matches = ontology.match_topics(question["raw_text"], max_topics=max_topics)
        suggestions[question["q_id"]] = [topic_id for topic_id, _ in matches]
    return suggestions


def export_suggestions(path: Path, suggestions: Dict[str, List[str]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_id", "topic_ids"])
        for q_id, topic_ids in suggestions.items():
            writer.writerow([q_id, ";".join(topic_ids)])
