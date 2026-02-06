import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


WORD_RE = re.compile(r"\b[\w-]+\b")


def normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


@dataclass
class Topic:
    topic_id: str
    name: str
    synonyms: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    parent_name: Optional[str] = None
    domain: Optional[str] = None
    term_norm: Optional[str] = None


class TopicOntology:
    def __init__(self, topics: Iterable[Topic]):
        self.topics: Dict[str, Topic] = {topic.topic_id: topic for topic in topics}
        self._patterns: Dict[str, List[Tuple[str, re.Pattern]]] = {}
        self._build_patterns()

    @classmethod
    def from_seed(cls, path: Path) -> "TopicOntology":
        if not path.exists():
            return cls([])
        entries = json.loads(path.read_text())
        topics = []
        for entry in entries:
            synonyms = entry.get("synonyms", [])
            if entry.get("term_norm"):
                synonyms = list({*synonyms, entry["term_norm"]})
            topics.append(
                Topic(
                    topic_id=entry["topic_id"],
                    name=entry["name"],
                    synonyms=synonyms,
                    parent_id=entry.get("parent_id"),
                    parent_name=entry.get("parent_name"),
                    domain=entry.get("domain"),
                    term_norm=entry.get("term_norm"),
                )
            )
        return cls(topics)

    def _build_patterns(self) -> None:
        for topic in self.topics.values():
            patterns = []
            candidates = [topic.name, *topic.synonyms]
            for term in {normalize_text(c) for c in candidates if c}:
                if not term:
                    continue
                pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
                patterns.append((term, pattern))
            self._patterns[topic.topic_id] = patterns

    def match_topics(self, text: str, max_topics: int = 3) -> List[Tuple[str, float]]:
        normalized = normalize_text(text)
        scores: List[Tuple[str, float]] = []
        for topic_id, patterns in self._patterns.items():
            score = 0.0
            for term, pattern in patterns:
                if pattern.search(normalized):
                    score = max(score, min(1.0, 0.2 + len(term.split()) * 0.2))
            if score > 0:
                scores.append((topic_id, score))
        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:max_topics]

    def related_topics(self, topic_id: str) -> List[str]:
        topic = self.topics.get(topic_id)
        if topic and topic.parent_id:
            return [topic.parent_id]
        return []
