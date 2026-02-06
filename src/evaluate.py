import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class Metrics:
    recall_at_k: float
    precision_at_k: float
    map_at_k: float
    ndcg_at_k: float


def recall_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 0.0
    pred = set(predicted[:k])
    return len(pred & set(gold)) / len(set(gold))


def precision_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    pred = predicted[:k]
    if not pred:
        return 0.0
    return len(set(pred) & set(gold)) / len(pred)


def average_precision_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    if not gold:
        return 0.0
    score = 0.0
    hits = 0
    for idx, topic in enumerate(predicted[:k], start=1):
        if topic in gold:
            hits += 1
            score += hits / idx
    return score / min(len(gold), k)


def ndcg_at_k(predicted: List[str], gold: List[str], k: int) -> float:
    def dcg(items: List[str]) -> float:
        score = 0.0
        for idx, topic in enumerate(items[:k], start=1):
            if topic in gold:
                score += 1.0 / (1 + (idx - 1))
        return score

    ideal = dcg(gold)
    if ideal == 0:
        return 0.0
    return dcg(predicted) / ideal


def compute_metrics(predicted: List[str], gold: List[str], k: int) -> Metrics:
    return Metrics(
        recall_at_k=recall_at_k(predicted, gold, k),
        precision_at_k=precision_at_k(predicted, gold, k),
        map_at_k=average_precision_at_k(predicted, gold, k),
        ndcg_at_k=ndcg_at_k(predicted, gold, k),
    )


def load_year_data(data_dir: Path, year: int) -> List[dict]:
    path = data_dir / f"papers_{year}.json"
    payload = json.loads(path.read_text())
    return payload["questions"]


def rolling_splits(years: Iterable[int]) -> List[Tuple[List[int], int]]:
    years = sorted(years)
    splits = []
    for idx, test_year in enumerate(years[1:], start=1):
        train_years = years[:idx]
        splits.append((train_years, test_year))
    return splits


def temporal_split(years: Iterable[int], test_year: int) -> Tuple[List[int], int]:
    train = [year for year in years if year < test_year]
    return train, test_year


def macro_average(metrics: List[Metrics]) -> Metrics:
    if not metrics:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    return Metrics(
        recall_at_k=sum(m.recall_at_k for m in metrics) / len(metrics),
        precision_at_k=sum(m.precision_at_k for m in metrics) / len(metrics),
        map_at_k=sum(m.map_at_k for m in metrics) / len(metrics),
        ndcg_at_k=sum(m.ndcg_at_k for m in metrics) / len(metrics),
    )
