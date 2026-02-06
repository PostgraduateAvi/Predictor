from pathlib import Path
from typing import Any, Dict


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    lines = []
    for key, value in data.items():
        if isinstance(value, (int, float)):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f"{key}: '{value}'")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def format_metrics_table(rows):
    header = "| Split | Recall@K | Precision@K | MAP@K | NDCG@K |"
    sep = "| --- | --- | --- | --- | --- |"
    body = [
        f"| {row['split']} | {row['recall_at_k']:.3f} | {row['precision_at_k']:.3f} | {row['map_at_k']:.3f} | {row['ndcg_at_k']:.3f} |"
        for row in rows
    ]
    return "\n".join([header, sep, *body])
