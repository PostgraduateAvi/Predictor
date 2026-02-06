import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.autotune import AutoTuner
from src.evaluate import compute_metrics, load_year_data, macro_average, rolling_splits, temporal_split
from src.labeling import auto_label_questions, export_suggestions, load_gold_labels
from src.ontology import TopicOntology
from src.predictor_interface import SimpleFrequencyPredictor
from src.utils import format_metrics_table


BACKUP_TEMPLATES = [
    "Define and classify {topic}.",
    "Discuss the clinical features and management of {topic}.",
    "Approach to diagnosis of {topic}.",
    "Complications and monitoring in {topic}.",
    "Outline the pathophysiology and treatment of {topic}.",
]


def load_config(path: Path) -> dict:
    if not path.exists():
        return {
            "recency_weight": 1.0,
            "marks_weight": 1.0,
            "frequency_weight": 1.0,
            "graph_weight": 0.1,
        }
    config = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            config[key.strip()] = float(value.strip().strip("'"))
    return config


def build_reports(
    reports_dir: Path,
    baseline_metrics: dict,
    baseline_paper_metrics: dict,
    tuning_history: dict,
    best_metrics,
    best_config: dict,
):
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows = [
        {
            "split": split,
            "recall_at_k": metrics.recall_at_k,
            "precision_at_k": metrics.precision_at_k,
            "map_at_k": metrics.map_at_k,
            "ndcg_at_k": metrics.ndcg_at_k,
        }
        for split, metrics in baseline_metrics.items()
    ]
    baseline_report = "# Baseline Report\n\n" + format_metrics_table(baseline_rows) + "\n"
    baseline_report += "\n## Per-Paper Metrics\n\n"
    for split, paper_metrics in baseline_paper_metrics.items():
        baseline_report += f"### {split}\n\n"
        paper_rows = [
            {
                "split": f"Paper {paper_id}",
                "recall_at_k": metrics.recall_at_k,
                "precision_at_k": metrics.precision_at_k,
                "map_at_k": metrics.map_at_k,
                "ndcg_at_k": metrics.ndcg_at_k,
            }
            for paper_id, metrics in paper_metrics.items()
        ]
        baseline_report += format_metrics_table(paper_rows) + "\n\n"
    (reports_dir / "baseline_report.md").write_text(baseline_report)

    tuning_lines = ["# Tuning Report", "", "## All Runs"]
    tuning_rows = []
    for run_id, result in tuning_history.items():
        tuning_rows.append(
            {
                "split": run_id,
                "recall_at_k": result.metrics.recall_at_k,
                "precision_at_k": result.metrics.precision_at_k,
                "map_at_k": result.metrics.map_at_k,
                "ndcg_at_k": result.metrics.ndcg_at_k,
            }
        )
    tuning_lines.append(format_metrics_table(tuning_rows))
    tuning_lines.append("\n## Best Configuration")
    tuning_lines.append(json.dumps(best_config, indent=2))
    (reports_dir / "tuning_report.md").write_text("\n".join(tuning_lines))

    model_card = [
        "# Final Model Card",
        "",
        "## Purpose",
        "Topic-level predictor for Yenepoya MD Medicine exam papers using topic frequency and ontology boosts.",
        "",
        "## Metrics (Rolling 2022–2025)",
        format_metrics_table(
            [
                {
                    "split": "macro",
                    "recall_at_k": best_metrics.recall_at_k,
                    "precision_at_k": best_metrics.precision_at_k,
                    "map_at_k": best_metrics.map_at_k,
                    "ndcg_at_k": best_metrics.ndcg_at_k,
                }
            ]
        ),
        "",
        "## Limitations",
        "- Relies on auto-tagged Harrison topics if manual gold labels are missing.",
        "- Topic canonicalization uses exact term matching and may miss paraphrases.",
    ]
    (reports_dir / "final_model_card.md").write_text("\n".join(model_card))


def generate_predictions(
    predictions_dir: Path,
    predictor: SimpleFrequencyPredictor,
    ontology: TopicOntology,
    train_questions: list,
    config: dict,
):
    predictions_dir.mkdir(parents=True, exist_ok=True)
    questions_by_paper = defaultdict(list)
    for question in train_questions:
        questions_by_paper[question.get("paper_id")].append(question)

    for paper_id in range(1, 5):
        paper_questions = questions_by_paper.get(paper_id, [])
        ranked = predictor.predict(paper_questions, ontology, config)
        ranked_topics = [item.topic_id for item in ranked]
        high_conf = ranked_topics[:30]
        backup = [topic for topic in ranked_topics[30:60] if topic not in high_conf]
        output = {
            "paper_id": paper_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "high_confidence": [
                {
                    "topic_id": topic,
                    "confidence": round(ranked[idx].score, 3) if idx < len(ranked) else 0,
                    "backup_questions": [
                        template.format(topic=ontology.topics.get(topic, None).name if ontology.topics.get(topic) else topic)
                        for template in BACKUP_TEMPLATES
                    ],
                }
                for idx, topic in enumerate(high_conf)
            ],
            "coverage_backup": [
                {
                    "topic_id": topic,
                    "confidence": round(ranked[30 + idx].score, 3) if (30 + idx) < len(ranked) else 0,
                    "backup_questions": [
                        template.format(topic=ontology.topics.get(topic, None).name if ontology.topics.get(topic) else topic)
                        for template in BACKUP_TEMPLATES
                    ],
                }
                for idx, topic in enumerate(backup)
            ],
        }
        (predictions_dir / f"predicted_2026_paper{paper_id}.json").write_text(
            json.dumps(output, indent=2)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--predictions-dir", default="predictions")
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--epsilon", type=float, default=0.001)
    parser.add_argument("--max-rounds", type=int, default=12)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ontology = TopicOntology.from_seed(data_dir / "topic_ontology_seed.json")
    predictor = SimpleFrequencyPredictor()

    years = [2022, 2023, 2024, 2025]
    gold_labels = load_gold_labels(data_dir / "gold_labels.csv")

    baseline_metrics = {}
    baseline_paper_metrics = {}
    for train_years, test_year in rolling_splits(years):
        train_questions = []
        for year in train_years:
            train_questions.extend(load_year_data(data_dir, year))
        test_questions = load_year_data(data_dir, test_year)
        auto_labels = auto_label_questions(test_questions, ontology)
        gold_for_test = {
            q["q_id"]: gold_labels.get(q["q_id"], auto_labels.get(q["q_id"], []))
            for q in test_questions
        }
        predictions = predictor.predict(train_questions, ontology, {})
        predicted_topics = [pred.topic_id for pred in predictions]
        per_question_metrics = [
            compute_metrics(predicted_topics, gold, args.k) for gold in gold_for_test.values()
        ]
        baseline_metrics[f"train_{train_years[-1]}_test_{test_year}"] = macro_average(
            per_question_metrics
        )
        paper_groups = defaultdict(list)
        for question in test_questions:
            paper_groups[question.get("paper_id")].append(question)
        paper_metrics = {}
        for paper_id, questions in paper_groups.items():
            gold_for_paper = [
                gold_for_test.get(question["q_id"], []) for question in questions
            ]
            per_metrics = [compute_metrics(predicted_topics, gold, args.k) for gold in gold_for_paper]
            paper_metrics[paper_id] = macro_average(per_metrics)
        baseline_paper_metrics[f"train_{train_years[-1]}_test_{test_year}"] = paper_metrics

    tuner = AutoTuner(data_dir, ontology)
    tune_results = tuner.tune(years, args.k, args.epsilon, args.max_rounds)
    best = tune_results["best"]

    best_config = best.config if best else {}
    best_metrics = best.metrics if best else macro_average(list(baseline_metrics.values()))

    build_reports(
        Path(args.reports_dir),
        baseline_metrics,
        baseline_paper_metrics,
        tune_results["history"],
        best_metrics,
        best_config,
    )

    all_questions = []
    for year in years:
        all_questions.extend(load_year_data(data_dir, year))

    suggestions = auto_label_questions(all_questions, ontology)
    export_suggestions(data_dir / "auto_label_suggestions.csv", suggestions)

    generate_predictions(Path(args.predictions_dir), predictor, ontology, all_questions, best_config)


if __name__ == "__main__":
    main()
