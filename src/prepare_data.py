import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


def _read_csv_from_zip(zip_path: Path, inner_path: str):
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_path) as handle:
            decoded = (line.decode("utf-8") for line in handle)
            reader = csv.DictReader(decoded)
            return list(reader)


def build_papers_json(questions, output_dir: Path, years=(2022, 2023, 2024, 2025)):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in questions:
        year = int(row["year"])
        if year not in years:
            continue
        paper_raw = row["paper_id"].strip()
        roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4}
        paper_id = roman_map.get(paper_raw, None)
        if paper_id is None:
            paper_id = int(paper_raw)
        q_id = row["question_id"]
        tags = [tag.strip() for tag in row.get("tags_raw", "").split(";") if tag.strip()]
        marks = None
        raw_marks = row.get("section", "")
        if "15" in raw_marks:
            marks = 15
        elif "10" in raw_marks:
            marks = 10
        elif "5" in raw_marks:
            marks = 5
        grouped[year][paper_id].append(
            {
                "q_id": q_id,
                "raw_text": row["question_text"],
                "year": year,
                "paper_id": paper_id,
                "marks": marks,
                "tags": tags,
                "harrison_tag_ids": row.get("harrison_tag_ids", ""),
                "harrison_tag_terms": row.get("harrison_tag_terms", ""),
                "templates_str": row.get("templates_str", ""),
            }
        )

    for year, papers in grouped.items():
        for paper_id, questions in papers.items():
            payload = {"year": year, "paper_id": paper_id, "questions": questions}
            out_path = output_dir / f"papers_{year}_paper{paper_id}.json"
            out_path.write_text(json.dumps(payload, indent=2))



def build_year_bundle(output_dir: Path, years=(2022, 2023, 2024, 2025)):
    for year in years:
        questions = []
        for paper_id in range(1, 5):
            path = output_dir / f"papers_{year}_paper{paper_id}.json"
            if path.exists():
                payload = json.loads(path.read_text())
                questions.extend(payload["questions"])
        bundle_path = output_dir / f"papers_{year}.json"
        bundle_path.write_text(
            json.dumps({"year": year, "paper_id": "all", "questions": questions}, indent=2)
        )


def build_topic_seed(topics, output_dir: Path):
    topic_entries = []
    for row in topics:
        topic_entries.append(
            {
                "topic_id": row["topic_id"],
                "name": row["canonical_topic_name"],
                "synonyms": [s.strip() for s in row.get("synonyms", "").split(";") if s.strip()],
                "parent_id": row.get("parent_id") or None,
                "parent_name": row.get("parent_name") or None,
                "domain": row.get("domain") or None,
                "term_norm": row.get("term_norm") or None,
            }
        )
    (output_dir / "topic_ontology_seed.json").write_text(json.dumps(topic_entries, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-zip", required=True)
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = _read_csv_from_zip(
        Path(args.artifacts_zip), "yenepoya_predictor_artifacts/yenepoya_master_questions_2006_2025.csv"
    )
    topics = _read_csv_from_zip(
        Path(args.artifacts_zip), "yenepoya_predictor_artifacts/topic_table_from_exam_matches.csv"
    )
    build_papers_json(questions, output_dir)
    build_year_bundle(output_dir)
    build_topic_seed(topics, output_dir)


if __name__ == "__main__":
    main()
