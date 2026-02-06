# Data Inputs

Place the following files in this `data/` folder to run the full pipeline:

- `papers_2022.json`
- `papers_2023.json`
- `papers_2024.json`
- `papers_2025.json`
- `topic_ontology_seed.json` (optional, used to seed the ontology)
- `gold_labels.csv` (optional, manual topic labels)

Each `papers_YYYY.json` file must follow this structure:

```json
{
  "year": 2022,
  "paper_id": 1,
  "questions": [
    {
      "q_id": "2022-1-001",
      "raw_text": "Question stem...",
      "marks": 10,
      "tags": ["optional", "labels"]
    }
  ]
}
```

If you already have the canonical dataset and ontology from the provided
`yenepoya_predictor_artifacts.zip`, you can generate the JSON files via:

```bash
python -m src.prepare_data --artifacts-zip yenepoya_predictor_artifacts.zip --output-dir data
```

This script writes the required `papers_2022.json`–`papers_2025.json` and a
`topic_ontology_seed.json` derived from the Harrison index topic table.
