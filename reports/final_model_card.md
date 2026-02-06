# Final Model Card

## Purpose
Topic-level predictor for Yenepoya MD Medicine exam papers using topic frequency and ontology boosts.

## Metrics (Rolling 2022–2025)
| Split | Recall@K | Precision@K | MAP@K | NDCG@K |
| --- | --- | --- | --- | --- |
| macro | 0.024 | 0.002 | 0.002 | 0.003 |

## Limitations
- Relies on auto-tagged Harrison topics if manual gold labels are missing.
- Topic canonicalization uses exact term matching and may miss paraphrases.