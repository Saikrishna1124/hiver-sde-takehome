# Roadmap

Status legend: [x] done and run, [~] scaffolded, [ ] not started.

- [x] Phase 0 - repo structure, config, requirements, .env.example, .gitignore
- [x] Phase 1 - data loader + preprocessing + `src/inspect_data.py` (run on the real slice)
- [x] Phase 2 - brand selection table -> AmazonHelp (`src/brand_stats.py`)
- [x] Phase 3 - conversation reconstruction
- [x] Phase 4 - intent discovery + locked 10-intent taxonomy
- [x] Phase 5 - leakage-safe splits
- [x] Phase 6 - majority baseline
- [x] Phase 7 - TF-IDF + LogisticRegression baseline
- [ ] Phase 8 - embedding intent classifier
- [ ] Phase 9 - FAISS retrieval
- [ ] Phase 10 - grounded reply generator (needs GEMINI_API_KEY)
- [ ] Phase 11 - rule-based AUTO / HUMAN escalation
- [x] Phase 12 - golden set: 200/200 hand-labelled; validate_golden_set passes
- [ ] Phase 13 - classifier evaluation harness
- [ ] Phase 14-16 - reply rubric, LLM judge, human agreement
- [ ] Phase 17 - escalation evaluation
- [x] Phase 18-19 - failure analysis + misleading-headline section (outputs/report.md)
- [x] Phase 20-22 - decision_log.md (15 decisions), outputs/report.md, README + evaluation/README updated
- [x] Phase 23-24 - tests (67 passing), REPRODUCE.md (< 4 min end-to-end)
- [ ] Phase 25-27 - security review, citations, final check

## Blockers

1. **Dataset is a slice, not the full file.** `data/raw/twcs.csv` holds 59,565
   rows uploaded by the user, not the full ~2.8M-row Kaggle file. All counts and
   metrics are reported as "on this slice".
2. **GEMINI_API_KEY** needed only for live Gemini calls (Phases 10, 15). The
   Stage 2 dataset is already reproduced from cache; no API key is required to
   run the full pipeline via `REPRODUCE.md`.
