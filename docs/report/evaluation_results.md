# Evaluation results (2026-09-30)

Raw data for the project report. The interpretation and conclusions are left to the student.

## Setup

| Item | Value |
|---|---|
| Corpus | `magnetic circuit.pdf` (7 p), `dc and ac machine.pdf` (22 p), `Unit 2 communication skills for Career building.pdf` (24 p) |
| Questions | 30: 26 answerable, 4 should-refuse (2 general, 2 near-domain traps: transformer EMF equation, stepper vs servo motor) |
| Question provenance | Drafted by an AI assistant from the note text at the student's request; AI-reviewed (expected pages cross-checked against page text, 26 questions reworded in exam style, 2 reference answers corrected). Student verification: pending (`"verified"` field in `evaluation/qa_dataset.json`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, CPU) |
| LLM and judge | `llama3.2:3b` via Ollama on an RTX 4050 laptop GPU (the same model generates and judges) |
| Retrieval | top_k = 5, fetch_k = 20, MMR λ = 0.6, hybrid weights 0.4 BM25 / 0.6 vector, score threshold 0.25 |
| Command | `python scripts/run_eval.py --llm` |

## Retrieval: all modes and chunk sizes

Numbers are after the text-cleaning improvement; the before run had identical values except where noted.

| Chunk size | Mode | Hit@5 | MRR | Refusal (threshold only) |
|---|---|---|---|---|
| 500 | similarity | 1.00 | 0.92 | 0.25 |
| 500 | mmr | 1.00 | 0.92 | 0.25 |
| 500 | hybrid | 1.00 | 0.92 | 0.25 |
| 1000 | similarity | 1.00 | 0.96 | 0.25 |
| 1000 | mmr | 1.00 | 0.95 | 0.25 |
| 1000 | hybrid | 1.00 | 0.96 (before: 0.94) | 0.25 |

Only "Who won the IPL in 2020?" fell below the score threshold. The best-chunk scores of the other three out-of-scope questions were:

| Question | Chunk size 500 | Chunk size 1000 |
|---|---|---|
| TCP three-way handshake | 0.31 | 0.26 |
| Transformer EMF equation | 0.61 | 0.56 |
| Stepper vs servo motor | 0.46 | 0.43 |

## Answer quality: MMR, chunk size 1000

| Run | Faithfulness (1–5) | Relevance (1–5) | Refusal (final answer) | Latency |
|---|---|---|---|---|
| Original question wording, digit-only judge prompt | 1.31 | 4.23 | 1.00 | 4.98 s |
| Original question wording, claim-by-claim judge prompt | 3.69 | 4.08 | 1.00 | 5.23 s |
| Reworded questions, before text-cleaning change | 3.62 | 4.19 | 1.00 | 5.57 s |
| Reworded questions, after text-cleaning change | 3.46 | 4.23 | 1.00 | 5.41 s |

Blueprint targets: Hit@5 ≥ 0.80, MRR ≥ 0.60, faithfulness ≥ 4.0, relevance ≥ 4.0, refusal ≥ 90%, latency < 20 s (local 3B model).

## Observations recorded during evaluation

- **Judge prompt format.** With a "reply with a single digit" prompt, llama3.2:3b returned "1" for every answer, including answers copied almost verbatim from the notes. With a claim-by-claim prompt it scored two real answers 3 and 4 and a deliberately fabricated answer 1.
- **Judge noise.** Between the before and after runs, faithfulness changed on 12 of 26 questions:
  - up on 4 and down on 8;
  - several swings of 3 points (e.g. cs-05 went 4 → 1, cs-06 went 1 → 4);
  - some changes were on pages the cleaning change did not affect.
- **Two-stage refusal.** The score threshold refused 1 of 4 out-of-scope questions. The LLM refused the other 3, giving 4 of 4 overall.
- **Rewording.** Rewording the questions in exam style (for example, describing "reluctance" without naming it) left Hit@5 at 1.00.
- **Text-cleaning change.** It re-joined one-word-per-line fragments and converted bullet glyphs (U+FFFD, private-use Wingdings characters, ▪, lone "o") to "- " items. After re-indexing, the live index contains no U+FFFD characters and 102 converted bullets. Chunk counts were unchanged: 16, 81 and 72.
- **Tables.** Comparison tables (magnetic vs electric circuit; slip-ring vs squirrel-cage motor; conference vs seminar) are extracted as run-on text. PyMuPDF's `find_tables()` detected many ordinary paragraphs as tables and split multi-line cells across rows, so it was not used.

## Files

- `evaluation/qa_dataset.json`: the question set.
- `evaluation/results/` (not in git): full JSON results and `answers-*.md` with every generated answer, its citations and its judge scores.
