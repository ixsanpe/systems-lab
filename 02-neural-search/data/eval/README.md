# Eval query set (queries.json + qrels.json)

50 English queries with graded relevance judgments (qrels) against the Zurich cantonal
courts corpus (`data/raw/<court>/<file>.json`), for the recall@k / nDCG@10 / MRR baseline.

**Written by Claude, not by the repo owner**

## How these were built

- Sampled the `Abstract` field across ~111k documents that have one (of ~222k total);
  the text before the first `|` is a recurring legal-topic tag (e.g. `Baubewilligung`,
  `Sozialhilfe`), close to a controlled vocabulary.
- For `ZH_Verwaltungsgericht` (21 queries), `ZH_Steuerrekurs` (7) and
  `ZH_Sozialversicherungsgericht` (6): abstracts contain a real case summary in German,
  which was read directly (no translation tool) to write an English query grounded in
  a specific fact pattern, and to grade candidate documents 0-3 by how well their actual
  content matches that fact pattern — not just the topic tag.
- For `ZH_Obergericht` (16 queries): **its `Abstract` field is consistently empty
  (topic tag only, no summary text)** in this corpus — a real ingestion artifact, not a
  sampling gap. There is no content to read until `data/extracted/` is populated and
  full text is available. These 16 queries' judgments are topic-tag matches only
  (relevance 2 = same topic tag, 0 = different topic tag) and are explicitly weaker
  ground truth than the other 34. `judgment_basis` in `queries.json` marks this per
  query — filter on it if you want a cleaner subset.

## Format

- `queries.json`: `query_id, query_en, topic_de, court, judgment_basis`.
- `qrels.json`: `query_id, doc_id, court, relevance (0-3), note`. `doc_id` is the raw
  JSON filename under `data/raw/<court>/` (verified to resolve for every row).

## Known limitations (read before trusting the baseline number)

- **Not pooled from real retrieval.** This is a seed judgment set, not TREC-style
  pooling — each query has a small hand-picked pool (4-8 docs: a couple of graded
  positives, sometimes a topic-adjacent doc at grade 1, one hard negative from a
  different topic). Anything a real system retrieves outside this pool is *unjudged*,
  not confirmed non-relevant — recall@k against this qrels file will be a biased/
  optimistic estimate until the pool is extended with docs the systems under test
  actually surface (standard pooling residual judging).
- **Obergericht judgments are coarse** (see above) — 16/50 queries, no fact-pattern
  grading possible until full text is extracted.
- **Court distribution isn't uniform**: 21 Verwaltungsgericht / 7 Steuerrekurs /
  6 Sozialversicherungsgericht / 16 Obergericht, roughly tracking each court's share of
  abstract richness rather than its share of the corpus (Obergericht is ~46% of raw
  docs by volume but has the weakest judgments here).
- Relevance grades reflect one person's (mine) reading of short abstracts, not
  independent double-judging — treat as a first-pass baseline, not adjudicated ground
  truth.
