# Plan

Target machine: MacBook Pro M4 Pro, 48 GB unified memory. Local only.

| # | Phase | Deliverable | Status |
|---|---|---|---|
| 1 | Setup | uv project, layout, README, data downloader | ✅ done |
| 2 | Benchmark | Translate FLORES+ devtest spa→arg with 3–4 local LLMs + Apertium; chrF / BLEU (sacrebleu); Spanish/Catalan drift rate; `results/benchmark.md` | ⏳ |
| 3 | Data prep | Back-translate PILAR Aragonese with Apertium `arg-spa`; clean, dedupe, filter FLORES+ overlap; train/valid splits | ⏳ |
| 4 | Fine-tune | LoRA on the best small model with mlx-lm; `configs/lora.yaml` | ⏳ |
| 5 | Evaluate | Re-run benchmark with the adapter; error analysis with examples | ⏳ |
| 6 | Publish | Hugging Face model card (+ dataset card) | ⏳ |

## Data split policy

- **FLORES+ devtest** — final test set, used only for reported numbers.
- **FLORES+ dev** — prompt / decoding choices in phase 2, early-stopping sanity
  checks in phase 4. Never trained on.
- **Training** — synthetic pairs from PILAR only, with any sentence that
  overlaps FLORES+ (exact or near-duplicate) removed.
