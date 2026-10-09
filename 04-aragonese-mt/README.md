# 04 · Aragonese MT

Benchmarking and LoRA fine-tuning small local LLMs for **Spanish → Aragonese**
(`spa_Latn → arg_Latn`) translation, entirely on Apple Silicon (MLX / mlx-lm,
Ollama for inference). Aragonese is a low-resource Romance language of northern
Aragon. Nothing runs in the cloud.

See [PLAN.md](PLAN.md) for the phases and their status.

## Quickstart

```bash
just install        # uv sync --dev
just download       # PILAR + FLORES+ → data/raw/ (checksummed, ~4 MB of zips)
just apertium-check # make sure the Apertium baseline works (see below)
just test
```

## Layout

```
04-aragonese-mt/
├── argmt/               # Python package
│   ├── paths.py         # canonical paths (relative to project root)
│   └── data/
│       └── download.py  # PILAR + FLORES+ downloader
├── configs/             # hyperparameter / model configs (phases 2, 4)
├── docker/              # Apertium spa↔arg image (macOS route)
├── results/             # benchmark tables and error analysis (committed)
├── tests/
└── data/                # git-ignored
    ├── raw/             # downloads, untouched: pilar/, flores/, MANIFEST.json
    └── processed/       # synthetic parallel corpus, splits (phase 3)
```

## Data

| Resource | What | Size | License |
|---|---|---|---|
| [PILAR](https://github.com/transducens/PILAR) `aragonese/crawled.zip` | Aragonese web crawl, language-ID filtered (blogs, consello.org, europapress, academiadelaragones.org) | 60,028 lines | Packaging CC0; underlying texts belong to their authors (notice-and-takedown policy) |
| PILAR `aragonese/literary.zip` | Literary translations into Aragonese (Instituto de l'Aragonés) | 24,675 lines | same as above |
| PILAR `FLORES+.zip` → `dev`, `devtest` | FLORES+ Spanish + Aragonese, reviewed by the Academia Aragonesa de la Lengua | 997 / 1,012 sentences | CC-BY-SA 4.0 |
| [apertium-spa-arg](https://github.com/apertium/apertium-spa-arg) 0.6.0 | Rule-based MT, both directions (`spa-arg`, `arg-spa`) | — | GPL-3.0 |

The downloader pins PILAR to commit `1ee5bc1` and verifies SHA-256 checksums.
Both PILAR Aragonese corpora are **monolingual**; parallel data is created in
phase 3 by back-translating them with Apertium `arg-spa`.

### Things to keep in mind

- **FLORES+ is evaluation-only.** PILAR ships it AES-encrypted so it doesn't get
  crawled into training sets. `data/` is git-ignored; don't commit decrypted
  references. Phase 3 also filters any FLORES+ overlap out of the training data
  (the crawl includes news sites, so this is a real risk).
- **Apertium has a home advantage on FLORES+.** The Aragonese FLORES+ references
  were produced by machine-translating the Spanish side *with Apertium* and then
  post-editing. Apertium's scores are therefore optimistic; treat it as a strong
  but biased baseline.
- **Synthetic data inherits Apertium's style.** Training on Apertium back-translations
  (Aragonese target side is human-written, Spanish source is synthetic) is the
  standard back-translation setup: the model learns to produce *real* Aragonese.
- **Orthography.** FLORES+ follows the current academy standard; the crawled text
  mixes older conventions (e.g. `ny`/`ñ`, `x`/`ix`). Expect some chrF noise from that.

## Apertium setup

Native Apertium works on Linux (`apt install apertium-spa-arg`). On macOS the
language pair isn't in Homebrew, so the easiest reliable route is Docker
(OrbStack or Docker Desktop):

```bash
just apertium-build   # builds argmt-apertium from Debian's apertium-spa-arg
just apertium-check   # uses native apertium if found, else the image
```

Alternatively, the Apertium project publishes macOS nightly packages
(see the [Apertium install wiki](https://wiki.apertium.org/wiki/Installation));
`just apertium-check` picks it up automatically if `apertium -l` lists `spa-arg`.

## Citation

```bibtex
@misc{PILAR,
  editor = {Galiano-Jiménez, Aarón and Sánchez-Martínez, Felipe and Pérez-Ortiz, Juan Antonio},
  title  = {PILAR},
  url    = {https://github.com/transducens/PILAR},
  year   = {2024}
}
```
