# Structured Data Extraction

Pull user-specified fields out of arbitrary PDFs and return them as structured JSON with provenance.

The point is to be **domain-agnostic**. Nothing domain-specific is checked in: the vocabulary, acronyms, and the shape of each field are all derived from the document itself at runtime. Hand it a PDF it has never seen, in a domain it was never built for, and it should still find your keys.

## How it works

Feed it a PDF and a list of keys. It converts the PDF to text, learns the document's own vocabulary, splits it into typed sections, embeds everything, finds the sections and rows closest to your keys, cleans up the messy tables it matched, then asks an LLM to read just those passages and return structured values.

```
PDF
 └─► convert  (docling)            → markdown + items.json
  └─► vocab  (YAKE)                → vocab.json
   └─► classify (RandomForest)     → sections.json
    └─► tables (complexity score)  → tables.json
     └─► embed (sentence-transformers) → *_embeddings.npy + row_map
      └─► search (cosine similarity)   → matches.json
       └─► refine matched tables (vision LLM, lazy) → cleaned HTML
        └─► extract (schema + LLM) → output/<stem>_extraction_<model>.json
```

Each step writes its output to disk and is skipped on a re-run if the file already exists, so iterating on a prompt does not re-run docling.

## Why this design

- **The document teaches its own vocabulary.** `vocab.py` runs unsupervised keyword extraction (YAKE) per PDF. A one-word key like `gain` is expanded with the document's related terms (`gain error`, `gain drift`) before search. This is the main reason a datasheet pipeline also works on an earnings report.
- **Search is semantic, not string matching.** "CMRR" and "common-mode rejection ratio" share no characters but are neighbours in embedding space. Cosine similarity over sentence-transformer embeddings is what lets a loose key hit the right passage.
- **Table rows are first-class targets.** Each row is embedded with its group header prepended, so a row reads like `Gain Error > G=1 | 0.01 | 0.04 %` instead of a context-free `0.01 | 0.04 %`.
- **The expensive model runs lazily.** Vision-LLM table cleanup only runs on tables that actually matched a key, so cost scales with what you asked for, not document size.
- **The LLM is boxed in.** `extract.py` discovers a per-key schema, builds a Pydantic model from it on the fly, and hands it to [Instructor](https://python.useinstructor.com/) so the model returns validated structured output, not prose.
- **Several models vote.** Extraction runs across a few LLMs in parallel; results are de-duplicated (preferring table-sourced records) and `rater.py` scores agreement and attaches a confidence to every record. One model also works: every fact just gets confidence 1.0.

## Pipeline at a glance

| Step | File | Job |
|------|------|-----|
| Convert | `convert.py` | Docling to markdown + `items.json` (label, page, bbox). OCR off, table model in FAST mode. |
| Vocab | `vocab.py` | Unsupervised domain vocabulary (YAKE, up to 4-grams). |
| Classify | `classify.py` | Pre-trained RandomForest tags each item as header/list/table/text/title. |
| Tables | `tables.py` | Score table complexity; rebuild symbol-only tables before embedding and matched tables after search (vision LLM, text-LLM fallback). |
| Embed | `embed.py` | SentenceTransformer over sections and table rows. nomic task prefixes (`search_document:` / `search_query:`). Sections stacked before rows. |
| Search | `search.py` | Cosine similarity vs the section+row matrix; hits above `SIMILARITY_THRESHOLD` (capped at `SEARCH_TOP_K`). |
| Extract | `extract.py` | Schema discovery, on-the-fly Pydantic model, Instructor extraction with retries, dedup, provenance. |
| Rate | `rater.py` | Cluster records into facts across models, score agreement (Cohen's kappa), write `rating_report.json`, `fused.json`, `user_report.json`. |
| Package | `fdo_records.py` | Wrap `user_report.json` into one FAIR-Digital-Object envelope per document with provenance. |

Supporting: `matching.py` (same-fact equivalence), `features.py` (classifier feature vector), `paths.py` (cache filename conventions), `config.py` (loads `.env`, derives paths, auto-selects CUDA/MPS/CPU).

## Installation

**Requirements:** Python 3.11+, plus an OpenAI-compatible LLM endpoint (the defaults assume a local [Ollama](https://ollama.com)). The first run also downloads the embedding model from Hugging Face.

```bash
# 1. clone and enter the repo
git clone <repo-url> && cd structured-data-extraction

# 2. create and activate a virtualenv
python3 -m venv venv
source venv/bin/activate

# 3. install the package 
pip install -e .
```

**4. Create a `.env` at the repo root.** It is required: the package reads these values on import and will not start without them. Copy this block and edit the values:

```bash
# Provider: any OpenAI-compatible endpoint (Ollama, OpenAI, ...)
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Models
SCHEMA_MODEL=glm-5.1:cloud
EXTRACTION_MODELS=gemma4:31b-cloud      # comma-separated for parallel voting
VISION_MODEL=kimi-k2.7-code:cloud       # must have vision abilities
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Tuning
EMBED_BATCH_SIZE=4        # small so long tables don't OOM a local GPU; raise on bigger hardware
MAX_LLM_RETRIES=2         # per-model retry budget on extraction failures
EXTRACTION_WORKERS=4      # parallel workers when running multiple extraction models

# Pipeline tuning — OPTIONAL: omit this whole block and the defaults below apply.
VOCAB_TOP_N=50                    # YAKE vocab terms kept per document
VOCAB_MAX_NGRAM=4                 # longest phrase YAKE treats as one term
SIMILARITY_THRESHOLD=0.55         # cosine floor for a section/row to count as a match
SEARCH_TOP_K=25                   # max matches kept per key (above the threshold)
VISION_COMPLEXITY_THRESHOLD=0.15  # min table complexity worth sending to the vision model
VISION_PARALLELISM=4              # parallel vision calls when refining matched tables
EXPAND_KEYS=true                  # expand each key with the document's own related vocab before search
```

Notes:
- `EXTRACTION_MODELS` is a comma-separated list. List several to cross-check agreement, or one to skip the agreement step. The first model is the "primary", used wherever a single model is needed (such as text-based table cleanup).
- For a non-Ollama provider, set `LLM_BASE_URL` / `LLM_API_KEY` accordingly (for OpenAI: `https://api.openai.com/v1` and `sk-...`).
- Shell environment variables override the file. Make sure every named model actually exists on your provider, or that step retries and then fails.
- The **pipeline tuning** block is optional: every value falls back to the default shown above if it's absent from `.env`. They are the knobs for recall vs. cost — e.g. lower `SIMILARITY_THRESHOLD` or raise `SEARCH_TOP_K` to retrieve more (and feed the LLM more), raise `VISION_COMPLEXITY_THRESHOLD` to send fewer tables to the (slow) vision model.

## Usage

```bash
# one PDF
structured-extract pdf/1167fc.pdf --keys gain CMRR "supply voltage" #when key is more than one word then it should be inside ""

# a whole directory
structured-extract pdf/ --keys revenue "net income"

# equivalent, without the console script
python -m structured_data_extraction.main pdf/1167fc.pdf --keys gain CMRR
```

Wrap multi-word keys in quotes. If you omit `--keys`, it prompts for a comma-separated list. A full run writes per-model extractions to `output/<stem>_extraction_<model>.json`, then fuses them into `user_report.json`, `fused.json`, `rating_report.json`, and the provenance-stamped `fdo_records.json`.

**Re-run a single stage.** Every step is runnable standalone against the caches in `intermediate/`, which is the fast loop when iterating on prompts:

```bash
python -m structured_data_extraction.extract --stem 1167fc --keys gain CMRR   # extraction only
python -m structured_data_extraction.rater                                    # re-score agreement
python -m structured_data_extraction.fdo_records                              # rebuild fdo_records.json
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
