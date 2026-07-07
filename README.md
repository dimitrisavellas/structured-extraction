# Structured Data Extraction

Pull user-specified fields out of arbitrary PDFs and return them as structured JSON with provenance.

It's **domain-agnostic**: nothing domain-specific is checked in. The vocabulary, acronyms, and field shapes are all derived from each document at runtime, so the same pipeline handles a chip datasheet or an earnings report without changes.

## How it works

Give it a PDF and a list of keys. It converts the PDF to text, learns the document's vocabulary, splits it into typed sections, and embeds them. Then it finds the sections and rows closest to your keys, cleans up any messy tables among them, and asks an LLM to read just those passages and return structured values.

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

- Vocabulary is learned per document. `vocab.py` runs YAKE keyword extraction on each PDF instead of using a fixed word list, and a one-word key like `gain` is expanded with the document's related terms (`gain error`, `gain drift`) before search.
- Search is semantic. "CMRR" and "common-mode rejection ratio" share no letters but sit close in embedding space, so cosine similarity over sentence-transformer embeddings finds the passage where string matching would miss it.
- Table rows are embedded one at a time, each with its group header prepended, so a row reads like `Gain Error > G=1 | 0.01 | 0.04 %` instead of a bare `0.01 | 0.04 %`.
- Vision runs lazily. The vision LLM only cleans up a table if that table matched a key, so cost tracks what you asked for rather than document size.
- The extraction LLM is constrained. `extract.py` discovers a per-key schema, builds a Pydantic model from it, and hands it to [Instructor](https://python.useinstructor.com/), which validates the output instead of accepting free text.
- Running several models is optional. Extraction fans out across the configured LLMs in parallel; records are de-duplicated (table-sourced ones preferred) and `rater.py` scores their agreement and attaches a confidence to each record. With one model, every record is confidence 1.0.

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
| Rate | `rater.py` | Cluster records into facts across models, attach a consensus confidence, score agreement (Cohen's kappa); write `fused.json` (every fact, confidence-sorted per key) and `rating_report.json`. |
| Package | `records.py` | Wrap `fused.json` into one provenance-stamped record per document (a metadata envelope plus the extracted fields). |

Supporting: `matching.py` (same-fact equivalence), `features.py` (classifier feature vector), `paths.py` (cache filename conventions), `config.py` (loads `.env`, derives paths, auto-selects CUDA/MPS/CPU).

## Design notes

The rationale the one-line table compresses out — the non-obvious choices, keyed to the file that makes them:

- **Convert (`convert.py`).** OCR is off and Docling's table model runs in `FAST` mode, roughly 2× faster. Turn OCR on for scanned documents(never really tested). Bounding boxes are translated from Docling's bottom-left origin to the top-left origin PyMuPDF uses later, so the two agree on where a table sits when it gets re-rendered.
- **Classify (`classify.py`).** A small supervised RandomForest re-labels items rather than trusting Docling's own labels, which mislabel multi-sentence captions and sub-headers. Its output, `sections.json`, is the canonical content stream every later step reads.
- **Tables (`tables.py`), symbol-only recovery.** Complex tables that Docling stripped down to bare symbols are rebuilt *before* embedding, not after. Those rows carry no words, so search could never retrieve them — meaning they'd never reach the lazy-vision refinement step either. De-garbling them up front gets their recovered text into the embeddings so they can match at all.
- **Embed (`embed.py`), task prefixes.** nomic needs `search_document:` / `search_query:` prefixes; without them the similarities collapse toward each other and `SIMILARITY_THRESHOLD` stops discriminating. Sections are stacked before rows in the matrix and `search.py` relies on that ordering.
- **Matching (`matching.py`).** Two records describe the same fact when their number and unit match exactly (after Unicode/dash normalisation) and their descriptor text clears a rapidfuzz similarity of 85 — fuzzy on the wording because models phrase conditions differently (`TA = −40°C` vs `T_A = -40°C`), exact on the number. Empty fields act as wildcards, so a sparse record still matches a fuller one.
- **Prompts.** The two LLM prompts are Jinja templates that ship inside the package — `schema_discovery.j2` (asks for each key's field shape and the document type) and `extraction.j2` (the actual value extraction). To tune them, run `structured-extract --dump-templates ./my-templates` for an editable copy, edit it, and set `TEMPLATES_DIR=./my-templates`. Overrides are per-file: change just `extraction.j2` and the other still loads its packaged default. No code change needed.

## Installation

**Requirements:** Python 3.11+, plus an OpenAI-compatible LLM endpoint (the defaults assume a local [Ollama](https://ollama.com)). The first run also downloads the embedding model from Hugging Face.

```bash
# 1. create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. install from PyPI
pip install structured-data-extraction
```

To modify the pipeline itself, install an editable checkout instead:

```bash
git clone <repo-url> && cd structured-data-extraction
pip install -e .
```

**Create a `.env` in the directory you run the tool from.** It is required: the package reads these values on startup and will not run without them. It searches the working directory and its parents, and real shell environment variables override the file. Copy this block and edit the values:

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

# Pipeline tuning = OPTIONAL: omit this whole block and the defaults below apply.
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

Wrap multi-word keys in quotes. If you omit `--keys`, it prompts for a comma-separated list. A full run writes per-model extractions to `output/<stem>_extraction_<model>.json`, then fuses them into `fused.json` (every fact, sorted by confidence) plus `rating_report.json`, and wraps `fused.json` into the provenance-stamped `records.json`.

A multi-model run then ends by printing the confidence-tier histogram and offering to keep a subset of tiers — pass `--levels` (e.g. `--levels 1.0 0.75`, or `all`) to choose up front, or omit it for an interactive prompt. Single-model runs skip this step, since every record is confidence 1.0.

**Re-run a single stage.** Every step is runnable standalone against the caches in `intermediate/`, which is the fast loop when iterating on prompts:

```bash
python -m structured_data_extraction.extract --stem 1167fc --keys gain CMRR   # extraction only
python -m structured_data_extraction.rater                                    # re-score agreement
python -m structured_data_extraction.records                                  # rebuild records.json
python -m structured_data_extraction.filter_records --levels 1.0 0.75         # query records by confidence
```

Confidence is `models-agreeing / total-models`. `filter_records` reads `records.json` and keeps any subset of the confidence tiers present in the data — pass `--levels` (e.g. `1.0 0.75`, or `all`), or omit it for an interactive menu. Needs more than one model to be meaningful.

## Outputs

Every stage caches its result under `intermediate/` and is skipped on a re-run if the file already exists; the finished deliverables land in `output/`.

```
intermediate/<stem>_items.json          docling items (label, page, bbox)
              <stem>_vocab.json          YAKE vocabulary
              <stem>_sections.json       classified content stream
              <stem>_tables.json         tables + complexity scores
              <stem>_*_embeddings.npy    section / key vectors
              <stem>_matches.json        search hits per key
              <stem>_run.json            run manifest (models, timings)

output/<stem>_extraction_<model>.json    one file per extraction model
       fused.json                        every fact, confidence-sorted per key  ← the deliverable
       records.json                      fused.json wrapped in a provenance envelope, one per document
       rating_report.json                agreement scoring + pairwise Cohen's kappa
```

A `records.json` entry pairs a provenance header with the extracted fields. Each field is a list of records sorted by confidence (`models-agreeing / total-models`), so the top record per key is the one the most models agreed on:

```jsonc
{
  "0150200051": {
    "type": "ExtractionResult",
    "provenance": {
      "source_file": "0150200051.pdf",
      "document_type": "cable connector product datasheet",
      "models": { "schema_discovery": "glm-5.1:cloud", "vision": "kimi-k2.7-code:cloud",
                  "embedding": "nomic-ai/nomic-embed-text-v1.5" },
      "user_keys": ["product family", "cable length", "Voltage", "net weight"]
    },
    "product_family": [
      { "confidence": 1.0,  "models": ["deepseek-v4-flash:cloud", "gemma4:31b-cloud",
                                       "gpt-oss:120b-cloud", "minimax-m2.5:cloud"],
        "value": "Cable",        "metadata": { "section_type": "text", "specified": true } },
      { "confidence": 0.25, "models": ["deepseek-v4-flash:cloud"],
        "value": "Cable Series", "metadata": { "section_type": "text", "specified": true } }
    ],
    "voltage": [
      { "confidence": 1.0, "models": ["deepseek-v4-flash:cloud", "gemma4:31b-cloud",
                                      "gpt-oss:120b-cloud", "minimax-m2.5:cloud"],
        "qualifier": "Maximum", "value": 60.0, "value_unit": "V AC",
        "metadata": { "section_type": "text", "specified": true } }
    ]
  }
}
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
