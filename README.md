# Structured Data Extraction

[![PyPI](https://img.shields.io/pypi/v/structured-data-extraction)](https://pypi.org/project/structured-data-extraction/)

Extract user-specified fields from arbitrary PDFs as structured JSON with provenance.

The pipeline is **domain-agnostic**: vocabulary, acronyms and field shapes are derived from each document at runtime, so the same code handles a chip datasheet or an earnings report without changes. It runs against any OpenAI-compatible endpoint, from cloud models to small local models on a laptop.

## Contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Running on local models](#running-on-local-models)
- [Outputs](#outputs)
- [Development](#development)

## How it works

Given a PDF and a list of keys, the pipeline converts the PDF to text, learns the document's vocabulary, classifies and embeds its sections, retrieves the passages closest to each key, repairs messy tables among them, and asks an LLM to return validated, structured values.

| Step | Module | Job |
|------|--------|-----|
| Convert | `convert.py` | Docling to markdown + `items.json` (label, page, bbox). |
| Vocab | `vocab.py` | Per-document vocabulary (YAKE, up to 4-grams); one-word keys are expanded with related terms. |
| Classify | `classify.py` | Pre-trained RandomForest tags each item as header, list, table, text or title. |
| Tables | `tables.py` | Scores table complexity; matched complex tables are rebuilt by a vision LLM. |
| Embed | `embed.py` | Sentence-transformer embeddings of sections and of each table row with its group header. |
| Search | `search.py` | Cosine similarity, keeping hits above `SIMILARITY_THRESHOLD` (at most `SEARCH_TOP_K`). |
| Extract | `extract.py` | Schema discovery, a Pydantic model built on the fly, [Instructor](https://python.useinstructor.com/)-validated extraction, dedup. |
| Rate | `rater.py` | Clusters records across models, attaches a consensus confidence, scores agreement (Cohen's kappa). |
| Package | `records.py` | Wraps the fused result into one provenance-stamped record per document. |

Design choices worth knowing:

- **Semantic search.** "CMRR" and "common-mode rejection ratio" share no letters but sit close in embedding space.
- **Lazy vision.** Only tables that matched a key are sent to the vision model, so cost tracks the request, not the document size.
- **Constrained output.** The LLM fills a Pydantic model; free text is rejected and retried.
- **Optional voting.** List several extraction models to get per-record confidence (`models agreeing / total models`).
- **Caching.** Every step writes to disk and is skipped on re-run, so iterating on a prompt does not re-run docling.

Each module's docstring explains its non-obvious design decisions.

## Installation

Requires Python 3.11+ and an OpenAI-compatible LLM endpoint (the defaults assume [Ollama](https://ollama.com)). The embedding model is downloaded from Hugging Face on first run.

```bash
python3 -m venv venv && source venv/bin/activate
pip install structured-data-extraction
```

For development, install an editable checkout:

```bash
git clone https://github.com/dimitrisavellas/structured-extraction.git
cd structured-extraction
pip install -e ".[dev]"
```

Or with [uv](https://docs.astral.sh/uv/), which uses the committed `uv.lock`:

```bash
uv sync --extra dev --group lint
```

## Configuration

Create a `.env` in the directory you run the tool from (parent directories are searched too). Shell environment variables override it.

```bash
# Provider: any OpenAI-compatible endpoint
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Models
SCHEMA_MODEL=glm-5.1:cloud
EXTRACTION_MODELS=gemma4:31b-cloud              # comma-separated for multi-model voting
VISION_MODEL=kimi-k2.7-code:cloud               # must support images
EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5  # Hugging Face model ID

# Runtime
EMBED_BATCH_SIZE=4      # raise on bigger GPUs
MAX_LLM_RETRIES=2       # retries per extraction model
EXTRACTION_WORKERS=4    # parallel workers across extraction models
```

Optional settings (defaults shown):

| Variable | Default | Purpose |
|----------|---------|---------|
| `SIMILARITY_THRESHOLD` | `0.55` | Cosine floor for a section or row to count as a match. |
| `SEARCH_TOP_K` | `25` | Maximum matches kept per key. |
| `VISION_COMPLEXITY_THRESHOLD` | `0.15` | Minimum table complexity sent to the vision model. |
| `VISION_PARALLELISM` | `4` | Parallel vision calls. |
| `VOCAB_TOP_N` / `VOCAB_MAX_NGRAM` | `50` / `4` | Vocabulary size and longest phrase. |
| `EXPAND_KEYS` | `true` | Expand keys with related document vocabulary before search. |
| `REASONING_EFFORT` | unset | Sent as `reasoning_effort` on schema and extraction calls; `none` disables thinking. |
| `LLM_TIMEOUT` | `600` | Seconds per schema or extraction request. |
| `EXTRACT_PER_SECTION` | `false` | One extraction call per matched section instead of one for all. |

The first model in `EXTRACTION_MODELS` is the primary, used wherever a single model is needed. Every named model must exist on your provider.

## Usage

### Command line

```bash
structured-extract report.pdf --keys revenue "net income"   # one PDF
structured-extract pdfs/ --keys gain CMRR                     # a directory
```

Quote multi-word keys. Without `--keys` you are prompted for them.

| Flag | Effect |
|------|--------|
| `--schema FILE` | Use your own schema and skip schema discovery (see below). |
| `--local` | Settings for small local models (see [Running on local models](#running-on-local-models)). |
| `--levels 1.0 0.75` | Keep only these confidence tiers (`all` keeps everything); multi-model runs prompt otherwise. |
| `--force` | Recompute every step, ignoring cached files. |
| `--output DIR` | Output directory (default `output/`). |
| `--dump-templates DIR` | Write the prompt templates to `DIR` for editing; point `TEMPLATES_DIR` at it. |

### Python

```python
from structured_data_extraction.main import process_single_pdf, process_batch, finalize_outputs

process_single_pdf("report.pdf", ["revenue", "net income"])  # or process_batch("pdfs/", keys)
finalize_outputs(levels=["all"])
```

### Custom schema

`--schema` takes a JSON file in the same format as `intermediate/*_schema.json` and skips schema discovery; keys are derived from it. Setting `"fixed"` on a sub-field turns it into a `Literal`, so the model can only emit that value. This is useful for slicing a table down to one variant or column:

```json
{
  "document_type": "operational amplifier datasheet",
  "schema": {
    "quiescent_current_s_typ": {
      "type": "list",
      "description": "Quiescent current typical value for the AD840S variant",
      "fields": {
        "differentiator": { "type": "string", "description": "Product variant", "fixed": "AD840S" },
        "qualifier": { "type": "string", "description": "Column label: min, typ, or max", "fixed": "Typ" },
        "condition": { "type": "string", "description": "Operating condition" },
        "value": { "type": "number", "description": "Quiescent current value", "unit": "mA" },
        "metadata": { "type": "object", "description": "Source and provenance" }
      }
    }
  }
}
```

### Single stages

Each step is also a console script, run from the same working directory:

```bash
structured-extract-convert  --input pdf/example.pdf
structured-extract-vocab    --input markdown/example.md
structured-extract-classify --stem example
structured-extract-embed    --stem example --keys gain CMRR
structured-extract-search   --stem example --original-keys gain CMRR
structured-extract-extract  --stem example --keys gain CMRR
structured-extract-rate                          # fuse models, score agreement
structured-extract-records                       # wrap fused.json into records.json
structured-extract-filter   --levels 1.0 0.75    # keep selected confidence tiers
```

## Running on local models

The pipeline runs fully offline with small models (tested with 9B models on a 16 GB MacBook Air via Ollama). Pass `--local`:

```bash
structured-extract report.pdf --keys "net income" --local
```

`--local` applies three settings that small models need:

- **One extraction call per section** (`EXTRACT_PER_SECTION`). When several tables repeat the same rows in one prompt, 9B models at temperature 0 tend to loop on them indefinitely.
- **Thinking off** (`REASONING_EFFORT=none`), unless you set `REASONING_EFFORT` yourself. Thinking at temperature 0 can run away.
- **30-minute request timeout** (`LLM_TIMEOUT`), unless you set a longer one.

Ollama loads models with a small context window by default (4096 tokens on machines with less than about 24 GB of GPU memory) and **silently truncates** longer prompts. The schema-discovery prompt can reach about 17k tokens, so start Ollama with a larger window:

```bash
OLLAMA_CONTEXT_LENGTH=40960 ollama serve
```

Passing `--schema` skips discovery. With `--local`, each extraction prompt holds one section (under 7k tokens in testing), so a 16k window is usually enough.

Expect lower precision than with large models: 9B models occasionally invent records. Listing two or more extraction models and filtering with `--levels` mitigates this.

## Outputs

Steps cache their results in `intermediate/`; deliverables go to `output/`.

```
output/<stem>_extraction_<model>.json   raw records, one file per extraction model
       fused.json                       every fact, confidence-sorted per key
       records.json                     fused.json in a provenance envelope, one entry per document
       rating_report.json               agreement statistics and pairwise Cohen's kappa
```

A `records.json` entry pairs provenance with the extracted fields, each a list of records sorted by confidence:

```jsonc
{
  "0150200051": {
    "type": "ExtractionResult",
    "provenance": {
      "source_file": "0150200051.pdf",
      "document_type": "cable connector product datasheet",
      "models": { "schema_discovery": "glm-5.1:cloud", "vision": "kimi-k2.7-code:cloud",
                  "embedding": "nomic-ai/nomic-embed-text-v1.5" },
      "user_keys": ["product family", "Voltage"]
    },
    "voltage": [
      { "confidence": 1.0, "models": ["deepseek-v4-flash:cloud", "gemma4:31b-cloud"],
        "qualifier": "Maximum", "value": 60.0, "value_unit": "V AC",
        "metadata": { "section_type": "text", "specified": true } }
    ]
  }
}
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

With uv:

```bash
uv sync --extra dev --group lint
uv run pytest
uv run ruff check src tests
uv run ruff format --check src tests
```

CI runs ruff and pytest (Python 3.11 and 3.12) on every pull request.

## License

MIT, see [LICENSE](LICENSE).
