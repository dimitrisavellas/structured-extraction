# Structured Data Extraction

Pull user-specified fields out of arbitrary PDFs and return them as
structured JSON with provenance.

The whole point is to be **domain-agnostic**. There is no checked-in list of datasheet
parameters or finance line-items. Everything domain-specific (vocabulary, acronyms, the
shape of each field) is derived from *the document itself* at runtime. Hand the pipeline
a PDF it has never seen, in a domain it was never built for, and it should still find your
keys.

---

## How it works 

Feed it a **PDF** and a list of **keys**. It converts the PDF to text, learns the
document's own vocabulary, splits it into typed sections, embeds everything, finds the
sections/rows that are semantically closest to your keys, cleans up any messy tables it
matched, and finally asks an LLM to read just those passages and return structured values.

```
PDF
 └─► convert  (docling)                → markdown + items.json
      └─► vocab  (YAKE)                → vocab.json
           └─► classify (RandomForest) → sections.json
                └─► tables (complexity scoring)            → tables.json
                          └─► embed (sentence-transformers)       → *_embeddings.npy + row_map
                               └─► search (cosine similarity)     → matches.json
                                    └─► refine matched tables (vision LLM, lazy) → cleaned HTML
                                         └─► extract (schema discovery + LLM) → output/<stem>_extraction_<model>.json
```

Each step writes its output to disk and is **skipped on a re-run if the file already
exists**. That makes iteration cheap (you don't re-run docling every time you tweak a
prompt). 

---

## Why this design

A few deliberate choices worth calling out, because they're what make the pipeline
general.

- **The document teaches the pipeline its own vocabulary.** Instead of shipping a
  glossary, `vocab.py` runs unsupervised keyword extraction (YAKE) over each PDF. A
  one-word key like `gain` then gets expanded with the document's own related terms
  (`gain error`, `gain drift`) before we search. This is the single biggest reason a
  datasheet pipeline also works on an earnings report.
- **Search happens in embedding space, not by string matching.** "CMRR" and "common-mode
  rejection ratio" don't share characters but they're neighbours in embedding space.
  Cosine similarity over sentence-transformer embeddings is what lets a user type a
  loose key and still hit the right passage.
- **Table rows are first-class search targets.** A datasheet's answer is usually one row
  in a 40-row table. We embed each row *with its group header prepended* so a row reads
  like `Gain Error > G=1 | 0.01 | 0.04 %` instead of a context-free `0.01 | 0.04 %`.
- **The expensive model is used lazily.** Vision-LLM table cleanup only runs on tables
  that actually matched a key, never on every table in the document. Cost scales with
  how much you asked for, not with document size.
- **The LLM is boxed in.** `extract.py` discovers a per-key schema, builds a Pydantic
  model from it on the fly, and hands it to [Instructor](https://python.useinstructor.com/)
  so the model is forced to return validated structured output instead of free prose.
- **Several models vote.** Extraction runs across a few LLMs in parallel and the results
  are de-duplicated, preferring table-sourced records over free-text ones. The pipeline then
  runs `rater.py` to score how much the models agreed and attach a confidence to every record
  (a single model still works — every fact just gets confidence 1.0).

---


### Orchestration & configuration

**`main.py`** : the orchestrator. `process_single_pdf` calls every step in order
for one PDF, then re-invokes the table refiner *only* for tables that search actually
matched (this is the "lazy vision" trick). `process_batch` loops the whole thing over a
directory of PDFs. This is the file you run.

**`config.py` + `.env`** : the tunable values (provider URL/key, model names, thresholds,
parallelism knobs) live in a `.env` file at the repo root; `config.py` loads it and exposes
the values to the rest of the code. Paths are derived relative to the repo root, so the
project is portable, and the embedding device is auto-selected (CUDA → Apple MPS → CPU).
The provider is any OpenAI-compatible endpoint — set `LLM_BASE_URL` / `LLM_API_KEY` to point
at Ollama, OpenAI, vLLM, etc. The `.env` file is gitignored (it can hold an API key);
edit it in place. Shell environment variables, if set, override the file.

### Pipeline steps

**`convert.py`** : runs [Docling](https://github.com/docling-project/docling) on the PDF.
Produces the markdown export plus a flat `items.json` where each item carries its Docling
label (`text`, `section_header`, `table`, `picture`…), page number, and bounding box.
Tables are exported as markdown so they survive the round-trip, and bbox handling is aware
that Docling uses a bottom-left coordinate origin (PyMuPDF, used later, uses top-left).
*Choice:* OCR is off and the table model runs in `FAST` mode — roughly 2× faster on
born-digital PDFs like datasheets and reports, which is the target. Turn OCR on if you
need scanned documents.

**`vocab.py`** : extracts the document's domain vocabulary with YAKE (unsupervised, up to
4-grams). This is the "learn the domain from the document" step. Lower YAKE score = more
important.

**`classify.py`** : loads the pre-trained RandomForest classifier (plus its two TF-IDF
vectorizers and the Docling category list from `models/`) and tags every text item as
`header`, `list`, `table`, `text`, or `title`. The result, `sections.json`, is the
canonical content stream every downstream step reads. *Choice:* a small supervised
classifier (rather than trusting Docling's labels directly) because Docling mislabels
multi-sentence captions, sub-headers, etc.

**`tables.py`** : three jobs:
1. `process_tables` scores each table's *complexity* (HTML/merge markers, ragged column
   counts, multi-line or very long cells, scientific notation). Anything over
   `VISION_COMPLEXITY_THRESHOLD` is routed `complex`; the rest are `simple` and left alone.
2. `refine_symbol_only_tables` (called from `main.py` **before** embedding) rebuilds the
   complex tables Docling stripped down to bare symbols. Those rows carry no words, so search
   could never retrieve them — meaning they'd never reach the lazy-vision step below.
   De-garbling them up front gets their recovered text into the embeddings.
3. `refine_matched_tables` (called from `main.py` **after** search) takes the complex
   tables that matched a key, re-renders their bounding box straight out of the PDF with
   PyMuPDF, and asks the vision LLM to rebuild them as clean HTML — falling back to a
   text-only LLM if vision fails. Docling mangles merged/nested tables; this recovers
   them, but only for tables we care about.

**`embed.py`** : turns text into vectors with a SentenceTransformer
(`nomic-embed-text-v1.5` by default). It embeds two things in one batch: whole sections,
and individual table rows. The row parser detects group-header rows and Min/Typ/Max
sub-headers and prepends the group label so each row embeds with proper context. User keys
are expanded against the YAKE vocab before being embedded. nomic needs task prefixes, so
documents are encoded as `search_document:` and keys as `search_query:` — without them all
the similarities collapse toward each other and the threshold stops discriminating. Sections
are stacked *before* rows in the output matrix; `search.py` relies on that ordering.


**`search.py`** : cosine similarity between each key vector and the combined
section+row matrix. Anything above `SIMILARITY_THRESHOLD` (capped at `SEARCH_TOP_K`
hits) becomes a match. Row matches are flagged and carry their row number + group header,
so the extractor knows which table (and which group within it) the hit came from. Output:
`matches.json`, keyed by your original (un-expanded) key.

**`extract.py`** : the LLM stage, and the densest file:
- merges the cleaned-up table HTML back into the section list and attaches each table's
  nearest heading for context;
- renders the matched sections into the prompt (truncating long prose, leaving tables whole);
- `discover_schema` asks an LLM what fields each key should have (qualifier, condition,
  value, unit…) and what kind of document this is;
- builds a Pydantic model from that schema on the fly and runs the extractor LLM through
  Instructor with retries;
- de-duplicates records (preferring table-sourced over free-text) and writes
  `output/<stem>_extraction_<model>.json` with full provenance.

### Supporting pieces

**`features.py`** : the single source of truth for the section classifier's features:
`[word-tfidf | char-tfidf | 30 regex flags | docling context]`. The regex flags encode
"does this look like a heading / list item / figure caption" heuristics; the context
features one-hot the item's own Docling label plus its neighbours' labels and the distance
to the nearest section header.

**`paths.py`** : tiny helpers for the `intermediate/` and `output/` filename conventions
(`<stem>_<name>`) plus JSON read/write, so every step names its cache files the same way.

**`matching.py`** : the equivalence logic — when do two extracted records describe the
**same fact**? Numbers and units must match exactly (after Unicode/dash normalisation); the
descriptor text (qualifier, condition…) only has to clear a rapidfuzz similarity of
`FUZZ_THRESHOLD` (85), because models phrase conditions differently (`TA = −40°C` vs
`T_A = -40°C`). Empty fields act as wildcards, so a sparse record still matches a fuller one.

**`rater.py`** : runs automatically at the end of the pipeline (and can be re-run on its
own). It reads every `output/*_extraction_*.json`, clusters the records into facts across the
models that produced them (via `matching.same_fact`), and scores agreement. Each fact gets a
**confidence** = how many models found it ÷ number of models (3/3 → 1.0, 2/3 → 0.67…). It
writes three files to `output/` — `rating_report.json` (every fact with its confidence and
disagreements), `fused.json` (the merged view), and `user_report.json` (only the facts a
majority of models agreed on) — and prints pairwise **Cohen's kappa** as a global
inter-rater agreement metric. With a single extraction model every fact passes through at
confidence 1.0 and the kappa step is skipped.

**`fdo_records.py`** (in the package) : the final pipeline step. It wraps `user_report.json` into one
FAIR-Digital-Object-style envelope per document (`output/fdo_records.json`), adding
provenance — source file, document type, and the schema/extraction/vision/embedding models
used.

### Templates

**`templates/schema_discovery.j2`** : prompt that asks the LLM for each key's field shape
and the document type.
**`templates/extraction.j2`** : prompt for the actual value extraction.

---

## Installation

**Requirements:** Python ≥ 3.11, and a running [Ollama](https://ollama.com) instance for
the LLM calls. First run also downloads the embedding model from Hugging Face. 

```bash
# 1. clone, then create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate            

# 2. install the package and its dependencies
pip install -e .                    # or: pip install -r requirements.txt
```


### Set up the provider and models

The pipeline talks to any **OpenAI-compatible** endpoint. Point it at your provider and pick
your models by editing `.env` at the repo root (values shown are the defaults):

| Role               | `.env` key            | Default                          |
|--------------------|-----------------------|----------------------------------|
| Provider endpoint  | `LLM_BASE_URL`        | `http://localhost:11434/v1`      |
| Provider API key   | `LLM_API_KEY`         | `ollama`                         |
| Schema discovery   | `SCHEMA_MODEL`        | `glm-5.1:cloud`                  |
| Extraction (1+, in parallel) | `EXTRACTION_MODELS` | `gemma4:31b-cloud,deepseek-v4-flash:cloud,glm-4.7:cloud` |
| Vision (table cleanup) | `VISION_MODEL`    | `kimi-k2.7-code:cloud`           |
| Embeddings         | `EMBEDDING_MODEL`     | `nomic-ai/nomic-embed-text-v1.5` |

`EXTRACTION_MODELS` is a comma-separated list — list several models to extract in parallel
and cross-check their agreement, or just **one** model if you prefer (the pipeline adapts and
skips the agreement step). The first model in the list is also the "primary" one used wherever
a single model is needed, such as text-based table cleanup. To use a non-Ollama provider, e.g. OpenAI:

```bash
# in .env
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
EXTRACTION_MODELS=gpt-4o-mini
```

If a named model isn't available, the extract step will retry and then fail for that
model — check that the model exists on your provider first.

---

## Usage

```bash
# one PDF
structured-extract pdf/1167fc.pdf --keys gain CMRR "supply voltage"

# a whole directory of PDFs
structured-extract pdf/ --keys revenue "net income"

# equivalent, without the console script:
python -m structured_data_extraction.main pdf/1167fc.pdf --keys gain CMRR
```

- Wrap multi-word keys in quotes.
- If you omit `--keys`, it prompts for a comma-separated list interactively.

A full run writes the per-model extractions to `output/<stem>_extraction_<model>.json`, then
automatically fuses them into `user_report.json`, `fused.json`, `rating_report.json`, and the
provenance-stamped `fdo_records.json` — no extra commands needed.



### Testing just the extraction stage (fast loop)

A cold run does conversion + embedding + search before it extracts, which is slow. Once
those caches exist in `intermediate/`, you can re-run **only** the LLM extraction against
them — the right command when you're iterating on prompts or schema:

```bash
python -m structured_data_extraction.extract --stem 1167fc --keys gain CMRR
```

Every step is runnable standalone this way (`-m ...search`, `-m ...embed`, etc.).


### Re-scoring model agreement on its own

The pipeline already runs this at the end, but you can re-score whatever outputs are present
in `output/` without re-extracting:

```bash
python -m structured_data_extraction.rater     # rebuild rating_report / fused / user_report
python -m structured_data_extraction.fdo_records   # rebuild fdo_records.json from user_report
```

`rater` clusters the records into facts, writes `rating_report.json`, `fused.json`, and
`user_report.json`, and prints per-fact confidence and pairwise Cohen's kappa. Both take no
arguments — they just analyse whatever outputs are present.


## License

MIT — see [LICENSE](LICENSE).
