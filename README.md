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
 └─► convert    (docling)            → markdown + items.json
      └─► vocab  (YAKE)              → vocab.json
           └─► classify_sections (RandomForest)  → sections.json
                └─► tables (complexity scoring)   → tables.json
                     └─► embed (sentence-transformers) → *_embeddings.npy + row_map
                          └─► search (cosine similarity)    → matches.json
                               └─► tables.refine (vision LLM, lazy) → cleaned HTML tables
                                    └─► extract (schema discovery + LLM) → output/<stem>_extraction.json
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
  are de-duplicated, preferring table-sourced records over free-text ones. After the fact,
  `rater.py` scores how much the models agreed and attaches a confidence to every record.

---


### Orchestration & configuration

**`main.py`** : the orchestrator. `process_single_pdf` calls every step in order
for one PDF, then re-invokes the table refiner *only* for tables that search actually
matched (this is the "lazy vision" trick). `process_batch` loops the whole thing over a
directory of PDFs. This is the file you run.

**`config.py`** : every path, model name, and threshold in one place. Paths are derived
relative to the repo root, so the project is portable. Model names, the embedding model,
the Ollama URL, and a couple of parallelism knobs are overridable via environment
variables; the numeric thresholds are plain constants you can edit. It also auto-selects the embedding device
(CUDA → Apple MPS → CPU).

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

**`classify_sections.py`** : loads the pre-trained RandomForest classifier and tags every
text item as `header`, `list`, `table`, `text`, or `title`. The result, `sections.json`,
is the canonical content stream every downstream step reads. *Choice:* a small supervised
classifier (rather than trusting Docling's labels directly) because Docling mislabels
multi-sentence captions, sub-headers, etc.

**`tables.py`** : two jobs:
1. `process_tables` scores each table's *complexity* (HTML/merge markers, ragged column
   counts, multi-line or very long cells, scientific notation). Anything over
   `TABLE_COMPLEXITY_THRESHOLD` is routed `complex`; the rest are `simple` and left alone.
2. `refine_matched_tables` (called from `main.py` **after** search) takes the complex
   tables that matched a key, re-renders their bounding box straight out of the PDF with
   PyMuPDF, and asks the vision LLM to rebuild them as clean HTML — falling back to a
   text-only LLM if vision fails. Docling mangles merged/nested tables; this recovers
   them, but only for tables we care about.

**`embed.py`** : turns text into vectors with a SentenceTransformer
(`nomic-embed-text-v1.5` by default). It embeds two things in one batch: whole sections,
and individual table rows. The row parser  detects group-header rows and Min/Typ/Max sub-headers and prepends the group label so each row embeds with proper context. User keys are expanded against the YAKE vocab and any discovered acronym
synonyms before being embedded. Sections are stacked *before* rows in the output matrix;
`search.py` relies on that ordering.


**`search.py`** : cosine similarity between each key vector and the combined
section+row matrix. Anything above `SIMILARITY_THRESHOLD` (capped at `SEARCH_TOP_K`
hits) becomes a match. Row matches are flagged and carry their row number + group header
so the extractor can later filter the original table down to the right rows. Output:
`matches.json`, keyed by your original (un-expanded) key.

**`extract.py`** : the LLM stage, and the densest file:
- merges the cleaned-up table HTML back into the section list and attaches each table's
  nearest heading for context;
- renders the matched sections into the prompt, shrinking long tables down to just the
  matched parameter group via `table_filter`;
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

**`classify.py`** : a thin wrapper that loads the trained RandomForest, the two TF-IDF
vectorizers, and the Docling category list from `models/`, and runs inference.

**`table_filter.py`** : once we know which rows matched, this trims a table down to just
the relevant parameter group(s) before sending it to the extractor (a big win on long
datasheet tables). Two implementations — one for markdown tables, one for HTML — the HTML
one understands `rowspan` so merged first-column cells stay attached to their group.

**`abbreviations.py`** : scans the document for explicit acronym definitions
("Common Mode Rejection Ratio (CMRR)") and builds a synonym map, so a key typed either way
matches. Again: derived from the document, not hardcoded.

**`rater.py`** : a post-extraction analysis, *not* part of the pipeline yet. It
reads every `output/*.json`, groups the records by `(file, key)` across the models that
produced them, and scores agreement. Two records from different models count as the **same
fact** when their value *and* unit match exactly and their descriptor text (qualifier,
condition…) is at least `THRESHOLD` (0.85) similar by Levenshtein ratio —> fuzzy on the
wording because models phrase conditions differently (`TA = −40°C` vs `T_A = -40°C`), exact
on the number. Each fact gets a **confidence** = how many models found it ÷ number of models
(3/3 → 1.0, 2/3 → 0.67…), and it prints pairwise **Cohen's kappa** as a global inter-rater
agreement metric.

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


### Set up the models (Ollama)

The pipeline calls several models by name. Pull them, or point the pipeline at models you
already have via environment variables. Defaults (from `config.py`):

| Role               | Default model                   | Env var               |
|--------------------|---------------------------------|-----------------------|
| Schema discovery   | `glm-5.1:cloud`                 | `OLLAMA_SCHEMA_MODEL` |
| Extraction (×3, in parallel) | `gemma4:31b-cloud`, `deepseek-v4-flash:cloud`, `gpt-oss:120b-cloud` | (edit `EXTRACTION_MODELS`) |
| Vision (table cleanup) | `qwen3-vl:235b-instruct-cloud` | `OLLAMA_VISION_MODEL` |
| Embeddings         | `nomic-ai/nomic-embed-text-v1.5` | `EMBEDDING_MODEL`   |

```bash
ollama list                         # see what you have
# point at your own models instead of the defaults, e.g.:
export OLLAMA_EXTRACT_MODEL=llama3.1:8b
export OLLAMA_BASE_URL=http://localhost:11434/v1
```

If a named model isn't available, the extract step will retry and then fail for that
model — check `ollama list` first.

---

## Usage

```bash
# one PDF
structured-extract --pdf pdf/1167fc.pdf --keys gain CMRR "supply voltage"

# a whole directory of PDFs
structured-extract --pdf-dir pdf/ --keys revenue "net income"

# equivalent, without the console script:
python -m structured_data_extraction.main --pdf pdf/1167fc.pdf --keys gain CMRR
```

- Wrap multi-word keys in quotes.
- If you omit `--keys`, it prompts for a comma-separated list interactively.



### Testing just the extraction stage (fast loop)

A cold run does conversion + embedding + search before it extracts, which is slow. Once
those caches exist in `intermediate/`, you can re-run **only** the LLM extraction against
them — the right command when you're iterating on prompts or schema:

```bash
python -m structured_data_extraction.extract --stem 1167fc --keys gain CMRR
```

Every step is runnable standalone this way (`-m ...search`, `-m ...embed`, etc.).


### Scoring model agreement (confidence)

Once extraction has written the per-model outputs to `output/`, score how much the models
agreed on each value:

```bash
python -m structured_data_extraction.rater
```

It reads everything in `output/`, prints per-key confidence (how many facts all the models
found vs. only some), lists the disagreements, and reports pairwise Cohen's kappa. No
arguments — it just analyses whatever outputs are present.


## License

MIT — see [LICENSE](LICENSE).
