"""Pipeline orchestrator : drives the full extraction workflow end to end.

Processes one PDF or a directory of PDFs through every stage in order:
  convert → vocab → classify → tables → embed → search → extract → rate → records

Each stage writes its output to intermediate/ or output/ and is skipped on rerun
if the artefact already exists (unless --force is set). Key-dependent caches
(embeddings, matches, extractions) are automatically invalidated when the user
changes their key list between runs.

After extraction, finalize_outputs() fuses the per-model results via rater.py,
writes provenance records, and optionally filters by confidence tier.
"""
import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from . import config, records, filter_records, matching, rater
from .paths import inter, read_json
from .convert import convert_pdf
from .vocab import extract_vocab
from .classify import classify_sections
from .tables import process_tables, refine_matched_tables, refine_symbol_only_tables
from .embed import embed_sections_and_keys
from .search import search
from .extract import extract_parallel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def _invalidate_stale_key_caches(stem, user_keys):
    keys_path = inter(stem, "original_keys.json")
    if not keys_path.exists():
        return
    if sorted(user_keys) == sorted(read_json(keys_path)):
        return
    logger.info(f"Keys changed; invalidating key-dependent caches for {stem}")
    for name in ("key_embeddings.npy", "key_texts.json", "original_keys.json",
                  "matches.json", "schema.json", "run.json"):
        p = inter(stem, name)
        if p.exists():
            p.unlink()
    for p in Path(config.OUTPUT_DIR).glob(f"{stem}_extraction_*.json"):
        p.unlink()


def process_single_pdf(pdf_path, user_keys, user_schema=None):
    """Run the full extraction pipeline on one PDF.

    Executes every stage in order: convert → vocab → classify → tables →
    embed → search → extract, writing intermediate artefacts to
    INTERMEDIATE_DIR and per-model extractions to OUTPUT_DIR. Each stage is
    skipped when its output already exists on disk (unless --force is set).
    If the user's key list has changed since the last run, key-dependent
    caches are automatically invalidated.

    Args:
        pdf_path: Path to the input PDF (str or Path).
        user_keys: List of extraction keys, e.g. ["gain bandwidth", "PSRR"].
                   Cleaned and validated internally (whitespace, quotes, and
                   duplicates are handled; raises ValueError on unusable input).

    Returns:
        A list of dicts on success — one dict per extraction model. Each dict
        maps a normalised field name to a list of extracted records::

            [
                {                                       # model 1
                    "gain_bandwidth": [
                        {"qualifier": "...", "condition": "...",
                         "value": 4.0, "unit": "MHz",
                         "metadata": {"source": "...", "section_type": "table",
                                      "specified": True, "document_type": "...",
                                      "model": "gemma4:31b-cloud"}},
                        ...
                    ],
                    ...
                },
                { ... },                                # model 2
            ]

        Returns None if the PDF cannot be found or conversion fails.

        These per-model dicts are the raw, un-fused extractions. To get the
        final confidence-scored output, call finalize_outputs().
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return None

    user_keys = matching.validate_keys(user_keys)
    stem = pdf_path.stem
    int_dir = Path(config.INTERMEDIATE_DIR)
    sections_path = int_dir / f"{stem}_sections.json"
    tables_file = int_dir / f"{stem}_tables.json"
    logger.info(f"=== Processing {pdf_path.name} ===")

    timings = {}

    @contextmanager
    def timed(label):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            timings[label] = timings.get(label, 0.0) + time.perf_counter() - t0

    with timed("convert"):
        md_path = convert_pdf(str(pdf_path))
    if md_path is None:
        logger.error(f"Conversion failed for {pdf_path.name}")
        return None

    with timed("vocab"):
        extract_vocab(md_path)
    with timed("classify"):
        classify_sections(stem)
    with timed("tables"):
        process_tables(str(sections_path))

    # tables Docling stripped to bare symbols can't be retrieved, so they'd never reach the
    # lazy-vision step below; de-garble them now so their recovered text gets embedded
    if tables_file.exists():
        with timed("pre-refine"):
            refine_symbol_only_tables(str(tables_file), pdf_path=str(pdf_path))

    _invalidate_stale_key_caches(stem, user_keys)
    with timed("embed"):
        embed_sections_and_keys(stem, user_keys=user_keys)

    with timed("search"):
        matches = search(stem, original_keys=user_keys)

    # refine matched complex tables with vision model
    matched_idxs = {
        match.get("section_idx", 0)
        for key_matches in matches.values()
        for match in key_matches
    }

    if matched_idxs and tables_file.exists():
        logger.info("Refining matched complex tables...")
        with timed("refine"):
            refine_matched_tables(str(tables_file), matched_idxs, pdf_path=str(pdf_path))

    with timed("extract"):
        results = extract_parallel(stem, user_keys, user_schema=user_schema)

    _log_run_summary(stem, timings, results, matches)
    logger.info(f"=== Done with {pdf_path.name} ===")
    return results


def _log_run_summary(stem, timings, results, matches):
    """End-of-run legibility: one line per stage with its wall-clock time, then a record count
    broken down by the section type each record came from (table/text/list)."""
    by_source, total = {}, 0
    for res in results:
        for records in res.values():
            for record in records:
                total += 1
                src = (record.get("metadata") or {}).get("section_type") or "unknown"
                by_source[src] = by_source.get(src, 0) + 1
    sources = ", ".join(f"{k}={v}" for k, v in sorted(by_source.items())) or "none"
    matched = sum(1 for hits in matches.values() if hits)
    stages = "".join(f"\n  {label:<11}{secs:6.1f}s" for label, secs in timings.items())
    logger.info(
        f"=== Run summary: {stem} ==={stages}"
        f"\n  {'TOTAL':<11}{sum(timings.values()):6.1f}s"
        f"\n  records: {total} ({sources}) from {len(results)} model(s); "
        f"keys matched: {matched}/{len(matches)}"
    )


def finalize_outputs(levels=None):
    """Fuse per-model extractions into the final deliverables.

    Reads every output/<stem>_extraction_<model>.json written by extract_parallel,
    clusters matching records across models into facts, and writes three files to
    OUTPUT_DIR:

      fused.json
        The primary output. Structure: {stem: {field: [fact, ...]}}.
        Each fact is a dict with the extracted fields (value, unit, qualifier,
        condition, ...) plus:
          - "confidence": float in (0, 1] — fraction of configured models that
            independently produced this fact (e.g. 2 of 3 → 0.67). With a single
            model every fact is 1.0.
          - "models": list[str] — which models agreed on this fact.
          - "metadata": {"section_type": str, "specified": bool}.
        Facts are sorted highest-confidence first within each field.

      rating_report.json
        Inter-model agreement statistics: total fact count, full-agreement count,
        pairwise Cohen's kappa and raw agreement for each model pair, and the list
        of facts where models disagreed. Useful for auditing model consistency.

      records.json
        Provenance-stamped envelopes wrapping fused.json: one entry per document,
        adding source_file, document_type, the models used, and the user's
        original keys. This is the archival output.

    After writing, if multiple extraction models are configured, the user is
    offered (or given, via `levels`) a confidence filter to keep only trusted
    tiers.

    Args:
        levels: Confidence tiers to keep (e.g. [1.0, 0.75]) or ["all"].
                None triggers an interactive prompt (multi-model only).
    """
    by_file = rater.load_outputs()
    if not by_file:
        logger.warning("No extraction outputs found — skipping rating + records step.")
        return
    rater.rate(by_file)
    out_path, n = records.write_records(config.OUTPUT_DIR)
    logger.info(f"Wrote records for {n} document(s) → {out_path}")
    _offer_filter(out_path, levels)


def _offer_filter(records_path, levels):
    """user filter records by confidence at the end of a run. If confidence came in as an argument
    (--levels) uses them, otherwise offers the interactive menu. One model means every record
    is confidence 1.0, so there's nothing to filter and we bail."""
    if len(config.EXTRACTION_MODELS) < 2:
        return
    filter_records.run(records_path, levels=levels)


def process_batch(pdf_dir, user_keys, user_schema=None):
    pdf_dir = Path(pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"Directory not found: {pdf_dir}")
        return []

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No PDFs found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdfs)} PDFs")
    results = []
    for pdf in pdfs:
        try:
            result = process_single_pdf(str(pdf), user_keys, user_schema=user_schema)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed {pdf.name}: {e}")

    logger.info(f"Processed {len(results)}/{len(pdfs)} files")
    return results


def _dump_templates(dest):
    """Copy the packaged prompt templates to `dest` for editing, then the user points
    TEMPLATES_DIR at that folder to use them. Only the built-in .j2 defaults are written;
    read_text/write_text keeps this working even if the package is imported from a zip."""
    src = resources.files("structured_data_extraction") / "templates"
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for t in src.iterdir():
        if t.name.endswith(".j2"):
            (dest / t.name).write_text(t.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Document extraction pipeline")
    parser.add_argument("path", nargs="?", help="Path to a PDF file or a directory of PDFs")
    parser.add_argument("--keys", nargs="+", help="Keys to extract")
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Recompute every step, ignoring cached files")
    parser.add_argument("--levels", nargs="+", metavar="LVL",
                        help="confidence tiers to keep (e.g. 1.0 0.75, or 'all'); "
                             "omit to be prompted after extraction")
    parser.add_argument("--schema", metavar="FILE",
                        help="Path to a JSON schema file (same format as intermediate/*_schema.json); "
                             "skips schema discovery. Keys are derived from the schema, so --keys is optional")
    parser.add_argument("--dump-templates", metavar="DIR",
                        help="Write the built-in prompt templates to DIR for editing, then exit; "
                             "point TEMPLATES_DIR at that folder to use your edits")
    args = parser.parse_args()

    if args.dump_templates:
        dest = _dump_templates(args.dump_templates)
        print(f"Wrote templates to {dest}\nSet TEMPLATES_DIR={dest} to use your edits.")
        return
    if not args.path:
        parser.error("a PDF file or directory is required (unless using --dump-templates)")

    config.FORCE = args.force

    user_schema = None
    if args.schema:
        schema_path = Path(args.schema)
        if not schema_path.exists():
            print(f"Schema file not found: {schema_path}")
            sys.exit(1)
        user_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        if not isinstance(user_schema, dict) or "schema" not in user_schema or "document_type" not in user_schema:
            print("Schema file must contain 'document_type' and 'schema' keys")
            sys.exit(1)

    user_keys = args.keys
    if not user_keys and user_schema:
        user_keys = list(user_schema["schema"].keys())
    elif not user_keys:
        print("Enter keys to extract (comma-separated):")
        raw = input("> ").strip()
        user_keys = [k.strip() for k in raw.split(",") if k.strip()]

    try:
        user_keys = matching.validate_keys(user_keys or [])
    except ValueError as e:
        print(f"Invalid keys: {e}")
        sys.exit(1)

    config.OUTPUT_DIR = args.output

    if Path(args.path).is_dir():
        results = process_batch(args.path, user_keys, user_schema=user_schema)
        if results:
            finalize_outputs(args.levels)
        print(f"Extracted from {len(results)} files → {config.OUTPUT_DIR}")
        if not results:
            print("No documents produced any extraction.")
            sys.exit(1)
    else:
        result = process_single_pdf(args.path, user_keys, user_schema=user_schema)
        if result:
            finalize_outputs(args.levels)
            print(json.dumps(result, indent=2, ensure_ascii=False))  #document them well
        else:
            print("Extraction failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
