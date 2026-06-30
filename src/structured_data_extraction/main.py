import argparse
import json
import logging
import sys
import time
from pathlib import Path
from . import config, fdo_records, matching, rater
from .convert import convert_pdf
from .vocab import extract_vocab
from .classify import classify_sections
from .tables import process_tables, refine_matched_tables, refine_symbol_only_tables
from .embed import embed_sections_and_keys
from .search import search
from .extract import extract_parallel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def process_single_pdf(pdf_path, user_keys):
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

    def stage(label, fn):
        t0 = time.perf_counter()
        try:
            return fn()
        finally:
            timings[label] = timings.get(label, 0.0) + time.perf_counter() - t0

    md_path = stage("convert", lambda: convert_pdf(str(pdf_path)))
    if md_path is None:
        logger.error(f"Conversion failed for {pdf_path.name}")
        return None

    stage("vocab", lambda: extract_vocab(md_path))
    stage("classify", lambda: classify_sections(stem))
    stage("tables", lambda: process_tables(str(sections_path)))

    # tables Docling stripped to bare symbols can't be retrieved, so they'd never reach the
    # lazy-vision step below; de-garble them now so their recovered text gets embedded
    if tables_file.exists():
        stage("pre-refine", lambda: refine_symbol_only_tables(str(tables_file), pdf_path=str(pdf_path)))

    stage("embed", lambda: embed_sections_and_keys(stem, user_keys=user_keys))

    matches = stage("search", lambda: search(stem, original_keys=user_keys))

    # refine matched complex tables with vision model
    matched_idxs = {
        match.get("section_idx", 0)
        for key_matches in matches.values()
        for match in key_matches
    }

    if matched_idxs and tables_file.exists():
        logger.info("Refining matched complex tables...")
        stage("refine", lambda: refine_matched_tables(str(tables_file), matched_idxs, pdf_path=str(pdf_path)))

    results = stage("extract", lambda: extract_parallel(stem, user_keys))

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


def finalize_outputs():
    """Fuse the per-model extractions into the user report, agreement stats, and FDO records.

    Reads every extraction.json in the output dir, so it works for one model or many
    (a single model just passes through with confidence 1.0 and no agreement rows).
    """
    by_file = rater.load_outputs()
    if not by_file:
        logger.warning("No extraction outputs found — skipping rating + FDO step.")
        return
    rater.rate(by_file)
    out_path, n = fdo_records.write_records(config.OUTPUT_DIR)
    logger.info(f"Wrote FDO records for {n} document(s) → {out_path}")


def process_batch(pdf_dir, user_keys):
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
            result = process_single_pdf(str(pdf), user_keys)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed {pdf.name}: {e}")

    logger.info(f"Processed {len(results)}/{len(pdfs)} files")
    return results


def main():
    parser = argparse.ArgumentParser(description="Document extraction pipeline")
    parser.add_argument("path", help="Path to a PDF file or a directory of PDFs")
    parser.add_argument("--keys", nargs="+", help="Keys to extract")
    parser.add_argument("--output", default=config.OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Recompute every step, ignoring cached files")
    args = parser.parse_args()

    config.FORCE = args.force
    user_keys = args.keys
    if not user_keys:
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
        results = process_batch(args.path, user_keys)
        if results:
            finalize_outputs()
        print(f"Extracted from {len(results)} files → {config.OUTPUT_DIR}")
        if not results:
            print("No documents produced any extraction.")
            sys.exit(1)
    else:
        result = process_single_pdf(args.path, user_keys)
        if result:
            finalize_outputs()
            print(json.dumps(result, indent=2, ensure_ascii=False))  #document them well
        else:
            print("Extraction failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
