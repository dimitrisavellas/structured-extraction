import argparse
import json
import logging
import sys
from pathlib import Path
from . import config
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

    stem = pdf_path.stem
    int_dir = Path(config.INTERMEDIATE_DIR)
    sections_path = int_dir / f"{stem}_sections.json"
    tables_file = int_dir / f"{stem}_tables.json"
    logger.info(f"=== Processing {pdf_path.name} ===")

    md_path = convert_pdf(str(pdf_path))
    if md_path is None:
        logger.error(f"Conversion failed for {pdf_path.name}")
        return None

    extract_vocab(md_path)
    classify_sections(stem)

    process_tables(str(sections_path), pdf_path=str(pdf_path))

    # tables Docling stripped to bare symbols can't be retrieved, so they'd never reach the
    # lazy-vision step below; de-garble them now so their recovered text gets embedded
    if tables_file.exists():
        refine_symbol_only_tables(str(tables_file))

    embed_sections_and_keys(stem, user_keys=user_keys)

    matches = search(stem, original_keys=user_keys)

    # refine matched complex tables with vision model
    matched_idxs = {
        match.get("section_idx", 0)
        for key_matches in matches.values()
        for match in key_matches
    }

    if matched_idxs and tables_file.exists():
        logger.info("Refining matched complex tables...")
        refine_matched_tables(str(tables_file), matched_idxs)

    results = extract_parallel(stem, user_keys)

    logger.info(f"=== Done with {pdf_path.name} ===")
    return results


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
    args = parser.parse_args()

    user_keys = args.keys
    if not user_keys:
        print("Enter keys to extract (comma-separated):")
        raw = input("> ").strip()
        user_keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not user_keys:
            print("No keys provided.")
            sys.exit(1)

    config.OUTPUT_DIR = args.output

    if Path(args.path).is_dir():
        results = process_batch(args.path, user_keys)
        print(f"Extracted from {len(results)} files → {config.OUTPUT_DIR}")
    else:
        result = process_single_pdf(args.path, user_keys)
        if result:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print("Extraction failed.")
            sys.exit(1)


if __name__ == "__main__":
    main()
