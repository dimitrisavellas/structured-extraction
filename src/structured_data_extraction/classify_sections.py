import argparse
import json
import logging
from pathlib import Path
from . import config
from .classify import SectionClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def classify_sections(pdf_stem):
    items_path = Path(config.INTERMEDIATE_DIR) / f"{pdf_stem}_items.json"
    out_path = Path(config.INTERMEDIATE_DIR) / f"{pdf_stem}_sections.json"

    if not items_path.exists():
        raise FileNotFoundError(f"Items not found: {items_path}. Run convert.py first.")

    if out_path.exists():
        logger.info(f"Skipping {pdf_stem} — sections already exist")
        return json.loads(out_path.read_text(encoding="utf-8"))

    items = json.loads(items_path.read_text(encoding="utf-8"))
    logger.info(f"Classifying {len(items)} items for {pdf_stem}...")

    model_dir = Path(config.SECTION_CLASSIFIER_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(f"Classifier not found at {model_dir}. "
                                "Expected the trained section_classifier model to ship with the project.")

    # only classify items that have text
    text_chunks = []
    for i, item in enumerate(items):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        text_chunks.append({
            "section_id": i,
            "docling_label": item.get("label", "text"),
            "text": text,
            "page": item.get("page"),
            "bbox": item.get("bbox"),
        })

    classifier = SectionClassifier(model_dir=str(model_dir))
    predicted = classifier.classify(text_chunks)

    # build output keeping original item order
    by_idx = {chunk["section_id"]: chunk for chunk in predicted}
    sections = []
    for i, item in enumerate(items):
        if i in by_idx:
            chunk = by_idx[i]
            sections.append({
                "type": chunk["section_type"],
                "content": chunk["text"],
                "docling_label": chunk["docling_label"],
                "page": chunk["page"],
                "bbox": chunk["bbox"],
                "section_idx": i,
            })
        elif item.get("label") == "picture":
            sections.append({
                "type": "picture",
                "content": "",
                "docling_label": item.get("label"),
                "page": item.get("page"),
                "bbox": item.get("bbox"),
                "section_idx": i,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Classified {len(sections)} sections → {out_path}")
    return sections


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sections")
    parser.add_argument("--stem", required=True)
    args = parser.parse_args()

    result = classify_sections(args.stem)
    counts = {}
    for s in result:
        counts[s["type"]] = counts.get(s["type"], 0) + 1
    print(f"Classified {len(result)} sections: {counts}")
