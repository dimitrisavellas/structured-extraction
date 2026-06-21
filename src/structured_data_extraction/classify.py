import argparse
import json
import logging
from collections import Counter
from pathlib import Path
import joblib
from . import config
from .features import build_features, context_rows

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SectionClassifier:

    def __init__(self, model_dir=None):
        d = Path(model_dir) if model_dir else Path(config.SECTION_CLASSIFIER_DIR)
        self.rf = joblib.load(d / "rf_classifier.joblib")
        self.word_vec = joblib.load(d / "tfidf_vectorizer.joblib")
        self.char_vec = joblib.load(d / "tfidf_char_vectorizer.joblib")
        self.dl_cats = joblib.load(d / "docling_cats.joblib")
        self.dl_idx = {c: i for i, c in enumerate(self.dl_cats)}

    def classify(self, chunks):
        if not chunks:
            return []
        ordered, rows = context_rows(chunks, text_key="text")
        X = build_features(rows, self.dl_idx, self.word_vec, self.char_vec, fit=False)
        preds = self.rf.predict(X)
        return [dict(c, section_type=str(p)) for c, p in zip(ordered, preds)]


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
    text_chunks = [
        {
            "section_id": i,
            "docling_label": item.get("label", "text"),
            "text": text,
            "page": item.get("page"),
            "bbox": item.get("bbox"),
        }
        for i, item in enumerate(items)
        if (text := (item.get("text") or "").strip())
    ]

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
    counts = Counter(s["type"] for s in result)
    print(f"Classified {len(result)} sections: {dict(counts)}")
