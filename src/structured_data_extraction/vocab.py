import argparse
import json
import logging
from pathlib import Path
import yake
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_vocab(md_path):
    md_path = Path(md_path)
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown not found: {md_path}")

    out_path = Path(config.INTERMEDIATE_DIR) / f"{md_path.stem}_vocab.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info(f"Skipping vocab — already exists at {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    logger.info(f"Extracting vocabulary from {md_path.name}...")
    text = md_path.read_text(encoding="utf-8")

    kw = yake.KeywordExtractor(
        lan="en", n=config.VOCAB_MAX_NGRAM,
        dedupLim=0.7, top=config.VOCAB_TOP_N,
    )
    keywords = kw.extract_keywords(text)

    # lower score = more important in YAKE
    vocab = [{"term": term, "score": round(score, 4)} for term, score in keywords]

    out_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    logger.info(f"Extracted {len(vocab)} keywords → {out_path}")
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract domain vocabulary")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    result = extract_vocab(args.input)
    print(f"Extracted {len(result)} keywords")
