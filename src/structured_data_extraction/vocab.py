import argparse
import logging
from pathlib import Path
import yake
from . import config
from .paths import inter, cached

logger = logging.getLogger(__name__)


def extract_vocab(md_path):
    md_path = Path(md_path)

    def build():
        logger.info(f"Extracting vocabulary from {md_path.name}...")
        kw = yake.KeywordExtractor(lan="en", n=config.VOCAB_MAX_NGRAM,
                                   dedupLim=0.7, top=config.VOCAB_TOP_N)
        # lower score = more important in YAKE
        return [{"term": term, "score": round(score, 4)}
                for term, score in kw.extract_keywords(md_path.read_text(encoding="utf-8"))]

    return cached(inter(md_path.stem, "vocab.json"), build)


def main():
    parser = argparse.ArgumentParser(description="Extract domain vocabulary")
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    result = extract_vocab(args.input)
    print(f"Extracted {len(result)} keywords")


if __name__ == "__main__":
    main()
