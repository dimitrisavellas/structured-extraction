"""Domain vocabulary extraction via YAKE.

Runs unsupervised keyword extraction on the document's Markdown text to surface
domain-specific terms (e.g. "supply voltage", "output impedance"). The resulting
vocab list is used in embed.py to expand user keys with related terms before
embedding, improving recall when the user's key wording doesn't exactly match
the document's phrasing.

YAKE was chosen because it is unsupervised (no training corpus needed), fast,
and works well on technical English. The top-N and max n-gram size are
configurable via VOCAB_TOP_N and VOCAB_MAX_NGRAM in .env.
"""
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
