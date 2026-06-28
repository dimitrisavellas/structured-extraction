import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = ROOT / "pdf"
MARKDOWN_DIR = ROOT / "markdown"
INTERMEDIATE_DIR = ROOT / "intermediate"
OUTPUT_DIR = ROOT / "output"
TEMPLATES_DIR = ROOT / "templates"
MODELS_DIR = ROOT / "models"
SECTION_CLASSIFIER_DIR = MODELS_DIR / "section_classifier"


def _auto_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or _auto_device()


# The models to run for extraction — one or many (the pipeline votes across them when >1).
# The first one is the "primary": used wherever a single model is needed (table text-refinement).
EXTRACTION_MODELS = [m.strip() for m in os.environ["EXTRACTION_MODELS"].split(",") if m.strip()]
assert EXTRACTION_MODELS, "EXTRACTION_MODELS in .env must list at least one model"

# When True (set by main.py --force), every cached pipeline step recomputes instead of skipping.
FORCE = False

# Section-type preference when deduping records that describe the same fact: table beats text beats list.
SECTION_TYPE_PRIORITY = {"table": 0, "text": 1, "list": 2}

# Tuning constants
VOCAB_TOP_N = 50
VOCAB_MAX_NGRAM = 4
SIMILARITY_THRESHOLD = 0.55
SEARCH_TOP_K = 25
VISION_COMPLEXITY_THRESHOLD = 0.15
VISION_PARALLELISM = 4
EXPAND_KEYS = True
