"""Global configuration: directory layout, device selection, and tuning knobs.

Directory paths default to subdirectories of the
working directory (pdf/, markdown/, intermediate/, output/).

The embedding device is auto-detected: CUDA if available, then MPS (Apple
Silicon), then CPU. Override with EMBEDDING_DEVICE in .env.

Tuning constants (similarity threshold, search top-K, vision complexity
threshold, etc.) have sensible defaults but can be overridden per-run in .env.
"""
import os
from pathlib import Path
from importlib import resources



CWD = Path.cwd()
PDF_DIR = CWD / "pdf"
MARKDOWN_DIR = CWD / "markdown"
INTERMEDIATE_DIR = CWD / "intermediate"
OUTPUT_DIR = CWD / "output"


SECTION_CLASSIFIER_DIR = resources.files("structured_data_extraction") / "models" / "section_classifier"


def _auto_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or _auto_device()


# The models to run for extraction, one or many 
# The first one is the "primary": used wherever a single model is needed 
EXTRACTION_MODELS = [m.strip() for m in os.environ["EXTRACTION_MODELS"].split(",") if m.strip()]
assert EXTRACTION_MODELS, "EXTRACTION_MODELS in .env must list at least one model"

# When True (set by main.py --force), every cached pipeline step recomputes instead of skipping.
FORCE = False

# Section-type preference when deduping records that describe the same fact: table beats text beats list.
SECTION_TYPE_PRIORITY = {"table": 0, "text": 1, "list": 2}

# Tuning constants — overridable in .env; the values here are the defaults if unset.
def _env_bool(name, default):
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in ("1", "true", "yes", "on")

VOCAB_TOP_N = int(os.environ.get("VOCAB_TOP_N", 50))
VOCAB_MAX_NGRAM = int(os.environ.get("VOCAB_MAX_NGRAM", 4))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.55))
SEARCH_TOP_K = int(os.environ.get("SEARCH_TOP_K", 25))
VISION_COMPLEXITY_THRESHOLD = float(os.environ.get("VISION_COMPLEXITY_THRESHOLD", 0.15))
VISION_PARALLELISM = int(os.environ.get("VISION_PARALLELISM", 4))
EXPAND_KEYS = _env_bool("EXPAND_KEYS", True)
