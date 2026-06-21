import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PDF_DIR = ROOT / "pdf"
MARKDOWN_DIR = ROOT / "markdown"
INTERMEDIATE_DIR = ROOT / "intermediate"
OUTPUT_DIR = ROOT / "output"
TEMPLATES_DIR = ROOT / "templates"
MODELS_DIR = ROOT / "models"

# LLM models
OLLAMA_SCHEMA_MODEL = os.environ.get("OLLAMA_SCHEMA_MODEL", "glm-5.1:cloud")
OLLAMA_EXTRACT_MODEL = os.environ.get("OLLAMA_EXTRACT_MODEL", "gemma4:31b-cloud")

# models used for parallel extraction
EXTRACTION_MODELS = [
    "gemma4:31b-cloud",
    "deepseek-v4-flash:cloud",
    "glm-4.7:cloud",
]
EXTRACTION_WORKERS = 3
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "kimi-k2.7-code:cloud") # qwen3-vl:235b-instruct-cloud is retired now unfortunately
                                                                                    # will check if qwen3.5:397b-cloud is better than kimi-k2.7-code:cloud when it's out of private beta
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
MAX_LLM_RETRIES = 3
PIPELINE_VERSION = "1.0"
# ponytail: best-effort reproducibility. Remote :cloud models may ignore `seed`;
# real determinism comes from the cached intermediate JSON, not the API.
SEED = 0

# embedding
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")


def _auto_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or _auto_device()

# thresholds 
VOCAB_TOP_N = 50
VOCAB_MAX_NGRAM = 4
SIMILARITY_THRESHOLD = 0.55
SEARCH_TOP_K = 25


VISION_COMPLEXITY_THRESHOLD = 0.15
VISION_PARALLELISM = int(os.environ.get("VISION_PARALLELISM", "4"))

# expand short keys with domain vocabulary before embedding
EXPAND_KEYS = True

SECTION_CLASSIFIER_DIR = MODELS_DIR / "section_classifier"
