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

# models used for parallel multi-model extraction
EXTRACTION_MODELS = [
    "gemma4:31b-cloud",
    "deepseek-v4-flash:cloud",
    "gpt-oss:120b-cloud",
]
EXTRACTION_WORKERS = 3
OLLAMA_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "qwen3-vl:235b-instruct-cloud")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "ollama")
MAX_LLM_RETRIES = 3
PIPELINE_VERSION = "1.0"

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
TABLE_COMPLEXITY_THRESHOLD = 0.20
VISION_PARALLELISM = int(os.environ.get("VISION_PARALLELISM", "4"))

# expand short keys with domain vocabulary before embedding
EXPAND_KEYS = True

SECTION_CLASSIFIER_DIR = MODELS_DIR / "section_classifier"
