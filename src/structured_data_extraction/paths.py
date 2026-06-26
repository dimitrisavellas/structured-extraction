import json
from pathlib import Path
from . import config


def inter(stem, name):
    return Path(config.INTERMEDIATE_DIR) / f"{stem}_{name}"


def out(stem, name):
    return Path(config.OUTPUT_DIR) / f"{stem}_{name}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return obj


def cached(path, build):
    """Return the JSON at `path` if it exists, else `build()` then write + return it."""
    path = Path(path)
    if path.exists():
        return read_json(path)
    return write_json(path, build())
