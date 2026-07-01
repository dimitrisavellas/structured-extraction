import os
from pathlib import Path
from . import config
from .paths import inter, read_json, write_json

FUSED = Path(config.OUTPUT_DIR) / "fused.json"


def _run(stem):
    """Load the run stamp extract.py wrote (extracted_at, user_keys) for the envelope's
    provenance. Returns {} when it's missing so make_record falls back to its defaults."""
    path = inter(stem, "run.json")
    return read_json(path) if path.exists() else {}


def _document_type(stem):
    path = inter(stem, "schema.json")
    return read_json(path).get("document_type", "unknown") if path.exists() else "unknown"


def make_record(stem, by_key):
    """Wrap one document's fused facts in a metadata envelope: a type tag, a timestamp, and
    provenance (source file, document type, the models used, and the user's keys)."""
    run = _run(stem)
    return {
        "type": "ExtractionResult",
        "generated_at": run.get("extracted_at"),
        "provenance": {
            "source_file": f"{stem}.pdf",
            "document_type": _document_type(stem),
            "models": {
                "schema_discovery": os.environ["SCHEMA_MODEL"],
                "vision": os.environ["VISION_MODEL"],
                "embedding": os.environ["EMBEDDING_MODEL"],
            },
            "user_keys": run.get("user_keys", []),
        },
        **by_key,
    }


def build_records(path=FUSED):
    report = read_json(path)
    return {stem: make_record(stem, by_key) for stem, by_key in report.items()}


def write_records(output_dir):
    """Wrap every document's fused facts in a provenance envelope and write records.json.
    Keeps all records; consumers filter by the `confidence` field (see filter_records.py)."""
    output_dir = Path(output_dir)
    envelopes = build_records(output_dir / "fused.json")
    out_path = output_dir / "records.json"
    write_json(out_path, envelopes)
    return out_path, len(envelopes)


if __name__ == "__main__":
    out_path, n = write_records(config.OUTPUT_DIR)
    print(f"wrote {out_path} ({n} documents)")
