import json
import os
from pathlib import Path
from . import config
from .paths import inter, read_json

USER_REPORT = Path(config.OUTPUT_DIR) / "user_report.json"
OUT_PATH = Path(config.OUTPUT_DIR) / "fdo_records.json"


def _run(stem):
    """Load the run stamp extract.py wrote (extracted_at, user_keys, models) for the envelope's
    provenance. Returns {} when it's missing so fdo_envelope falls back to its defaults."""
    path = inter(stem, "run.json")
    return read_json(path) if path.exists() else {}


def _document_type(stem):
    path = inter(stem, "schema.json")
    return read_json(path).get("document_type", "unknown") if path.exists() else "unknown"


def fdo_envelope(stem, by_key):
    run = _run(stem)
    return {
        "type": "ExtractionResult",
        "generated_at": run.get("extracted_at"),
        "provenance": {
            "source_file": f"{stem}.pdf",
            "document_type": _document_type(stem),
            "models": {
                "schema_discovery": os.environ["SCHEMA_MODEL"],
                "extraction": run.get("models") or config.EXTRACTION_MODELS,
                "vision": os.environ["VISION_MODEL"],
                "embedding": os.environ["EMBEDDING_MODEL"],
            },
            "user_keys": run.get("user_keys", []),
        },
        **by_key,  
    }


def build_records(path=USER_REPORT):
    report = read_json(path)
    return {stem: fdo_envelope(stem, by_key) for stem, by_key in report.items()}


def write_records(output_dir):
    """Build FDO envelopes from <output_dir>/user_report.json and write fdo_records.json.

    Takes output_dir explicitly so the pipeline can honour a --output override
    (config.OUTPUT_DIR may be mutated at runtime, after this module is imported).
    """
    output_dir = Path(output_dir)
    envelopes = build_records(output_dir / "user_report.json")
    out_path = output_dir / "fdo_records.json"
    out_path.write_text(json.dumps(envelopes, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path, len(envelopes)


if __name__ == "__main__":
    out_path, n = write_records(config.OUTPUT_DIR)
    print(f"wrote {out_path} ({n} documents)")
