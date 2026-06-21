import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "output" / "user_report.json"
_META = {"confidence", "models", "model"} 


@dataclass
class Record:
    source: str        # document stem
    field: str         # extracted key 
    fields: dict       # schema-driven value, unit, qualifier, ...
    confidence: float  # fusion confidence
    models: list       # provenance


def load_records(path=DEFAULT_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Record(source, field, {k: v for k, v in r.items() if k not in _META},
               r.get("confidence", 1.0), r.get("models", []))
        for source, by_field in data.items()
        for field, recs in by_field.items()
        for r in recs
    ]

records = load_records()
