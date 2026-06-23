import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "output" / "user_report.json"
_META = {"confidence", "models", "model"} 


@dataclass
class Record:
    source: str        # document stem
    key: str           # extracted key 
    attributes: dict   # schema-driven value, unit, qualifier, ...
    confidence: float  # fusion confidence
    models: list       # provenance


def load_records(path=DEFAULT_PATH):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Record(source, key, {k: v for k, v in r.items() if k not in _META},
               r.get("confidence", 1.0), r.get("models", []))
        for source, by_key in data.items()
        for key, recs in by_key.items()
        for r in recs
    ]

records = load_records()
print(records[15])
