## In progress hasn't finished yet 

import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from . import config
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

dir = Path(config.OUTPUT_DIR)

extractions = {}
for f in sorted(dir.glob("*.json")):
    with open(f) as fh:
        extractions[f.stem] = json.load(fh)

print(f"loaded {len(extractions)} files")

strata = {}
for ex in extractions.values():
    file = ex["provenance"]["source_file"]
    model = ex["provenance"]["models"]["extraction"]
    for key in ex["provenance"]["schema_fields"]:
        by_model = strata.setdefault((file, key), {})
        rows = by_model.setdefault(model, [])
        for row in ex[key]:
            record = {}
            for k in row:
                if k != "metadata":
                    record[k] = row[k]
            rows.append(record)

rater1 = config.EXTRACTION_MODELS[0]
rater2 = config.EXTRACTION_MODELS[1]
rater3 = config.EXTRACTION_MODELS[2]

raters = [rater1, rater2, rater3]

print(f"Raters: {rater1}, {rater2}, {rater3}")


counts = {}
for by_model in strata.values():
    for model, rows in by_model.items():
            counts.setdefault(model, 0)
            counts[model] += len(rows)
    for model, count in counts.items():
        print(f"{model}: {count} records")


records_by_model = {}
for (file, key), by_model in strata.items():
    for model, rows in by_model.items():
        flat = records_by_model.setdefault(model, [])
        for row in rows:
            flat.append({"file": file, "key": key, "record": row})


class Records:
    def __init__(self, by_model):
        self.by_model = by_model

    def __getitem__(self, model_and_index):
        model, idx = model_and_index
        rows = self.by_model.get(model)
        if not rows:
            raise KeyError(f"no records for model {model}")
        return rows[idx]["record"]

    def count(self, model):
        return len(self.by_model.get(model, []))

record = Records(records_by_model)


def match_records(raters):
    matches = []
