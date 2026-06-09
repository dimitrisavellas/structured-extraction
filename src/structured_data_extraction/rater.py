## work in progress, not yet ready for use
import json
from collections import defaultdict
from pathlib import Path
from rapidfuzz import fuzz
from sklearn.metrics import cohen_kappa_score
from . import config

THRESHOLD = 0.85


def normalize(s):
    return str(s).strip().casefold()


def to_number(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return normalize(val)


class Record:
    def __init__(self, model, data):
        self.model = model
        self.exact = {}  
        self.text = {}   
        for key in sorted(data):
            val = data[key]
            if key == "metadata":
                continue
            elif key == "unit" or key.endswith("_unit"):
                self.exact[key] = str(val).strip()
            elif key == "value" or isinstance(val, (int, float)):
                self.exact[key] = to_number(val)
            else:
                self.text[key] = str(val)

    @property
    def fields(self):
        # the original schema fields of this record, sorted
        merged = {**self.exact, **self.text}
        return {k: merged[k] for k in sorted(merged)}

    def matches(self, other):
        if self.exact != other.exact:
            return False
        for key, val in self.text.items():
            ov = other.text.get(key, "")

            if val and ov and fuzz.ratio(val, ov, processor=normalize) < THRESHOLD * 100:
                return False
        return True


class Records:
    def __init__(self):
        self.by_model = defaultdict(list)

    def add(self, model, data):
        self.by_model[model].append(Record(model, data))

    def cluster(self):
        pool = [r for records in self.by_model.values() for r in records]
        used = set()
        facts = []
        for seed in pool:
            if id(seed) in used:
                continue
            used.add(id(seed))
            fact = {seed.model: seed}
            for other in pool:
                if id(other) in used or other.model in fact:
                    continue
                if seed.matches(other):
                    used.add(id(other))
                    fact[other.model] = other
            facts.append(fact)
        return facts


def load():
    strata = defaultdict(Records)
    loaded = 0
    for f in sorted(Path(config.OUTPUT_DIR).glob("*.json")):
        ex = json.loads(f.read_text(encoding="utf-8"))
        if ex.get("type") != "ExtractionResult":
            continue 
        loaded += 1
        file = ex["provenance"]["source_file"]
        model = ex["provenance"]["models"]["extraction"]
        for key in ex["provenance"]["schema_fields"]:
            if key not in ex:
                print(f"  warning: {f.name}: schema field {key!r} missing from output, skipping")
                continue
            for row in ex[key]:
                strata[(file, key)].add(model, row)
    print(f"loaded {loaded} files")
    return strata


def fact_entry(fact, n):
    r = next(iter(fact.values()))
    return {"confidence": round(len(fact) / n, 2), "models": sorted(fact), **r.fields}


def agreement(all_facts, raters):
    # pairwise Cohen's kappa over presence/absence of every fact
    present = {m: [1 if m in fact else 0 for fact in all_facts] for m in raters}
    rows = []
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            agree = sum(x == y for x, y in zip(present[a], present[b])) / len(all_facts)
            if len(set(present[a] + present[b])) < 2:
                # both raters gave the same answer to every fact -> kappa is undefined
                kappa = "n/a (no disagreement)"
            else:
                kappa = round(cohen_kappa_score(present[a], present[b], labels=[0, 1]), 3)
            rows.append({"raters": [a, b], "kappa": kappa, "agreement": round(agree, 4)})
    return rows


def build_report(file, file_strata, raters):
    fields = {}
    all_facts = []
    for key, records in sorted(file_strata.items()):
        n = len(records.by_model)
        facts = records.cluster()
        all_facts += facts
        full = sum(len(fact) == n for fact in facts)
        entries = [fact_entry(fact, n) for fact in facts]
        fields[key] = {"facts": len(facts), "full_agreement": full, "facts_list": entries}

    report = {"source_file": file, "raters": raters, "fields": fields}
    if all_facts:
        report["agreement"] = agreement(all_facts, raters)
        print(f"\n[{file}] inter-rater agreement over {len(all_facts)} facts:")
        for row in report["agreement"]:
            a, b = row["raters"]
            print(f"  {a} ↔ {b}: kappa={row['kappa']}, agreement={row['agreement']:.1%}")
    return report


def main():
    strata = load()
    if not strata:
        print("No extraction outputs found.")
        return

    raters = config.EXTRACTION_MODELS[:3]

    by_file = defaultdict(dict)
    for (file, key), records in strata.items():
        by_file[file][key] = records

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for file, file_strata in sorted(by_file.items()):
        report = build_report(file, file_strata, raters)
        out = config.OUTPUT_DIR / f"{Path(file).stem}_rating.json"
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
