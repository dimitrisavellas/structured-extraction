## work in progress, not yet ready for use 
import json
from collections import Counter, defaultdict
from pathlib import Path
from Levenshtein import ratio
from sklearn.metrics import cohen_kappa_score
from . import config

THRESHOLD = 0.85  # min descriptor similarity for two records to be the same fact


def to_number(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return str(val).strip().lower()


class Record:
    def __init__(self, model, data):
        self.model = model
        self.descriptor = ""
        self.value = {}
        self.unit = {}
        for key in sorted(data):
            val = data[key]
            if key == "metadata":
                continue
            elif key == "unit" or key.endswith("_unit"):
                self.unit[key] = str(val).strip()/improve
            elif key == "value" or isinstance(val, (int, float)):
                self.value[key] = to_number(val)
            else:
                self.descriptor += " " + str(val)
        self.descriptor = self.descriptor.strip().lower()

    def matches(self, other):
        return (self.value == other.value and self.unit == other.unit
                and ratio(self.descriptor, other.descriptor) >= THRESHOLD)


class Records:
    def __init__(self):
        self.by_model = defaultdict(list)

    def add(self, model, data):
        self.by_model[model].append(Record(model, data))

    def count(self, model):
        return len(self.by_model[model])

    def cluster(self):
        # group matching records across models, one record per model per fact
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
    files = sorted(Path(config.OUTPUT_DIR).glob("*.json"))
    for f in files:
        ex = json.loads(f.read_text(encoding="utf-8"))
        file = ex["provenance"]["source_file"]
        model = ex["provenance"]["models"]["extraction"]
        for key in ex["provenance"]["schema_fields"]:
            for row in ex[key]:
                strata[(file, key)].add(model, row)
    print(f"loaded {len(files)} files")
    return strata


def main():
    strata = load()
    if not strata:
        print("No extraction outputs found.")
        return

    raters = config.EXTRACTION_MODELS[:3]
    print(f"Raters: {', '.join(raters)}")

    counts = Counter()
    for records in strata.values():
        for model in records.by_model:
            counts[model] += records.count(model)
    for model, count in counts.items():
        print(f"{model}: {count} records")

    # confidence per fact = how many models found it
    all_facts = []
    for (file, key), records in sorted(strata.items()):
        n = len(records.by_model)
        facts = records.cluster()
        all_facts += facts
        full = sum(1 for fact in facts if len(fact) == n)
        print(f"\n[{file}] {key}: {len(facts)} facts | full agreement: {full}/{len(facts)}")
        for fact in facts:
            if len(fact) < n:
                r = next(iter(fact.values()))
                models = ", ".join(sorted(fact))
                print(f"    conf={len(fact) / n:.2f}  {r.descriptor!r} {r.value} {r.unit} — {models}")

    # pairwise Cohen's kappa over presence/absence of every fact
    present = {m: [1 if m in fact else 0 for fact in all_facts] for m in raters}
    print(f"\nInter-rater agreement over {len(all_facts)} facts:")
    for i, a in enumerate(raters):
        for b in raters[i + 1:]:
            k = cohen_kappa_score(present[a], present[b])
            agree = sum(x == y for x, y in zip(present[a], present[b])) / len(all_facts)
            print(f"  {a} ↔ {b}: kappa={k:.3f}, agreement={agree:.1%}")


if __name__ == "__main__":
    main()
