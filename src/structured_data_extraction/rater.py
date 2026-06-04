## work in progress, not yet ready for use
import json
from collections import Counter, defaultdict
from pathlib import Path
import Levenshtein
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
        self.value = {}
        self.unit = {}
        self.text = {}
        for key in sorted(data):
            val = data[key]
            if key == "metadata":
                continue
            elif key == "unit" or key.endswith("_unit"):
                self.unit[key] = str(val).strip()
            elif key == "value" or isinstance(val, (int, float)):
                self.value[key] = to_number(val)
            else:
                self.text[key] = str(val)

    @property
    def descriptor(self):

        return " ".join(self.text[k] for k in sorted(self.text) if self.text[k])

    def matches(self, other):

        if self.value != other.value or self.unit != other.unit:
            return False

        for key in self.text:
            if key in other.text and self.text[key] and other.text[key]:
                if Levenshtein.ratio(self.text[key], other.text[key], processor=normalize) < THRESHOLD:
                    return False
        return True


class Records:
    def __init__(self):
        self.by_model = defaultdict(list)

    def add(self, model, data):
        self.by_model[model].append(Record(model, data))

    def count(self, model):
        return len(self.by_model[model])

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
    files = sorted(Path(config.OUTPUT_DIR).glob("*.json"))
    for f in files:
        ex = json.loads(f.read_text(encoding="utf-8"))
        file = ex["provenance"]["source_file"]
        model = ex["provenance"]["models"]["extraction"]
        for key in ex["provenance"]["schema_fields"]:
            if key not in ex:
                print(f"  warning: {f.name}: schema field {key!r} missing from output, skipping")
                continue
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
            agree = sum(x == y for x, y in zip(present[a], present[b])) / len(all_facts)
            if len(set(present[a] + present[b])) < 2:
                # both raters gave the same answer to every fact -> kappa is undefined
                kappa = "n/a (no disagreement)"
            else:
                k = cohen_kappa_score(present[a], present[b], labels=[0, 1])
                kappa = f"{k:.3f}"
            print(f"  {a} ↔ {b}: kappa={kappa}, agreement={agree:.1%}")


if __name__ == "__main__":
    main()
