"""Does smarter fusion earn F1 over majority-vote / best single model?

Runs ONLY on the per-model records already in output/ (no extraction). For each
fused fact we know its vote-pattern (which subset of the 3 models produced it).
A "keep-rule" is just a set of vote-patterns to report. We:
  - score the fixed rules (gemma-alone, majority>=2, union),
  - find the single best keep-set on the FULL set (oracle = overfit ceiling),
  - and learn the keep-set honestly with leave-one-doc-out CV.

If LOO F1 doesn't beat max(gemma, majority), the ensemble still isn't earning it.
"""

import json
from collections import Counter, defaultdict
from itertools import combinations

from structured_data_extraction import config, rater
import validation as V

GEMMA = "gemma4:31b-cloud"


def build_facts():
    gt = json.loads(V.GT_PATH.read_text(encoding="utf-8"))
    by_file = rater.load_outputs()
    models = config.EXTRACTION_MODELS
    n = len(models)
    stems = sorted(set(gt) & set(by_file))

    # facts[stem][field] = list of (vote_pattern, representative_record)
    facts = defaultdict(lambda: defaultdict(list))
    for stem in stems:
        for field, recs_by_model in by_file[stem].items():
            clusters = rater.fuse({m: recs_by_model.get(m, []) for m in models})
            for f in rater.dedup([rater.consensus(c, n) for c in clusters], n):
                pattern = tuple(f["models"])
                rep = {k: v for k, v in f.items() if k not in ("confidence", "models")}
                facts[stem][field].append((pattern, rep))
    return gt, by_file, facts, stems


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    gt, by_file, facts, stems = build_facts()
    patterns = sorted({p for s in stems for fl in facts[s].values() for p, _ in fl},
                      key=lambda p: (len(p), p))
    keepsets = [frozenset(c) for r in range(1, len(patterns) + 1) for c in combinations(patterns, r)]

    # per-doc (tp,fp,fn) for every keep-set, computed once
    def doc_score(stem, S):
        tp = fp = fn = 0
        for key, gt_recs in gt[stem].items():
            field = V.field_name(key)
            preds = [rep for pat, rep in facts[stem].get(field, []) if pat in S]
            a, b, c = V.score(preds, gt_recs)
            tp += a; fp += b; fn += c
        return tp, fp, fn

    table = {s: {S: doc_score(s, S) for S in keepsets} for s in stems}

    def total(stem_list, S):
        t = [0, 0, 0]
        for s in stem_list:
            for i, x in enumerate(table[s][S]):
                t[i] += x
        return tuple(t)

    # gemma-alone baseline: raw gemma records, not fused
    g = [0, 0, 0]
    for s in stems:
        for key, gt_recs in gt[s].items():
            preds = by_file[s].get(V.field_name(key), {}).get(GEMMA, [])
            for i, x in enumerate(V.score(preds, gt_recs)):
                g[i] += x

    majority = frozenset(p for p in patterns if len(p) >= 2)
    union = frozenset(patterns)
    oracle = max(keepsets, key=lambda S: f1(*total(stems, S)))

    # leave-one-doc-out learned keep-set
    loo = [0, 0, 0]
    chosen = Counter()
    for held in stems:
        train = [s for s in stems if s != held]
        best = max(keepsets, key=lambda S: f1(*total(train, S)))
        chosen[best] += 1
        for i, x in enumerate(table[held][best]):
            loo[i] += x

    rows = [
        ("gemma alone", tuple(g)),
        ("majority >=2 (current)", total(stems, majority)),
        ("union (all patterns)", total(stems, union)),
        (f"oracle keep-set [overfit]", total(stems, oracle)),
        ("LOO-learned keep-set", tuple(loo)),
    ]
    print(f"\n{len(stems)} docs | patterns seen: {[ '+'.join(_short(p)) for p in patterns ]}\n")
    print(f"{'strategy':<28}{'tp':>5}{'fp':>5}{'fn':>5}{'recall':>9}{'prec':>8}{'f1':>8}")
    print("-" * 71)
    for name, (tp, fp, fn) in rows:
        m = V.prf(tp, fp, fn)
        print(f"{name:<28}{tp:>5}{fp:>5}{fn:>5}{m['recall']:>9.3f}{m['precision']:>8.3f}{m['f1']:>8.3f}")

    print(f"\noracle keep-set: {{{', '.join('+'.join(_short(p)) for p in sorted(oracle))}}}")
    print("\nLOO chose (keep-set: #folds):")
    for S, k in chosen.most_common():
        print(f"  {k:>2}x  {{{', '.join('+'.join(_short(p)) for p in sorted(S))}}}")


def _short(pattern):
    return [m.split(":")[0].split("-")[0][:4] for m in pattern]


if __name__ == "__main__":
    main()
