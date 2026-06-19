import json
from pathlib import Path

from structured_data_extraction import config, matching, rater

GT_PATH = Path(__file__).resolve().parent / "validation.json"


def field_name(key):
    return matching.normalize_key(key)


def _filled(value):
    return value is not None and str(value).strip() != ""


def matches_gt(pred, gt):
    pred_values = [v for k, v in pred.items() if k not in matching._META_FIELDS]
    for field, gval in gt.items():
        if field == "metadata" or not _filled(gval):
            continue
        if not any(matching.compare(field, pv, gval) == matching.MATCH for pv in pred_values):
            return False
    return True


def score(preds, gts):
    candidates = [[j for j, gt in enumerate(gts) if matches_gt(pred, gt)] for pred in preds]
    gt_for = [-1] * len(gts)

    def assign(i, seen):
        for j in candidates[i]:
            if not seen[j]:
                seen[j] = True
                if gt_for[j] == -1 or assign(gt_for[j], seen):
                    gt_for[j] = i
                    return True
        return False

    tp = sum(assign(i, [False] * len(gts)) for i in range(len(preds)))
    return tp, len(preds) - tp, len(gts) - tp


def prf(tp, fp, fn):
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "recall": round(recall, 3), "precision": round(precision, 3),
            "f1": round(f1, 3)}


# the deliverable is user_report.json — score that exact file so the reported quality describes
# what the user actually receives, not an in-memory re-fusion. per-model rows stay as diagnostics.
USER_REPORT = "user_report"


def evaluate(by_file, user_report, gt, models):
    variants = list(models) + [USER_REPORT]
    totals = {v: [0, 0, 0] for v in variants}

    # only score docs we actually extracted, so a partial/smoke run is meaningful
    stems = sorted(set(gt) & set(by_file))
    for stem in stems:
        for key, gt_recs in gt[stem].items():
            field = field_name(key)

            for m in models:
                tp, fp, fn = score(by_file.get(stem, {}).get(field, {}).get(m, []), gt_recs)
                for i, x in enumerate((tp, fp, fn)):
                    totals[m][i] += x

            tp, fp, fn = score(user_report.get(stem, {}).get(field, []), gt_recs)
            for i, x in enumerate((tp, fp, fn)):
                totals[USER_REPORT][i] += x

    return {
        "variants": variants,
        "scored_docs": stems,
        "total_docs": len(gt),
        "overall": {v: prf(*totals[v]) for v in variants},
    }


def print_report(report):
    print(f"\nscored {len(report['scored_docs'])}/{report['total_docs']} docs "
          f"(tp/fp/fn and metrics summed across all scored files)")
    print(f"\n{'variant':<28}{'tp':>5}{'fp':>5}{'fn':>5}"
          f"{'recall':>9}{'precision':>11}{'f1':>7}")
    print("-" * 72)
    for v in report["variants"]:
        m = report["overall"][v]
        print(f"{v:<28}{m['tp']:>5}{m['fp']:>5}{m['fn']:>5}"
              f"{m['recall']:>9.3f}{m['precision']:>11.3f}{m['f1']:>7.3f}")


def main():
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    by_file = rater.load_outputs()
    if not by_file:
        print(f"No extraction outputs in {config.OUTPUT_DIR} — run validation_extract.py first.")
        return
    user_path = Path(config.OUTPUT_DIR) / "user_report.json"
    if not user_path.exists():
        print(f"No {user_path.name} — run `python -m structured_data_extraction.rater` first.")
        return
    user_report = json.loads(user_path.read_text(encoding="utf-8"))
    report = evaluate(by_file, user_report, gt, config.EXTRACTION_MODELS)
    print_report(report)
    out = Path(config.OUTPUT_DIR) / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
