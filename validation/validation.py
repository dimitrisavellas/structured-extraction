import argparse
import json
import logging
from pathlib import Path

from structured_data_extraction import config, matching, rater
from structured_data_extraction.main import process_single_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("validation")

GT_PATH = Path(__file__).resolve().parent / "validation.json"
VALIDATION_DIR = Path(__file__).resolve().parent / "validation-set"

KEYS = {
    "0150200051": ["product family","cable length", "product specification", "Voltage", "net weight"],
    "2D iPSC p7 d0 222d119683938118b7f7d1f17fc21466": ["DMEM","mTeSR", "StemMACS"],
    "2D iPSC p7 d3 p8 d0 222d11968393813abd5de2b3a86d09c2": ["protocol","seeded"],
    "AD8-0000-E01": ["logic voltage", "supply current","humidity limits"],
    "AD840": ["gain bandwidth","quiescent current","input voltage noise"],
    "ada4523-1": ["internal chopping frequency", "EMF", "Output Voltage Swing Low"],
    "ALD1722_E": ["rise time", "overshoot factor", "Average Long term input offset"],
    "ALD2711": ["rise time", "PSRR", "slew rate"],
    "BC546": ["collector cutoff current","DC current gain","Thermal Resistance"],
    "Diff d-0 222d11968393818e8b47f1429be49193": ["CHIR","serological pipette"],
    "Diff d12 222d11968393810ea8b5d3a5bbfb24e1": ["lactate stock","serological pipette"],
    "Diff d1 222d1196839381baa62bea713941d2f4": ["albumin","falcon"],
    "Diff d-1 222d1196839381faaefdca27cb30255b": ["aliquot volume"],
    "Diff d14 222d119683938174ad21db3320bcaaa7": ["lactate selection medium", "comment"],
    "Diff d20 222d1196839381438012c3c03bc4af26": ["cardiac culture medium", "serological pipette"],
    "Diff d21 222d11968393810fa4a3ec2fecb80b09": ["PSC", "DNase", "Centrifuge","Cardiac Culture", "RPM","counts","BCT","purity"],
    "Diff d2 222d1196839381169b8dd844acd1f341": ["serological pipette"],
    "Diff d-2 222d11968393816ab739f96052bc35fa": ["ROCK inhibitor", "Incubated","detach", "RT PBS","mio per plate","aggregate cells"],
    "Diff d5 222d11968393812c9f44f4153ae70998": ["albumin","date"],
    "Diff d7 222d119683938193b326d3e17d4031f4": ["CM", "day 7"],
    "Diff d9 222d119683938107bbb6ea581e9f1820": ["freezer"],
    "eMT d1 2D CMs d0 222d11968393816eb11cc3fc5d43b60f": ["BCT", "DMEM","fibronectin"],
    "eMT d27 222d11968393810dba38caf880faf6df": ["16.1", "16.2"],
    "eMT d3 2D CMs d2 222d1196839381cf9bbbe4db5c387a95": ["eMT16.1","eMT16.2", "eMT day", "2D layer"],
    "eMT d63 222d11968393813db7bdede059f4319a": ["Date", "EMT medium"],
    "lf356": ["Input offset voltage", "Still Air", "Input Capacitance"],
    "LM321-D": ["Output Voltage","Phase Margin"],
    "lmv358a": ["CMRR", "Voltage Output Swing"],
    "rhf310a": ["Heavy Ions", "TID", "Maximum Power Dissipation"],
    "TL431": ["Min Cathode Current", "a1", "F"],
    "TLV27x": ["Input Bias Current","VOH","Power Supply Rejection Ratio"],
    "tsb951": ["Junction Temperature", "ESD", "SR", "GBP"],
}


def run_extraction(stems=None):
    pending = {s: keys for s, keys in KEYS.items() if keys and (not stems or s in stems)}
    if not pending:
        logger.error("Nothing to run — check KEYS / --stems.")
        return

    logger.info(f"Running extraction for {len(pending)} file(s)")
    done, failed = [], []
    for stem, keys in pending.items():
        pdf_path = VALIDATION_DIR / f"{stem}.pdf"
        if not pdf_path.exists():
            logger.error(f"Missing PDF, skipping: {pdf_path.name}")
            failed.append(stem)
            continue
        try:
            result = process_single_pdf(str(pdf_path), keys)
            (done if result else failed).append(stem)
        except Exception as e:
            logger.error(f"Failed {stem}: {e}")
            failed.append(stem)

    logger.info(f"Done: {len(done)} succeeded, {len(failed)} failed/skipped")
    if failed:
        logger.info(f"Not extracted: {failed}")


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



USER_REPORT = "user_report"


def evaluate(by_file, user_report, gt, models, only=None):
    variants = list(models) + [USER_REPORT]
    totals = {v: [0, 0, 0] for v in variants}

    # only score docs  actually extracted
    stems = sorted(set(gt) & set(by_file))
    if only:
        stems = [s for s in stems if s in set(only)]
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
    parser = argparse.ArgumentParser(description="Run + score the validation set")
    parser.add_argument("--no-run", action="store_true", help="score existing output/ only")
    parser.add_argument("--stems", nargs="+", help="limit run + scoring to these stems")
    args = parser.parse_args()

    if not args.no_run:
        run_extraction(args.stems)

    by_file = rater.load_outputs()
    if not by_file:
        print(f"No extraction outputs in {config.OUTPUT_DIR} — run without --no-run first.")
        return

    rater.rate(by_file) # writes the user report

    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    user_report = json.loads((Path(config.OUTPUT_DIR) / "user_report.json").read_text(encoding="utf-8"))
    report = evaluate(by_file, user_report, gt, config.EXTRACTION_MODELS, only=args.stems)
    print_report(report)
    out = Path(config.OUTPUT_DIR) / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
