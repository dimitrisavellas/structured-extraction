import json
import re
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
from . import config
from .matching import same_fact, _empty, _as_number, _norm_text, _norm_unit, _META_FIELDS


def _model_slug(model_name):
    return re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def _filled_fields(rec):
    return sum(1 for k, v in rec.items() if k not in _META_FIELDS and not _empty(v))


def fuse(records_by_model):
    pool = [(m, r) for m, recs in records_by_model.items() for r in recs]
    used = set()
    clusters = []
    for i, (model, rec) in enumerate(pool):
        if i in used:
            continue
        used.add(i)
        cluster = {model: rec}
        for j in range(i + 1, len(pool)):
            mj, rj = pool[j]
            if j in used or mj in cluster:
                continue
            if same_fact(rec, rj):
                used.add(j)
                cluster[mj] = rj
        clusters.append(cluster)
    return clusters


_SEC_PRI = {"table": 0, "text": 1, "list": 2}


def _best_section(records: list[dict]) -> str:
    """
    Return the highest-priority section_type found across the records.
    Priority is table > text > list; an unknown/missing type loses to all of them.

    Arguments
    records: A list of records, each a dict that may hold a "metadata" dict
        with a "section_type" key.

    Returns
    The winning section_type as a string ("table", "text", "list", or "unknown").
    """
    types = [r.get("metadata", {}).get("section_type", "unknown") for r in records]
    return min(types, key=lambda t: _SEC_PRI.get(t, 99))


def _fused_metadata(records: list[dict]) -> dict:
    """
    Build the metadata for a fused record.

    Arguments
    records: The records in one cluster, each a dict that may hold a "metadata" dict.

    Returns
    A metadata dict with "section_type" and "specified" (defaulting to True).
    """
    rep = max(records, key=_filled_fields)
    return {"section_type": _best_section(records),
            "specified": rep.get("metadata", {}).get("specified", True)}


def consensus(cluster, n_models):
    recs = list(cluster.values())
    rep = max(recs, key=_filled_fields)
    fields = {k: v for k, v in rep.items() if k != "metadata"}
    return {"confidence": round(len(cluster) / n_models, 2), "models": sorted(cluster),
            **fields, "metadata": _fused_metadata(recs)}


def _fields_match(f, va, vb):
    """Return True if va and vb are the same value for field f (strict equality, no fuzzing)."""
    na, nb = _as_number(va), _as_number(vb)
    if na is not None and nb is not None:
        return na == nb
    if f == "unit" or f.endswith("_unit"):
        return _norm_unit(va) == _norm_unit(vb)
    return _norm_text(va) == _norm_text(vb)


def _strict_eq(a, b):
    for f in (set(a) | set(b)) - _META_FIELDS:
        va, vb = a.get(f), b.get(f)
        if _empty(va) or _empty(vb):
            continue
        if not _fields_match(f, va, vb):
            return False
    return True


def _merge_into(existing, incoming, n_models):
    models = sorted(set(existing["models"]) | set(incoming["models"]))
    rep = max((existing, incoming), key=_filled_fields)
    return {**{k: v for k, v in rep.items() if k not in _META_FIELDS},
            "metadata": {**rep.get("metadata", {}),
                         "section_type": _best_section([existing, incoming])},
            "confidence": round(len(models) / n_models, 2),
            "models": models}


def dedup(facts, n_models):
    merged = []
    for fact in facts:
        match_idx = next((i for i, kept in enumerate(merged) if _strict_eq(fact, kept)), None)
        if match_idx is not None:
            merged[match_idx] = _merge_into(merged[match_idx], fact, n_models)
        else:
            merged.append(fact)
    return merged


def load_outputs():
    by_file = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    slug_to_model = {_model_slug(m): m for m in config.EXTRACTION_MODELS}
    for path in sorted(Path(config.OUTPUT_DIR).glob("*_extraction_*.json")):
        stem, _, slug = path.stem.rpartition("_extraction_")
        model = slug_to_model.get(slug)
        if model is None:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))

        for field, records in data.items():
            if isinstance(records, list):
                by_file[stem][field][model].extend(records)
    return by_file


def fuse_outputs(by_file, models):
    n = len(models)
    facts = []
    for stem, fields in sorted(by_file.items()):
        for field, recs_by_model in sorted(fields.items()):
            clusters = fuse({m: recs_by_model.get(m, []) for m in models})
            for f in dedup([consensus(c, n) for c in clusters], n):
                facts.append({"source_file": f"{stem}.pdf", "field": field, **f})
    return facts


def _group_by_doc(facts):
    grouped = defaultdict(lambda: defaultdict(list))
    for f in facts:
        stem = Path(f["source_file"]).stem
        grouped[stem][f["field"]].append({k: v for k, v in f.items()
                                          if k not in ("source_file", "field")})
    return {stem: dict(fields) for stem, fields in grouped.items()}


def majority_cutoff(n):
    return n // 2 + 1


def rate(by_file, models=None):
    models = config.EXTRACTION_MODELS if models is None else models
    n = len(models)
    Path(config.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    all_facts = fuse_outputs(by_file, models)

    # fused.json is for full view 
    fused_path = Path(config.OUTPUT_DIR) / "fused.json"
    fused_path.write_text(json.dumps(_group_by_doc(all_facts), indent=2, ensure_ascii=False),
                          encoding="utf-8")

    cutoff = majority_cutoff(n)
    user_facts = [f for f in all_facts if len(f["models"]) >= cutoff]
    user_path = Path(config.OUTPUT_DIR) / "user_report.json"
    user_path.write_text(json.dumps(_group_by_doc(user_facts), indent=2, ensure_ascii=False),
                         encoding="utf-8")

    rows = agreement(all_facts, models)
    report = {
        "models": models,
        "facts": len(all_facts),
        "full_agreement": sum(1 for f in all_facts if len(f["models"]) == n),
        "agreement": rows,
        "partial_agreement": [f for f in all_facts if len(f["models"]) < n],
    }
    out = Path(config.OUTPUT_DIR) / "rating_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_agreement(len(all_facts), rows)
    print(f"\nwrote {out}\nwrote {fused_path}\nwrote {user_path}")


def agreement(facts, models):
    if not facts:
        return []
    present = {m: [1 if m in f["models"] else 0 for f in facts] for m in models}
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            agree = sum(x == y for x, y in zip(present[a], present[b])) / len(facts)
            if len(set(present[a] + present[b])) < 2:
                # cohen_kappa_score requires at least two label classes; skip if all labels are identical
                kappa = None
            else:
                kappa = round(cohen_kappa_score(present[a], present[b], labels=[0, 1]), 3)
            rows.append({"raters": [a, b], "kappa": kappa, "agreement": round(agree, 4)})
    return rows


def _print_agreement(n_facts, rows):
    print(f"\ninter-rater agreement across {n_facts} facts (all files):")
    for row in rows:
        a, b = row["raters"]
        kappa = "n/a" if row["kappa"] is None else row["kappa"]
        print(f"  {a} <-> {b}: kappa={kappa}, agreement={row['agreement']:.1%}")


def main():
    by_file = load_outputs()
    if not by_file:
        print(f"No extraction outputs found in {config.OUTPUT_DIR}")
        return
    rate(by_file)


if __name__ == "__main__":
    main()
