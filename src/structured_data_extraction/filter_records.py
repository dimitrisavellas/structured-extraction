"""Filter extracted records by consensus confidence tier.

After multi-model extraction and rating, each record carries a confidence score
(fraction of models that agreed on it). This module lets the user keep only the
tiers they trust — e.g. only records all models agreed on (1.0), or everything
above majority (>= 0.67).

Usable both as a CLI tool (structured-filter-records) and programmatically from
the pipeline's end-of-run step. When no --levels are given, an interactive menu
lets the user pick tiers; with a single extraction model every record is
confidence 1.0, so the filter is a no-op.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from . import config

RECORDS = Path(config.OUTPUT_DIR) / "records.json"
_META = {"type", "generated_at", "provenance"}  # envelope keys that aren't extracted fields


def load(path=RECORDS):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records(data):
    """Yield every extracted record across all documents (skipping envelope meta)."""
    for env in data.values():
        for key, recs in env.items():
            if key not in _META and isinstance(recs, list):
                yield from recs


def counts(data):
    """{confidence tier: record count}, highest tier first."""
    tally = Counter(
        round(r["confidence"], 2) for r in _records(data) if r.get("confidence") is not None
    )
    return dict(sorted(tally.items(), reverse=True))


def tier_summary(data):
    """One line per tier: the cumulative '>= tier' count (and its share of the total), plus the
    count exactly at the tier. Returned as lines so a caller can print (CLI) or log (pipeline)."""
    tiers = counts(data)
    total = sum(tiers.values()) or 1
    lines = []
    for tier, at in tiers.items():
        kept = sum(n for t, n in tiers.items() if t >= tier)
        lines.append(f"  >= {tier:<4}  {kept:>4} kept ({kept / total:.0%})   {at:>4} at this tier")
    return lines


def keep(data, tiers):
    """Return a subset with the same envelope shape, only records whose confidence is in `tiers`.
    A field left with no records is dropped; a document left with no fields is omitted."""
    chosen = {round(t, 2) for t in tiers}
    out = {}
    for stem, env in data.items():
        kept = {}
        for key, val in env.items():
            if key in _META:
                kept[key] = val
            elif isinstance(val, list):
                recs = [
                    r
                    for r in val
                    if r.get("confidence") is not None and round(r["confidence"], 2) in chosen
                ]
                if recs:
                    kept[key] = recs
        if any(k not in _META for k in kept):
            out[stem] = kept
    return out


def _pick(tiers):
    """Interactive menu: choose which confidence tiers to keep."""
    for i, tier in enumerate(tiers, 1):
        print(f"  {i}. {tier}")
    print("  a. all")
    try:
        raw = input("Select tiers (e.g. '1,3' or 'a'): ").strip().lower()
    except EOFError:
        return tiers  # no one on stdin (piped/non-interactive) → keep all rather than crash
    if raw in ("", "a", "all"):
        return tiers
    return [
        tiers[int(x) - 1]
        for x in raw.replace(" ", "").split(",")
        if x.isdigit() and 1 <= int(x) <= len(tiers)
    ] or tiers


def run(path=RECORDS, levels=None, json_out=None):
    """Filter the records at `path` by confidence and print (or save) the subset.

    Shared by the CLI and the pipeline's end-of-run step.
    levels: tiers to keep (e.g. [1.0, 0.75]) or ["all"]; None => interactive menu.
    """
    data = load(path)
    tiers = list(counts(data))  # highest first
    if not tiers:
        print(f"No scored records in {path}")
        return
    print("Confidence tiers (of all records):")
    print("\n".join(tier_summary(data)))

    if len(config.EXTRACTION_MODELS) < 2:
        print("Only one extraction model — every record is confidence 1.0, nothing to filter.")
        chosen = tiers
    elif levels:
        chosen = tiers if "all" in levels else [float(x) for x in levels]
    else:
        chosen = _pick(tiers)

    out = keep(data, chosen)
    payload = json.dumps(out, indent=2, ensure_ascii=False)
    n = sum(1 for _ in _records(out))
    if json_out:
        Path(json_out).write_text(payload, encoding="utf-8")
        print(f"wrote {n} record(s) across {len(out)} document(s) to {json_out}")
    else:
        print(payload)
        print(f"\n{n} record(s) across {len(out)} document(s)")


def main():
    ap = argparse.ArgumentParser(description="Filter records by consensus confidence")
    ap.add_argument(
        "--levels",
        nargs="+",
        metavar="LVL",
        help="tiers to keep, e.g. 1.0 0.75 (or 'all'); omit for an interactive menu",
    )
    ap.add_argument("--file", default=str(RECORDS), help="records.json to read")
    ap.add_argument("--json", help="write the kept records here instead of printing")
    args = ap.parse_args()
    run(args.file, args.levels, args.json)


if __name__ == "__main__":
    main()
