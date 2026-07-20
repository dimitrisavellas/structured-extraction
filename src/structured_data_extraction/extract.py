"""LLM-based structured extraction : schema discovery and value extraction.

1. Schema discovery: a single LLM call (SCHEMA_MODEL) examines the matched
   sections and the user's keys, and returns a per-key field specification
   (sub-fields, types, units) plus a document_type label. The spec is used to
   build a Pydantic model on the fly so the extraction LLM's output is
   validated and typed automatically via the Instructor library.

2. Parallel extraction: every model listed in EXTRACTION_MODELS extracts values
   independently, each through Instructor with retries. Running multiple models
   produces overlapping records whose agreement is later scored by rater.py;
   with a single model the extraction still works, just without a consensus
   signal.

Prompt templates (schema_discovery.j2 and extraction.j2) are Jinja files that
ship inside the package. Users can override them by running
`--dump-templates ./dir` and setting TEMPLATES_DIR to that directory.
"""
import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional
import instructor
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, PackageLoader
from openai import OpenAI
from pydantic import BaseModel, Field, create_model
from . import config, matching
from .paths import inter, read_json, write_json, cached
from .tables import strip_code_fence

logger = logging.getLogger(__name__)


_loaders = []
if os.environ.get("TEMPLATES_DIR"):
    _loaders.append(FileSystemLoader(os.environ["TEMPLATES_DIR"]))
_loaders.append(PackageLoader("structured_data_extraction", "templates"))
jinja_env = Environment(loader=ChoiceLoader(_loaders))
MAX_SECTION_CHARS = 5000


class ExtractionMetadata(BaseModel):
    source: str = ""
    section_type: str = ""
    specified: bool = True


class ValueRecord(BaseModel):
    qualifier: str = Field(default="", description="Sub-category or variant")
    condition: str = Field(default="", description="Conditions for this value")
    value: Any = Field(description="The extracted value")
    unit: str = Field(default="", description="Unit of measurement")
    metadata: ExtractionMetadata = Field(default_factory=ExtractionMetadata)

_raw = None


def _raw_client():
    global _raw
    if _raw is None:
        _raw = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
    return _raw


def _get_client():
    # instructor wraps the client so create(..., response_model=Model) returns a validated
    # Pydantic object instead of raw text
    return instructor.from_openai(_raw_client(), mode=instructor.Mode.JSON)


def _build_extraction_model(schema, user_keys):
    """Build a Pydantic ExtractionResult model on the fly, one list field per user key.

    A key whose discovered schema has a `fields` spec gets a bespoke record model (one
    sub-field per entry, plus a synthesised `<name>_unit` when a field is unit-bearing and
    there's no shared `unit`). Keys without a spec fall back to the generic ValueRecord.
    The model is handed to Instructor as the response_model.
    """
    # number is Optional so a model returning null (e.g. "volume": null) still validates
    type_map = {"string": str, "number": Optional[float], "any": Any, "object": dict}
    # schema keys come straight from the LLM, which may not match normalize_key's casing
    # (e.g. it returns "PSRR" but normalize_key("PSRR") == "psrr"). Look up normalized so an
    # acronym field keeps its rich (differentiator/qualifier) schema instead of falling back
    # to the generic ValueRecord.
    schema_by_norm = ({matching.normalize_key(k): v for k, v in schema.items()}
                      if isinstance(schema, dict) else {})
    field_defs = {}

    for key in user_keys:
        field_name = matching.normalize_key(key)
        entry = schema_by_norm.get(field_name)
        if not isinstance(entry, dict):
            entry = {}
        description = entry.get("description") or key

        fields_spec = entry.get("fields")
        if not fields_spec:
            field_defs[field_name] = (List[ValueRecord], Field(default_factory=list, description=description))
            continue

        sub_fields = {
            "metadata": (ExtractionMetadata, Field(default_factory=ExtractionMetadata)),
        }

        has_unit_field = "unit" in fields_spec
        for sub_name, sub_def in fields_spec.items():
            if sub_name == "metadata":
                continue
            sub_desc = sub_def.get("description", "")
            fixed = sub_def.get("fixed")
            if fixed is not None:
                field_type = Literal[fixed]
                sub_fields[sub_name] = (field_type, Field(default=fixed, description=sub_desc))
            else:
                field_type = type_map.get(sub_def.get("type", "string"), str)
                default = "" if field_type is str else None
                sub_fields[sub_name] = (field_type, Field(default=default, description=sub_desc))
            if sub_def.get("unit") and not has_unit_field:
                sub_fields[f"{sub_name}_unit"] = (str, Field(default="", description=f"Unit for {sub_name}"))

        record_model = create_model(f"{field_name}_Record", **sub_fields)
        field_defs[field_name] = (List[record_model], Field(default_factory=list, description=description))

    return create_model("ExtractionResult", **field_defs)


def _merge_cleaned_tables(sections, tables):
    """Swap each table section's content for its vision-cleaned HTML and tag every
    non-header section with the most recent preceding heading (nearest_header)."""
    table_by_idx = {table["section_idx"]: table for table in tables}
    merged = []
    last_header = ""

    for i, section in enumerate(sections):
        merged_section = dict(section)
        if section.get("type") == "header":
            last_header = (section.get("content") or "").strip()

        # tables are keyed by section_idx: try the positional index first, then the stored one
        idx = i if i in table_by_idx else section.get("section_idx")
        if section.get("type") == "table" and idx in table_by_idx:
            cleaned = table_by_idx[idx].get("content", "")
            if cleaned:
                merged_section["content"] = cleaned
                merged_section["refined_format"] = table_by_idx[idx].get("refined_format", "")

        if section.get("type") != "header" and last_header:
            merged_section["nearest_header"] = last_header
        merged.append(merged_section)
    return merged


def _format_sections(sections, max_chars=MAX_SECTION_CHARS):
    """Render the matched sections into the prompt block: prefer full_content, prefix the nearest heading, and truncate long prose but
    never tables."""
    parts = []
    for i, section in enumerate(sections):
        sec_type = section.get("type", "unknown")
        sec_idx = section.get("section_idx", i)
        content = section.get("full_content", "") or section.get("content", "")
        header = section.get("nearest_header", "")

        header_prefix = f"Surrounding section heading: {header}\n" if header else ""

        if sec_type != "table" and len(content) > max_chars:
            content = content[:max_chars] + "..."

        parts.append(f"[Section {sec_idx}] ({sec_type}):\n{header_prefix}{content}")

    return "\n\n".join(parts)


def discover_schema(markdown_text, matched_sections, vocab, user_keys):
    """Ask the schema LLM for each key's field shape. Returns document_type, schema,
    pulling document_type out of the response's document_type"""
    template = jinja_env.get_template("schema_discovery.j2")

    section_content = _format_sections(matched_sections)
    vocab_terms = ", ".join(item["term"] for item in vocab[:30]) if vocab else "N/A"

    prompt = template.render(
        section_content=section_content,
        vocab_terms=vocab_terms,
        user_keys=", ".join(user_keys),
        document_preview=markdown_text[:5000],
    )

    logger.info("Calling LLM for schema discovery...")
    client = _raw_client()
    resp = client.chat.completions.create(
        model=os.environ["SCHEMA_MODEL"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = strip_code_fence(resp.choices[0].message.content.strip())

    try:
        schema = json.loads(raw)
        if isinstance(schema, dict):
            doc_type = schema.pop("_document_type", "unknown")
            if not isinstance(doc_type, str) or not doc_type.strip():
                doc_type = "unknown"
            return doc_type, schema
    except json.JSONDecodeError:
        logger.error(f"Failed to parse schema: {raw[:200]}")
    return "unknown", {}


def extract_values(client, matched_sections, vocab, user_keys, schema,
                   document_type="unknown", source_file="", model=None):
    """Run one extraction LLM through Instructor and return {field: [record dicts]}.
    Retries up to MAX_LLM_RETRIES; returns {} if every attempt fails."""
    model = model or config.EXTRACTION_MODELS[0]
    template = jinja_env.get_template("extraction.j2")

    section_content = _format_sections(matched_sections)
    vocab_terms = ", ".join(item["term"] for item in vocab[:30]) if vocab else "N/A"

    prompt = template.render(
        document_type=document_type,
        section_content=section_content,
        vocab_terms=vocab_terms,
        user_keys=", ".join(user_keys),
        schema=json.dumps(schema, indent=2),
        source_file=source_file,
    )

    extraction_model = _build_extraction_model(schema, user_keys)

    logger.info(f"Calling LLM for extraction (model={model})...")
    last_err = None
    max_retries = int(os.environ["MAX_LLM_RETRIES"])

    for attempt in range(max_retries + 1):
        try:
            result = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_model=extraction_model,
            )
            output = {}
            for field in extraction_model.model_fields:
                records = getattr(result, field, [])
                output[field] = [record.model_dump() for record in records]
            return output
        except Exception as e:
            last_err = e
            logger.warning(f"[{model}] Attempt {attempt + 1} failed: {e}")

    logger.error(f"[{model}] Extraction failed after {max_retries + 1} attempts: {last_err}")
    return {}


def _dedup_extraction(extraction):
    """Within each field, drop duplicate records, keeping the one from the highest-priority
    section (table > text > list). Records are duplicates when all non-metadata fields match."""
    deduped = {}
    for field, records in extraction.items():
        best = {}
        for record in records:
            # identity key over every field except metadata: equal values -> same fact
            dedup_key = tuple(sorted((k, str(v)) for k, v in record.items() if k != "metadata"))

            sec_type = record.get("metadata", {}).get("section_type", "unknown")
            existing = best.get(dedup_key)
            if existing is None:
                best[dedup_key] = record
            else:
                old_priority = config.SECTION_TYPE_PRIORITY.get(existing.get("metadata", {}).get("section_type", "unknown"), 99)
                new_priority = config.SECTION_TYPE_PRIORITY.get(sec_type, 99)
                if new_priority < old_priority:
                    best[dedup_key] = record
        deduped[field] = list(best.values())
    return deduped


def _neighbor_line(sections, idx, side):
    """Return the adjacent line beside section `idx`: the last line of the previous section
    or the first line of the next. Empty string if that neighbour is a header/table or out of bounds."""
    j = idx - 1 if side == "prev" else idx + 1
    if not 0 <= j < len(sections):
        return ""
    neighbor = sections[j]
    if neighbor.get("type") in ("header", "table"):
        return ""
    lines = [ln for ln in (neighbor.get("content") or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    return lines[-1] if side == "prev" else lines[0]


def _prepare_sections(stem):
    """Load cached matches/sections/vocab and build the deduped list of sections to extract from.

    Each matched section is widened by one neighbour line on each side (so an orphaned label
    keeps its value), then collapsed by section_idx keeping the highest-similarity hit.
    Returns (markdown_text, vocab, sections_sorted_by_similarity).
    """
    matches = read_json(inter(stem, "matches.json"))
    sections = read_json(inter(stem, "sections.json"))
    vocab = read_json(inter(stem, "vocab.json"))

    if inter(stem, "tables.json").exists():
        sections = _merge_cleaned_tables(sections, read_json(inter(stem, "tables.json")))

    md_path = Path(config.MARKDOWN_DIR) / f"{stem}.md"
    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    best_by_idx = {}
    for key_matches in matches.values():
        for match in key_matches:
            idx = match.get("section_idx", 0)
            enriched = dict(match)
            if idx < len(sections):
                content = sections[idx].get("content", "")
                sec_type = sections[idx].get("type", "unknown")
                if sec_type != "table":
                    # widen by one neighbor line each side so an orphaned label keeps
                    # its value (and vice versa)
                    prev_line = _neighbor_line(sections, idx, "prev")
                    next_line = _neighbor_line(sections, idx, "next")
                    content = "\n".join(p for p in (prev_line, content, next_line) if p)
                enriched["full_content"] = content
                enriched["type"] = sec_type
            similarity = enriched.get("similarity", 0)
            if idx not in best_by_idx or similarity > best_by_idx[idx].get("similarity", 0):
                best_by_idx[idx] = enriched

    unique = sorted(best_by_idx.values(), key=lambda x: x.get("similarity", 0), reverse=True)
    return md_text, vocab, unique


def _run_one_model(model_name, stem, out_dir, unique, vocab, user_keys, schema, doc_type):
    """Extract with one model, dedup, stamp each record with its model/doc_type, and write
    output/<stem>_extraction_<slug>.json. Returns the cached file unchanged on a re-run."""
    slug = matching.model_slug(model_name)
    out_path = out_dir / f"{stem}_extraction_{slug}.json"
    if out_path.exists() and not config.FORCE:
        logger.info(f"[{model_name}] already exists, skipping")
        return json.loads(out_path.read_text(encoding="utf-8"))

    client = _get_client()
    extraction = extract_values(client, unique, vocab, user_keys, schema,
                                doc_type, f"{stem}.pdf", model=model_name)

    if not extraction:
        # extract_values returns {} only when every attempt failed; surface it as a failure
        # rather than writing an empty file that a later run would treat as a cached success.
        raise RuntimeError("no extraction produced (all attempts failed)")

    pre = sum(len(v) for v in extraction.values())
    extraction = _dedup_extraction(extraction)
    post = sum(len(v) for v in extraction.values())
    if post < pre:
        logger.info(f"[{model_name}] Dedup: {pre} → {post}")

    # tag each record with which model produced it
    for records in extraction.values():
        for record in records:
            metadata = record.get("metadata")
            if isinstance(metadata, dict):
                metadata["document_type"] = doc_type
                metadata["model"] = model_name

    total = sum(len(v) for v in extraction.values())
    out_path.write_text(json.dumps(extraction, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[{model_name}] Extracted {total} records → {out_path}")
    return extraction


def extract_parallel(stem, user_keys, user_schema=None):
    """Run every configured extraction model over one document in parallel.

    Stamps run provenance (run.json), discovers the schema once for all models, then fans
    _run_one_model out across EXTRACTION_WORKERS threads. Returns the per-model extraction
    dicts for the runs that succeeded.
    """
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = config.EXTRACTION_MODELS


    cached(inter(stem, "run.json"),
           lambda: {"extracted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "user_keys": list(user_keys)})

    md_text, vocab, unique = _prepare_sections(stem)

    if user_schema is not None:
        doc_type, schema = user_schema["document_type"], user_schema["schema"]
        write_json(inter(stem, "schema.json"), user_schema)
        logger.info("Using user-supplied schema (skipped schema discovery)")
    else:
        disc = cached(inter(stem, "schema.json"),
                      lambda: dict(zip(("document_type", "schema"),
                                       discover_schema(md_text, unique, vocab, user_keys))))
        doc_type, schema = disc["document_type"], disc["schema"]
    logger.info(f"Schema: document_type='{doc_type}'")

    results = []
    failures = []
    workers = max(1, min(int(os.environ["EXTRACTION_WORKERS"]), len(models)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_one_model, m, stem, out_dir, unique, vocab, user_keys, schema, doc_type): m
            for m in models
        }
        for fut in as_completed(futures):
            model_name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                # one model failing doesn't sink the document — the others still write output.
                # keep only the first line of the message so the user sees the cause, not a trace.
                msg = next((ln for ln in str(e).splitlines() if ln.strip()), type(e).__name__)
                failures.append((model_name, msg))
                logger.error(f"[{model_name}] extraction failed: {msg}")

    if failures and not results:
        model_name, msg = failures[0]
        logger.error(f"All {len(models)} extraction model(s) failed. First error — [{model_name}]: {msg}")
    elif failures:
        logger.warning(f"Partial extraction: {len(results)}/{len(models)} models OK; "
                       f"failed: {', '.join(m for m, _ in failures)}")
    logger.info(f"Parallel extraction done: {len(results)}/{len(models)} succeeded")
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract values from document")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    args = parser.parse_args()

    results = extract_parallel(args.stem, args.keys)
    for result in results:
        total = sum(len(records) for records in result.values())
        model = next((rec.get("metadata", {}).get("model")
                      for records in result.values() for rec in records), "?")
        print(f"{model}: {total} records")


if __name__ == "__main__":
    main()
