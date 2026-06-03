import argparse
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import instructor
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from . import config
from .table_filter import filter_html_table, filter_markdown_table
from .abbreviations import synonyms_for_keys
from .tables import strip_code_fence

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

jinja_env = Environment(loader=FileSystemLoader(config.TEMPLATES_DIR))
MAX_SECTION_CHARS = 5000 # cap for free-text/list sections only; tables aren't truncated


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

SECTION_TYPE_PRIORITY = {"table": 0, "text": 1, "list": 2}


def _raw_client():
    return OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)


def _get_client():
    return instructor.from_openai(_raw_client(), mode=instructor.Mode.JSON)


def _build_extraction_model(schema, user_keys):
    type_map = {"string": str, "number": float, "any": Any, "object": dict}
    field_defs = {}

    for key in user_keys:
        field_name = re.sub(r'[^a-z0-9_]', '_', key.lower())
        entry = schema.get(field_name)
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
        for sub_name, sub_def in fields_spec.items():
            if sub_name == "metadata":
                continue
            field_type = type_map.get(sub_def.get("type", "string"), str)
            sub_desc = sub_def.get("description", "")
            default = "" if field_type is str else None
            sub_fields[sub_name] = (field_type, Field(default=default, description=sub_desc))
            if sub_def.get("unit"):
                sub_fields[f"{sub_name}_unit"] = (str, Field(default="", description=f"Unit for {sub_name}"))

        record_model = create_model(f"{field_name}_Record", **sub_fields)
        field_defs[field_name] = (List[record_model], Field(default_factory=list, description=description))

    return create_model("ExtractionResult", **field_defs)


def _merge_cleaned_tables(sections, tables):
    table_by_idx = {table["section_idx"]: table for table in tables}
    merged = []
    last_header = ""

    for i, section in enumerate(sections):
        merged_section = dict(section)
        if section.get("type") == "header":
            last_header = (section.get("content") or "").strip()

        idx = i if i in table_by_idx else section.get("section_idx")
        if section.get("type") == "table" and idx in table_by_idx:
            cleaned = table_by_idx[idx].get("content", "")
            if cleaned:
                merged_section["content"] = cleaned
                merged_section["refined_format"] = table_by_idx[idx].get("refined_format", "")

        # attach the most recent heading to every non-header section so the LLM
        # has structural context regardless of section type
        if section.get("type") != "header" and last_header:
            merged_section["nearest_header"] = last_header
        merged.append(merged_section)
    return merged


def _row_text_first_cell(row_text):
    if not row_text:
        return ""
    cells = [cell.strip() for cell in row_text.split("|") if cell.strip()]
    if not cells:
        return ""
    first = cells[0]
    # row_text may have a "[col_header] value" annotation from step5 — strip it
    if first.startswith("[") and "]" in first:
        first = first.split("]", 1)[1].strip()
    return first


def _format_sections(sections, row_matches=None, max_chars=MAX_SECTION_CHARS):
    matched_rows_by_idx = {}
    matched_ids_by_idx = {}
    if row_matches:
        for match in row_matches:
            if not match.get("is_row"):
                continue
            idx = match.get("section_idx")
            matched_rows_by_idx.setdefault(idx, set()).add(match.get("row_number", -1))
            first = _row_text_first_cell(match.get("row_text", ""))
            if first:
                matched_ids_by_idx.setdefault(idx, set()).add(first)
            group_header = match.get("group_header") or ""
            if group_header:
                matched_ids_by_idx.setdefault(idx, set()).add(group_header)

    parts = []
    for i, section in enumerate(sections):
        sec_type = section.get("type", "unknown")
        sec_idx = section.get("section_idx", i)
        content = section.get("full_content", "") or section.get("content", "")
        refined = section.get("refined_format", "")
        header = section.get("nearest_header", "")

        header_prefix = f"Surrounding section heading: {header}\n" if header else ""

        if sec_type == "table" and content:
            filtered = None
            if refined == "html" and sec_idx in matched_ids_by_idx:
                filtered = filter_html_table(content, matched_ids_by_idx[sec_idx])
            elif sec_idx in matched_rows_by_idx:
                filtered = filter_markdown_table(content, matched_rows_by_idx[sec_idx])

            # tables are the high-value content: use the filtered group when we have
            # one, otherwise send the whole table — never hard-cut it mid-row
            if filtered:
                content = filtered
        else:
            if len(content) > max_chars:
                content = content[:max_chars] + "..."

        parts.append(f"[Section {sec_idx}] ({sec_type}):\n{header_prefix}{content}")

    return "\n\n".join(parts)


def _format_synonyms(synonyms):
    # "cmrr → Common Mode Rejection Ratio" lines for the prompt; empty if none
    if not synonyms:
        return ""
    lines = [f"- {key} → {', '.join(values)}" for key, values in synonyms.items() if values]
    return "\n".join(lines)


def discover_schema(markdown_text, matched_sections, vocab, user_keys, row_matches=None,
                    synonyms=None):
    template = jinja_env.get_template("schema_discovery.j2")

    section_content = _format_sections(matched_sections, row_matches=row_matches)
    vocab_terms = ", ".join(item["term"] for item in vocab[:30]) if vocab else "N/A"

    prompt = template.render(
        section_content=section_content,
        vocab_terms=vocab_terms,
        user_keys=", ".join(user_keys),
        key_synonyms=_format_synonyms(synonyms),
        document_preview=markdown_text[:5000],
    )

    logger.info("Calling LLM for schema discovery...")
    client = _raw_client()
    resp = client.chat.completions.create(
        model=config.OLLAMA_SCHEMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    raw = strip_code_fence(resp.choices[0].message.content.strip())

    try:
        schema = json.loads(raw)
        if isinstance(schema, dict):
            # the model emits _document_type alongside the field defs; pop it out
            doc_type = schema.pop("_document_type", "unknown")
            if not isinstance(doc_type, str) or not doc_type.strip():
                doc_type = "unknown"
            return doc_type, schema
    except json.JSONDecodeError:
        logger.error(f"Failed to parse schema: {raw[:200]}")
    return "unknown", {}


def extract_values(client, matched_sections, vocab, user_keys, schema,
                   document_type="unknown", source_file="", row_matches=None,
                   model=None, synonyms=None):
    model = model or config.OLLAMA_EXTRACT_MODEL
    template = jinja_env.get_template("extraction.j2")

    section_content = _format_sections(matched_sections, row_matches=row_matches)
    vocab_terms = ", ".join(item["term"] for item in vocab[:30]) if vocab else "N/A"

    prompt = template.render(
        document_type=document_type,
        section_content=section_content,
        vocab_terms=vocab_terms,
        user_keys=", ".join(user_keys),
        key_synonyms=_format_synonyms(synonyms),
        schema=json.dumps(schema, indent=2),
        source_file=source_file,
    )

    extraction_model = _build_extraction_model(schema, user_keys)

    logger.info(f"Calling LLM for extraction (model={model})...")
    last_err = None

    for attempt in range(config.MAX_LLM_RETRIES + 1):
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

    logger.error(f"[{model}] Extraction failed after {config.MAX_LLM_RETRIES + 1} attempts: {last_err}")
    return {}


def _dedup_extraction(extraction):
    deduped = {}
    for field, records in extraction.items():
        best = {}
        for record in records:
            dedup_key = tuple(sorted((k, str(v)) for k, v in record.items() if k != "metadata"))

            sec_type = record.get("metadata", {}).get("section_type", "unknown")
            existing = best.get(dedup_key)
            if existing is None:
                best[dedup_key] = record
            else:
                old_priority = SECTION_TYPE_PRIORITY.get(existing.get("metadata", {}).get("section_type", "unknown"), 99)
                new_priority = SECTION_TYPE_PRIORITY.get(sec_type, 99)
                if new_priority < old_priority:
                    best[dedup_key] = record
        deduped[field] = list(best.values())
    return deduped


def _prepare_sections(matches_path, sections_path, vocab_path, markdown_path, tables_path):
    # shared loading + matched-section prep used by both extract paths
    matches = json.loads(Path(matches_path).read_text(encoding="utf-8"))
    sections = json.loads(Path(sections_path).read_text(encoding="utf-8"))
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))

    if tables_path and Path(tables_path).exists():
        tables = json.loads(Path(tables_path).read_text(encoding="utf-8"))
        sections = _merge_cleaned_tables(sections, tables)

    md_text = ""
    if markdown_path and Path(markdown_path).exists():
        md_text = Path(markdown_path).read_text(encoding="utf-8")

    best_by_idx = {}
    row_matches = []
    for key_matches in matches.values():
        for match in key_matches:
            idx = match.get("section_idx", 0)
            enriched = dict(match)
            if idx < len(sections):
                enriched["full_content"] = sections[idx].get("content", "")
                enriched["type"] = sections[idx].get("type", "unknown")
                enriched["refined_format"] = sections[idx].get("refined_format", "")
            if enriched.get("is_row"):
                row_matches.append(enriched)
            similarity = enriched.get("similarity", 0)
            if idx not in best_by_idx or similarity > best_by_idx[idx].get("similarity", 0):
                best_by_idx[idx] = enriched

    unique = sorted(best_by_idx.values(), key=lambda x: x.get("similarity", 0), reverse=True)
    return md_text, vocab, unique, row_matches


def _build_extraction_result(stem, doc_type, schema, user_keys, num_unique,
                             extraction, extraction_model, slug=None):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fdo_id = f"fdo:{stem}:{slug}:{timestamp}" if slug else f"fdo:{stem}:{timestamp}"
    return {
        "id": fdo_id, "type": "ExtractionResult",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "source_file": f"{stem}.pdf", "document_type": doc_type,
            "pipeline_version": config.PIPELINE_VERSION,
            "models": {"schema_discovery": config.OLLAMA_SCHEMA_MODEL,
                       "extraction": extraction_model,
                       "vision": config.OLLAMA_VISION_MODEL,
                       "embedding": config.EMBEDDING_MODEL},
            "user_keys": user_keys, "matched_sections": num_unique,
            "schema_fields": list(schema.keys()),
        }, **extraction,
    }


def _model_slug(model_name):
    return re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')


def extract_parallel(matches_path, sections_path, vocab_path, user_keys,
                     markdown_path=None, tables_path=None):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    stem = Path(matches_path).stem.replace("_matches", "")
    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = config.EXTRACTION_MODELS

    md_text, vocab, unique, row_matches = _prepare_sections(
        matches_path, sections_path, vocab_path, markdown_path, tables_path)

    # acronym/spelled-out synonyms — reuse what embed cached, else derive from text
    syn_path = Path(config.INTERMEDIATE_DIR) / f"{stem}_key_synonyms.json"
    if syn_path.exists():
        synonyms = json.loads(syn_path.read_text(encoding="utf-8"))
    else:
        synonyms = synonyms_for_keys(user_keys, md_text)

    # schema discovery runs once for all models
    doc_type, schema = discover_schema(md_text, unique, vocab, user_keys, row_matches,
                                       synonyms=synonyms)
    logger.info(f"Schema: document_type='{doc_type}'")

    def run_one_model(model_name):
        slug = _model_slug(model_name)
        out_path = out_dir / f"{stem}_extraction_{slug}.json"
        if out_path.exists():
            logger.info(f"[{model_name}] already exists, skipping")
            return json.loads(out_path.read_text(encoding="utf-8"))

        client = _get_client()
        extraction = extract_values(client, unique, vocab, user_keys, schema,
                                    doc_type, f"{stem}.pdf", row_matches,
                                    model=model_name, synonyms=synonyms)

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

        result = _build_extraction_result(stem, doc_type, schema, user_keys, len(unique),
                                           extraction, model_name, slug=slug)

        total = sum(len(v) for v in extraction.values())
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"[{model_name}] Extracted {total} records → {out_path}")
        return result

    results = []
    with ThreadPoolExecutor(max_workers=config.EXTRACTION_WORKERS) as pool:
        futures = {pool.submit(run_one_model, m): m for m in models}
        for fut in as_completed(futures):
            model_name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                logger.error(f"[{model_name}] extraction failed: {e}")

    logger.info(f"Parallel extraction done: {len(results)}/{len(models)} succeeded")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract values from document")
    parser.add_argument("--stem", required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    args = parser.parse_args()

    int_dir = Path(config.INTERMEDIATE_DIR)
    md_dir = Path(config.MARKDOWN_DIR)

    results = extract_parallel(
        matches_path=str(int_dir / f"{args.stem}_matches.json"),
        sections_path=str(int_dir / f"{args.stem}_sections.json"),
        vocab_path=str(int_dir / f"{args.stem}_vocab.json"),
        user_keys=args.keys,
        markdown_path=str(md_dir / f"{args.stem}.md"),
        tables_path=str(int_dir / f"{args.stem}_tables.json"),
    )
    meta_keys = {"id", "type", "generated_at", "provenance"}
    for result in results:
        total = sum(len(records) for field, records in result.items() if field not in meta_keys)
        model = result.get("provenance", {}).get("models", {}).get("extraction", "?")
        print(f"{model}: {total} records")
