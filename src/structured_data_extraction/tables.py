import base64
import concurrent.futures
import json
import logging
import re
from collections import Counter
from pathlib import Path
import fitz
from openai import OpenAI
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _ramp(x, lo, hi):
    # 0 below lo, 1 above hi, linear in between
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def score_table_complexity(content):

    text = content.lower()
    score = sum(0.15 for m in ("colspan", "rowspan", "<figure", "<img", "nested") if m in text)

    pipe_rows = [line for line in content.split("\n") if "|" in line]
    if pipe_rows:
        delims = [row.count("|") for row in pipe_rows]
        mode = Counter(delims).most_common(1)[0][0]

        # broken row structure: rows off the modal delimiter count
        ragged = sum(1 for d in delims if d != mode) / len(delims)

        # sparsity + hierarchy over data rows (separators skipped)
        data = []
        for row in pipe_rows:
            inner = [c.strip() for c in row.split("|")[1:-1]]
            if inner and all(set(c) <= {"-", ":", " ", ""} for c in inner):
                continue
            data.append(inner)
        cells = sum(len(r) for r in data)
        empty = sum(1 for r in data for c in r if not c)
        sparsity = empty / cells if cells else 0.0
        group_rows = sum(1 for r in data if r and r[0] and not any(r[1:]))
        hierarchy = group_rows / len(data) if data else 0.0
        ncols = max(mode - 1, 1)

        score += _ramp(sparsity, 0.15, 0.60) * 0.40   # sparse grid (dominant)
        score += ragged * 0.30                         # broken rows
        score += min(hierarchy / 0.30, 1.0) * 0.20     # nested groups
        score += _ramp(ncols, 4, 12) * 0.10            # wide tables

    return min(score, 1.0)


def _render(pdf_path, page_number, bbox=None, dpi=200):
    # render a whole page, or just the bbox region if one is given
    doc = fitz.open(pdf_path)
    idx = max(0, min(page_number - 1, len(doc) - 1))
    page = doc[idx]

    clip = None
    if bbox:
        h = page.rect.height
        # docling uses BOTTOMLEFT origin, PyMuPDF uses top-left
        origin = bbox.get("coord_origin", "BOTTOMLEFT")
        if "BOTTOM" in origin:
            clip = fitz.Rect(bbox["l"], h - bbox["t"], bbox["r"], h - bbox["b"])
        else:
            clip = fitz.Rect(bbox["l"], bbox["t"], bbox["r"], bbox["b"])

    pix = page.get_pixmap(dpi=dpi, clip=clip)
    b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
    doc.close()
    return b64


TABLE_VISION_PROMPT = (
    "You are a table-to-HTML conversion expert. Look at the table in the provided image "
    "and convert it to clean, semantic HTML. Use proper <thead>, <tbody>, <th>, <td> tags. "
    "Merge cells only where the original has merged cells. "
    "Preserve ALL values exactly as shown — do not omit or invent data. "
    "Do NOT add any explanation, just output the HTML table.\n\n"
    "Here is the markdown version of the table for reference (it may be garbled):\n"
)


def strip_code_fence(text):
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text


def _client():
    return OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.OLLAMA_API_KEY)


def _call_vision_model(b64_img, table_md):
    client = _client()
    resp = client.chat.completions.create(
        model=config.OLLAMA_VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": TABLE_VISION_PROMPT + table_md},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
            ],
        }],
        temperature=0.0,
    )
    return strip_code_fence(resp.choices[0].message.content.strip())


def _call_text_model(table_md):
    client = _client()
    prompt = (
        "Convert the following markdown table to clean, semantic HTML. "
        "Use proper <thead>, <tbody>, <th>, <td> tags. "
        "Merge cells only where the original has merged cells. "
        "Do NOT add any explanation, just output the HTML table.\n\n"
        + table_md
    )
    resp = client.chat.completions.create(
        model=config.OLLAMA_EXTRACT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return strip_code_fence(resp.choices[0].message.content.strip())


def process_tables(sections_path, pdf_path=None):
    sections_path = Path(sections_path)
    if not sections_path.exists():
        raise FileNotFoundError(f"Sections not found: {sections_path}")

    stem = sections_path.stem.replace("_sections", "")
    out_path = Path(config.INTERMEDIATE_DIR) / f"{stem}_tables.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.info(f"Skipping tables — already exists")
        return json.loads(out_path.read_text(encoding="utf-8"))

    sections = json.loads(sections_path.read_text(encoding="utf-8"))
    table_sections = [(i, s) for i, s in enumerate(sections) if s["type"] == "table"]

    if not table_sections:
        logger.info("No tables found")
        out_path.write_text("[]", encoding="utf-8")
        return []

    # try to find the PDF if not provided
    resolved_pdf = pdf_path
    if not resolved_pdf:
        candidate = Path(config.PDF_DIR) / f"{stem}.pdf"
        if candidate.exists():
            resolved_pdf = str(candidate)

    logger.info(f"Scoring {len(table_sections)} tables...")
    results = []

    for i, (section_idx, section) in enumerate(table_sections):
        content = section["content"]
        complexity = score_table_complexity(content)
        page = section.get("page") or 1

        logger.info(f"  Table {i}: complexity={complexity:.2f}")

        results.append({
            "section_idx": section_idx,
            "complexity_score": round(complexity, 4),
            "page": page,
            "content": content,
            "bbox": section.get("bbox"),
            "pdf_path": resolved_pdf,
        })

    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Scored {len(results)} tables → {out_path}")
    return results


def _refine_one_table(table):
    idx = table["section_idx"]
    pdf_path = table.get("pdf_path")
    page = table.get("page", 1)
    content = table.get("content", "")
    bbox = table.get("bbox")

    html = None
    used_vision = False
    if pdf_path and Path(pdf_path).exists():
        try:
            image_b64 = _render(pdf_path, page, bbox)
            html = _call_vision_model(image_b64, content)
            used_vision = True
        except Exception as e:
            logger.warning(f"  Section {idx}: vision failed ({e}), falling back to text LLM")

    if html is None:
        try:
            html = _call_text_model(content)
        except Exception as e:
            logger.error(f"  Section {idx}: failed — {e}")
            return idx, None, False

    return idx, html, used_vision


def refine_matched_tables(tables_path, matched_section_indices):
    tables_path = Path(tables_path)
    if not tables_path.exists():
        return []

    tables = json.loads(tables_path.read_text(encoding="utf-8"))



    matched = [t for t in tables if t.get("section_idx") in matched_section_indices]
    to_refine = [t for t in matched
                 if t.get("complexity_score", 0) >= config.VISION_COMPLEXITY_THRESHOLD]

    skipped = len(matched) - len(to_refine)
    if skipped:
        logger.info(f"  Refining {len(to_refine)} matched tables above complexity "
                    f"{config.VISION_COMPLEXITY_THRESHOLD}; {skipped} simpler ones left as markdown")

    if not to_refine:
        for table in tables:
            table.pop("pdf_path", None)
        tables_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
        return tables

    workers = max(1, min(config.VISION_PARALLELISM, len(to_refine)))
    logger.info(f"  Refining {len(to_refine)} tables ({workers} workers in parallel)...")

    by_idx = {table["section_idx"]: table for table in tables}
    vision_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_refine_one_table, table) for table in to_refine]
        for fut in concurrent.futures.as_completed(futures):
            idx, html, used_vision = fut.result()
            if html is None:
                continue
            by_idx[idx]["original_content"] = by_idx[idx].get("content", "")
            by_idx[idx]["content"] = html
            by_idx[idx]["refined"] = True
            by_idx[idx]["refined_format"] = "html"
            if used_vision:
                vision_count += 1
            logger.info(f"  Section {idx}: refined ({len(html)} chars, vision={used_vision})")

    for table in tables:
        table.pop("pdf_path", None)

    tables_path.write_text(json.dumps(tables, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Refined {vision_count} tables via vision model (of {len(to_refine)} attempted)")
    return tables
