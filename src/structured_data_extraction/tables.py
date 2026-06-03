import base64
import concurrent.futures
import json
import logging
import re
from pathlib import Path
import fitz
from openai import OpenAI
from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def score_table_complexity(content):
    score = 0.0
    lines = content.split("\n")
    if not lines:
        return 0.0

    for marker in ["colspan", "rowspan", "<figure", "<img", "nested"]:
        if marker in content.lower():
            score += 0.15

    # ragged column counts = merged cells
    pipe_rows = [line for line in lines if "|" in line]
    if pipe_rows:
        col_counts = {len([c for c in row.split("|") if c.strip()]) for row in pipe_rows}
        if len(col_counts) > 1:
            score += 0.2

        for row in pipe_rows[:5]:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            for cell in cells:
                if "\n" in cell or len(cell) > 100:
                    score += 0.05
                    break

    for pattern in [r"\d+\.\d+[eE][+-]?\d+", r"±", r"×10"]:
        if re.search(pattern, content):
            score += 0.1
            break

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
        route = "complex" if complexity >= config.TABLE_COMPLEXITY_THRESHOLD else "simple"
        page = section.get("page") or 1

        logger.info(f"  Table {i}: complexity={complexity:.2f}, route={route}")

        results.append({
            "section_idx": section_idx,
            "complexity_score": round(complexity, 4),
            "route": route,
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

    to_refine = []
    for table in tables:
        if table.get("section_idx") not in matched_section_indices:
            continue
        if table.get("route") != "complex":
            logger.info(f"  Skipping simple table (section {table['section_idx']})")
            continue
        to_refine.append(table)

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
