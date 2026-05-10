"""
pdf_processor.py  –  Research paper PDF extraction
====================================================

FIX 1 — Subsection hierarchy
  Sections numbered 2.1, 2.2, 3.1.1 etc. are nested under their parent
  section as subsections, not shown as flat siblings.

FIX 2 — Figure / table images
  Captions are matched to page regions; the region is cropped from the
  PDF page and base64-encoded as a PNG so the frontend can render
  <img src="data:image/png;base64,..."> directly.

FIX 3 — Entity extraction for library sidebar
  Keyword-based entity extraction (methods, datasets, models, metrics)
  stored on the returned dict so main.py can populate paper.entities
  even before the Gemini background job runs.

Extraction pipeline (priority order):
  1. GROBID  (ML + layout, runs at http://localhost:8070)
  2. pdfplumber fallback (font-size + bold heuristics)
  3. Raw text fallback

Public API (unchanged):
  extract_text_from_pdf(file_path, fmt_hint="auto") -> dict
  chunk_text(...)
  create_chunks_from_sections(...)
"""

from __future__ import annotations

import re
import io
import base64
import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter, defaultdict

import pdfplumber

# ── Config ─────────────────────────────────────────────────────────────────────

GROBID_URL     = os.getenv("GROBID_URL", "http://localhost:8070")
GROBID_TIMEOUT = int(os.getenv("GROBID_TIMEOUT", "120"))
FIGURE_DPI     = 150   # resolution for cropped figure images

# ── Shared constants ───────────────────────────────────────────────────────────

BACK_MATTER = {
    "references", "reference", "bibliography",
    "acknowledgments", "acknowledgements",
    "appendix", "appendices",
    "supplementary material", "supplemental",
    "broader impact",
}

KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "background", "preliminaries",
    "methodology", "methods", "approach", "model", "architecture", "framework",
    "experiments", "experimental setup", "results", "evaluation", "analysis",
    "discussion", "conclusion", "conclusions", "future work", "acknowledgments",
    "limitations", "ethics", "appendix", "training details", "data", "dataset",
    "training", "inference", "ablation", "overview", "motivation", "problem",
    "setup", "implementation", "baseline", "system", "formulation",
]

AFFIL_KW = [
    "university", "institute", "laboratory", "department", "school", "college",
    "center", "centre", "@", "http", "doi", "equal contribution", "correspond",
    "preprint", "abstract", "proceedings", "annual meeting",
    "association for", "conference", "workshop", "transactions", "pages ",
    "microsoft", "google", "amazon", "facebook", "meta", "apple",
    "deepmind", "openai", "nvidia", "intel", "ibm", "research", "brain",
]

VENUE_RE = [
    r"proceedings of", r"annual meeting of",
    r"association for computational linguistics",
    r"neural information processing", r"international conference on",
    r"pages \d+[\s\u2013\-]\d+", r"c\s*[©]\s*\d{4}", r"^\d{3,4}$",
    r"^[A-Z][a-z]+,\s*[A-Z][a-z]+\s+\d{4}",
]

# ── FIX 3: entity keyword lists ────────────────────────────────────────────────

METHOD_KEYWORDS = [
    "Transformer", "BERT", "GPT", "LLM", "LSTM", "CNN", "RNN", "Attention",
    "Gradient Descent", "Adam", "SGD", "Neural Network", "Random Forest",
    "SVM", "Reinforcement Learning", "Fine-tuning", "Contrastive Learning",
    "Diffusion", "GAN", "Retrieval", "RAG", "GraphRAG", "Embedding",
    "Knowledge Graph", "Louvain", "Leiden", "Community Detection",
    "Map-Reduce", "Self-Reflection", "Chain-of-Thought", "In-Context Learning",
    "Prompt Engineering", "Zero-Shot", "Few-Shot", "Instruction Tuning",
]

DATASET_KEYWORDS = [
    "ImageNet", "COCO", "SQuAD", "GLUE", "SuperGLUE", "MNIST", "CIFAR",
    "Wikipedia", "CommonVoice", "LibriSpeech", "MS MARCO",
    "Natural Questions", "TriviaQA", "HotPotQA", "MultiHop-RAG",
    "MT-Bench", "MMLU", "HellaSwag", "ARC", "TruthfulQA",
]

MODEL_KEYWORDS = [
    "GPT-4", "GPT-3.5", "GPT-3", "LLaMA", "LLaMA 2", "Gemini", "Claude",
    "PaLM", "Mistral", "Falcon", "Vicuna", "Alpaca", "BERT", "RoBERTa",
    "T5", "FLAN-T5", "DeBERTa", "ELECTRA", "BART", "Codex", "StarCoder",
    "DeepSeek", "gpt-4-turbo",
]

METRIC_KEYWORDS = [
    "Accuracy", "F1", "F1 Score", "BLEU", "ROUGE", "Precision", "Recall",
    "Perplexity", "Exact Match", "MRR", "NDCG", "MAP",
    "Win Rate", "Comprehensiveness", "Diversity",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_path: str, fmt_hint: str = "auto") -> Dict[str, Any]:
    """
    Main extraction entry point.
    Tries GROBID first, falls back to pdfplumber, then raw text.
    FIX 3: entities are extracted immediately from full_text on every path.
    """
    try:
        result = _extract_with_grobid(file_path)
        if result and result.get("full_text", "").strip():
            print(f"✅ GROBID extraction succeeded for {Path(file_path).name}")
            result["entities"] = _extract_entities_from_text(result["full_text"])
            return result
        print("⚠️  GROBID returned empty — falling back to pdfplumber")
    except GrobidUnavailableError:
        print("⚠️  GROBID not running — falling back to pdfplumber")
    except Exception as e:
        print(f"⚠️  GROBID error ({type(e).__name__}: {e}) — falling back to pdfplumber")

    try:
        result = _extract_pdfplumber_linear(file_path)
        result["entities"] = _extract_entities_from_text(result.get("full_text", ""))
        return result
    except Exception as e:
        print(f"⚠️  pdfplumber failed ({e}), using raw fallback")
        result = _fallback(file_path)
        result["entities"] = _extract_entities_from_text(result.get("full_text", ""))
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3 — fast keyword entity extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_entities_from_text(full_text: str) -> Dict[str, List[str]]:
    """
    Keyword-based entity extraction run immediately on upload.
    Populates paper.entities before the slow Gemini background job finishes.
    """
    text_lower = full_text.lower()

    def _find(keywords: List[str], limit: int) -> List[str]:
        found = []
        for kw in keywords:
            if kw.lower() in text_lower:
                found.append(kw)
            if len(found) >= limit:
                break
        return found

    return {
        "methods":  _find(METHOD_KEYWORDS, 8),
        "datasets": _find(DATASET_KEYWORDS, 6),
        "models":   _find(MODEL_KEYWORDS, 6),
        "metrics":  _find(METRIC_KEYWORDS, 5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# GROBID path
# ═══════════════════════════════════════════════════════════════════════════════

class GrobidUnavailableError(Exception):
    pass


def _grobid_is_alive() -> bool:
    try:
        r = requests.get(f"{GROBID_URL}/api/isalive", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _extract_with_grobid(file_path: str) -> Dict[str, Any]:
    if not _grobid_is_alive():
        raise GrobidUnavailableError(f"GROBID not reachable at {GROBID_URL}")

    with open(file_path, "rb") as f:
        response = requests.post(
            f"{GROBID_URL}/api/processFulltextDocument",
            files={"input": (Path(file_path).name, f, "application/pdf")},
            data={
                "consolidateHeader":      "0",
                "consolidateCitations":   "0",
                "includeRawAffiliations": "1",
                "teiCoordinates":         "",
            },
            timeout=GROBID_TIMEOUT,
        )

    if response.status_code != 200:
        raise RuntimeError(f"GROBID HTTP {response.status_code}: {response.text[:200]}")

    return _parse_grobid_tei(response.text, file_path)


def _parse_grobid_tei(xml: str, file_path: str) -> Dict[str, Any]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(xml, "lxml-xml")

    title    = _grobid_title(soup, file_path)
    authors  = _grobid_authors(soup)
    abstract = _grobid_abstract(soup)
    sections = _grobid_sections(soup)           # FIX 1
    tables_and_figures = _grobid_figures_with_images(soup, file_path)  # FIX 2

    full_text_parts = [title, abstract] + [
        f"{s['title']}\n{s['content']}" for s in sections
    ]
    full_text = _clean("\n\n".join(p for p in full_text_parts if p))

    return {
        "title":              title,
        "authors":            authors,
        "abstract":           abstract,
        "sections":           sections,
        "tables_and_figures": tables_and_figures,
        "full_text":          full_text,
    }


def _grobid_title(soup, file_path: str) -> str:
    t = soup.find("titleStmt")
    if t:
        tag = t.find("title", {"type": "main"}) or t.find("title")
        if tag and tag.get_text(strip=True):
            return _clean_inline(tag.get_text(" ", strip=True))[:400]
    fd = soup.find("fileDesc")
    if fd:
        tag = fd.find("title")
        if tag and tag.get_text(strip=True):
            return _clean_inline(tag.get_text(" ", strip=True))[:400]
    return Path(file_path).stem.replace("_", " ").title()


def _grobid_authors(soup) -> List[str]:
    authors = []
    for author in soup.find_all("author"):
        persname = author.find("persName")
        if not persname:
            continue
        forename = persname.find("forename")
        surname  = persname.find("surname")
        if surname:
            first = forename.get_text(strip=True) if forename else ""
            last  = surname.get_text(strip=True)
            name  = f"{first} {last}".strip() if first else last
            if name and len(name) > 1:
                authors.append(name)
    return _dedup(authors)[:20]


def _grobid_abstract(soup) -> str:
    abstract_tag = soup.find("abstract")
    if abstract_tag:
        paras = abstract_tag.find_all("p")
        text  = " ".join(p.get_text(" ", strip=True) for p in paras) if paras \
                else abstract_tag.get_text(" ", strip=True)
        text  = re.sub(r"\s+", " ", text).strip()
        if len(text) > 50:
            return _clean(text[:3000])
    return ""


# ── FIX 1: GROBID section hierarchy ───────────────────────────────────────────

def _heading_depth(heading: str) -> int:
    """
    Return numeric depth from section numbering in heading text.
    "1 Introduction"    → 1
    "2.1 Background"    → 2
    "3.1.2 Details"     → 3
    "A Related Work"    → 1
    "A.1 Subsection"    → 2
    Unnumbered heading  → 1
    """
    # Dotted numeric prefix: "2.1", "3.1.2"
    m = re.match(r"^(\d+(?:\.\d+)+)\s", heading)
    if m:
        return len(m.group(1).split("."))
    # Dotted alpha-numeric prefix: "A.1", "B.2.1"
    m = re.match(r"^([A-Z](?:\.\d+)+)\s", heading)
    if m:
        return len(m.group(1).split("."))
    # Single numeric prefix only: "2 Background" → top-level (depth 1)
    return 1


def _grobid_sections(soup) -> List[Dict]:
    """
    Build hierarchical section list from TEI <div> blocks.

    Depth strategy (in priority order):
      1. GROBID's  <head n="2.1">  attribute — most reliable, set by GROBID
         directly from the PDF's section numbering.
      2. Dotted number prefix in the heading text  ("2.1 Foo" → depth 2).
      3. DOM nesting: a <div> that is a descendant of another <div> that
         also has a <head> → subsection.
      4. Default depth 1.

    This multi-signal approach handles papers where GROBID omits the n=
    attribute AND papers where the heading text carries no number prefix.
    """
    body = soup.find("body")
    if not body:
        return []

    def _depth_from_n(head_tag) -> Optional[int]:
        """Read depth from GROBID's n= attribute, e.g. n='2.1' → 2."""
        n = head_tag.get("n", "")
        if not n:
            return None
        # n may look like "2.1", "A.1", "3.1.2"
        parts = re.split(r"[.\-]", n.strip())
        if len(parts) > 1:
            return len(parts)
        # Single token like "2" or "A" → top-level
        return 1

    def _depth_from_text(heading: str) -> Optional[int]:
        """Infer depth from dotted prefix in heading text."""
        m = re.match(r"^(\d+(?:\.\d+)+)\s", heading)
        if m:
            return len(m.group(1).split("."))
        m = re.match(r"^([A-Z](?:\.\d+)+)\s", heading)
        if m:
            return len(m.group(1).split("."))
        return None

    # Build a set of all divs that CONTAIN another div with a head —
    # used as fallback to detect nesting when n= and text-prefix both fail.
    parent_divs: set = set()
    all_head_divs = [d for d in body.find_all("div", recursive=True)
                     if d.find("head", recursive=False)]
    for div in all_head_divs:
        for ancestor in div.parents:
            if ancestor in all_head_divs:
                parent_divs.add(id(ancestor))
                break

    flat: List[Dict] = []
    for div in all_head_divs:
        head = div.find("head", recursive=False)
        heading_raw = _clean_inline(head.get_text(" ", strip=True))
        if not heading_raw or not _is_valid_section_title(heading_raw):
            continue
        tl = heading_raw.lower().lstrip("0123456789. ")
        if any(tl.startswith(bm) for bm in BACK_MATTER):
            continue

        # Collect only direct <p> children (not paragraphs inside nested divs)
        direct_paras = [
            child.get_text(" ", strip=True)
            for child in div.children
            if hasattr(child, "name") and child.name == "p"
        ]
        content = _clean(re.sub(r"\s+", " ", " ".join(direct_paras)).strip())

        # Determine depth using the priority chain
        depth = (
            _depth_from_n(head)
            or _depth_from_text(heading_raw)
            or (2 if any(id(a) in {id(d) for d in all_head_divs}
                         for a in div.parents if a != body) else 1)
        )

        flat.append({
            "title":   heading_raw,
            "content": content[:40000],
            "depth":   depth,
        })

    # Build two-level hierarchy (top-level + one subsection layer)
    # Deep nesting (depth 3+) is collapsed into the nearest depth-2 parent.
    sections: List[Dict] = []
    current_top: Optional[Dict] = None
    current_sub: Optional[Dict] = None   # for depth-3 collapse

    for item in flat:
        if item["depth"] == 1:
            current_sub = None
            current_top = {
                "title":       item["title"],
                "content":     item["content"],
                "level":       1,
                "subsections": [],
            }
            sections.append(current_top)
        elif item["depth"] == 2:
            current_sub = {
                "title":   item["title"],
                "content": item["content"],
                "level":   2,
                "subsections": [],
            }
            if current_top is None:
                current_top = {
                    "title": "Body", "content": "",
                    "level": 1, "subsections": [],
                }
                sections.append(current_top)
            current_top["subsections"].append(current_sub)
        else:
            # depth ≥ 3: collapse into current depth-2 if available, else depth-1
            target = current_sub or current_top
            if target is None:
                current_top = {
                    "title": "Body", "content": "",
                    "level": 1, "subsections": [],
                }
                sections.append(current_top)
                target = current_top
            # Append content to parent rather than creating another nesting level
            sep = "\n\n" if target["content"] else ""
            target["content"] = (
                target["content"] + sep
                + f"[{item['title']}] " + item["content"]
            )[:40000]

    return [
        s for s in sections
        if _is_valid_section_title(s["title"])
        and (s["content"].strip() or s["subsections"])
    ][:40]


# ── FIX 2: figure/table image cropping ────────────────────────────────────────

def _grobid_figures_with_images(soup, file_path: str) -> List[Dict]:
    """
    Extract figure/table captions from TEI and crop matching page regions
    from the PDF using pdfplumber, returning base64-encoded PNG images.
    """
    results: List[Dict] = []
    seen: set = set()

    cap_re = re.compile(
        r"^(Figure|Fig\.?|Table|Algorithm)\s*\.?\s*(\d+[a-zA-Z]?)\s*[:.]\s*(.{0,400})",
        re.IGNORECASE,
    )

    try:
        pdf       = pdfplumber.open(file_path)
        pdf_pages = pdf.pages
    except Exception:
        pdf       = None
        pdf_pages = []

    for fig_tag in soup.find_all("figure"):
        head_tag = fig_tag.find("head")
        desc_tag = fig_tag.find("figDesc")

        label_text = (head_tag.get_text(" ", strip=True) if head_tag else "") or \
                     (desc_tag.get_text(" ", strip=True)[:80] if desc_tag else "")

        m = cap_re.match(label_text)
        if not m and desc_tag:
            m = cap_re.match(desc_tag.get_text(" ", strip=True))
        if not m:
            continue

        kind_raw = m.group(1).lower()
        num      = m.group(2)
        cap_text = re.sub(r"\s+", " ", m.group(3) or "").strip()
        if not cap_text and desc_tag:
            cap_text = re.sub(r"\s+", " ", desc_tag.get_text(" ", strip=True)).strip()[:300]

        kind  = "figure" if "fig" in kind_raw else ("table" if "table" in kind_raw else "algorithm")
        label = f"{'Figure' if 'fig' in kind_raw else kind_raw.title()} {num}"
        lk    = label.lower().replace(" ", "")

        if lk in seen:
            continue
        seen.add(lk)

        item: Dict[str, Any] = {"type": kind, "label": label, "caption": cap_text[:300]}

        if pdf is not None:
            img_b64 = _crop_figure_image(pdf_pages, label, kind)
            if img_b64:
                item["image_b64"] = img_b64

        results.append(item)
        if len(results) >= 25:
            break

    if pdf is not None:
        try: pdf.close()
        except Exception: pass

    return results


def _crop_figure_image(pages, label: str, kind: str) -> Optional[str]:
    """
    Find the caption line on a PDF page and crop the nearby region.
    Returns base64 PNG or None.
    """
    search_text = label.lower()   # e.g. "figure 1" or "table 2"

    for page in pages:
        try:
            words = page.extract_words(x_tolerance=3, y_tolerance=3) or []
        except Exception:
            continue

        page_text_lower = " ".join(w["text"] for w in words).lower()
        if search_text not in page_text_lower:
            continue

        # Find Y of the label word on this page
        caption_y = None
        label_parts = search_text.split()          # ["figure", "1"]
        for i, w in enumerate(words):
            if w["text"].lower() == label_parts[0]:
                # Check the next word matches the number
                if i + 1 < len(words) and words[i+1]["text"].lower() == label_parts[1]:
                    caption_y = w["top"]
                    break

        if caption_y is None:
            continue

        page_h = float(page.height)
        page_w = float(page.width)

        try:
            if kind == "figure":
                # Region above the caption label = the figure itself
                crop_top    = max(0.0, caption_y - page_h * 0.38)
                crop_bottom = max(0.0, caption_y - 4)
            else:
                # Region below the caption label = the table rows
                crop_top    = caption_y + 10
                crop_bottom = min(page_h, caption_y + page_h * 0.35)

            if crop_bottom - crop_top < 15:
                continue

            cropped = page.crop((0, crop_top, page_w, crop_bottom))
            pil_img = cropped.to_image(resolution=FIGURE_DPI).original
            buf     = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=True)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            continue

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# pdfplumber fallback path
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pdfplumber_linear(file_path: str) -> Dict[str, Any]:
    with pdfplumber.open(file_path) as pdf:
        all_lines: List[Dict] = []
        for page in pdf.pages:
            all_lines.extend(_page_to_lines_simple(page))

    full_text  = _clean("\n".join(l["text"] for l in all_lines))
    body_size, max_size = _size_stats(all_lines)
    p0_lines   = [l for l in all_lines if l.get("page", 0) == 0]
    tables_and_figures = _captions_with_images_pdfplumber(all_lines, file_path)

    return {
        "title":              _title_linear(p0_lines, full_text, max_size, file_path),
        "authors":            _authors_linear(p0_lines, full_text, max_size),
        "abstract":           _abstract_linear(all_lines, full_text),
        "sections":           _sections_linear(all_lines, body_size, max_size),
        "tables_and_figures": tables_and_figures,
        "full_text":          full_text,
    }


def _captions_with_images_pdfplumber(lines: List[Dict], file_path: str) -> List[Dict]:
    """Caption detection + image cropping for the pdfplumber fallback path."""
    cap_re = re.compile(
        r"^(Figure|Fig\.?|Table|Algorithm)\s*\.?\s*(\d+[a-zA-Z]?)\s*[:.]?\s*(.{0,400})",
        re.IGNORECASE,
    )
    caption_items: List[Dict] = []
    seen: set = set()

    for l in lines:
        m = cap_re.match(l["text"].strip())
        if not m:
            continue
        kind_raw = m.group(1).lower()
        num      = m.group(2)
        cap_text = re.sub(r"\s+", " ", m.group(3)).strip()
        kind     = "figure" if "fig" in kind_raw else ("table" if "table" in kind_raw else "algorithm")
        label    = f"{'Figure' if 'fig' in kind_raw else kind_raw.title()} {num}"
        lk       = label.lower().replace(" ", "")
        if lk in seen:
            continue
        seen.add(lk)
        caption_items.append({
            "type": kind, "label": label, "caption": cap_text[:300],
            "page": l.get("page", 0), "y": l.get("y", 0),
        })
        if len(caption_items) >= 25:
            break

    if not caption_items:
        return []

    try:
        pdf   = pdfplumber.open(file_path)
        pages = pdf.pages
    except Exception:
        return [{"type": c["type"], "label": c["label"], "caption": c["caption"]}
                for c in caption_items]

    results: List[Dict] = []
    for cap in caption_items:
        item: Dict[str, Any] = {"type": cap["type"], "label": cap["label"], "caption": cap["caption"]}
        try:
            page_idx = cap["page"]
            if 0 <= page_idx < len(pages):
                page   = pages[page_idx]
                cap_y  = float(cap["y"])
                page_h = float(page.height)
                page_w = float(page.width)

                if cap["type"] == "figure":
                    crop_top    = max(0.0, cap_y - page_h * 0.35)
                    crop_bottom = max(0.0, cap_y - 4)
                else:
                    crop_top    = cap_y + 10
                    crop_bottom = min(page_h, cap_y + page_h * 0.30)

                if crop_bottom - crop_top > 15:
                    cropped = page.crop((0, crop_top, page_w, crop_bottom))
                    pil_img = cropped.to_image(resolution=FIGURE_DPI).original
                    buf     = io.BytesIO()
                    pil_img.save(buf, format="PNG", optimize=True)
                    item["image_b64"] = base64.b64encode(buf.getvalue()).decode()
        except Exception:
            pass
        results.append(item)

    try: pdf.close()
    except Exception: pass
    return results


def _page_to_lines_simple(page) -> List[Dict]:
    try:
        words = page.extract_words(x_tolerance=3, y_tolerance=3,
                                   extra_attrs=["fontname", "size"])
    except Exception:
        return []
    if not words:
        return []

    buckets: Dict[int, List] = defaultdict(list)
    for w in words:
        buckets[int(round(w["top"] / 3) * 3)].append(w)

    lines = []
    for y in sorted(buckets):
        lw   = sorted(buckets[y], key=lambda w: w["x0"])
        text = _join_words(lw)
        if not text.strip():
            continue
        sz   = sum(w["size"] for w in lw) / len(lw)
        bold = any("Bold" in w["fontname"] or "bold" in w["fontname"]
                   or "Medi" in w["fontname"] for w in lw)
        lines.append({
            "text": text.strip(), "size": round(sz, 1), "bold": bold,
            "x": sum(w["x0"] for w in lw) / len(lw),
            "y": y, "font": lw[0]["fontname"],
            "page": getattr(page, "page_number", 0) - 1,
        })
    return lines


def _title_linear(p0_lines, full_text, max_size, file_path) -> str:
    cands = [l for l in p0_lines if len(l["text"].strip()) >= 4
             and not _is_venue(l["text"]) and l["x"] > 40]
    if not cands:
        return Path(file_path).stem.replace("_", " ").title()
    page_max = max(l["size"] for l in cands)
    thr = page_max - 3.0
    lines, last_y = [], -1
    for l in sorted(cands, key=lambda l: l["y"]):
        txt, sz = l["text"].strip(), l["size"]
        if sz < thr:
            if lines: break
            continue
        if any(re.search(p, txt.lower()) for p in [
            r"abstract", r"introduction", r"@", r"university", r"proceedings",
        ]):
            if lines: break
            continue
        if lines and (l["y"] - last_y) > 45: break
        lines.append(txt)
        last_y = l["y"]
        if len(lines) >= 4: break
    t = re.sub(r"\s+", " ", " ".join(lines)).strip()
    t = re.sub(r"[\s*†‡⁰¹²³⁴⁵⁶⁷⁸⁹]+$", "", t).strip()
    return (t or Path(file_path).stem.replace("_", " ").title())[:400]


def _authors_linear(p0_lines, full_text, max_size) -> List[str]:
    if not p0_lines: return []
    pg_max = max((l["size"] for l in p0_lines if l["x"] > 40), default=12)
    thr    = pg_max - 3.0
    found_title = False
    zone, aff_count = [], 0
    for l in sorted(p0_lines, key=lambda l: l["y"]):
        txt, sz = l["text"].strip(), l["size"]
        if not txt or _is_venue(txt) or l["x"] < 40: continue
        if re.match(r"^abstract\b", txt, re.IGNORECASE): break
        if sz >= thr: found_title = True; continue
        if not found_title or sz < 7: continue
        if any(kw in txt.lower() for kw in AFFIL_KW):
            aff_count += 1
            if aff_count >= 1 and zone: break
            continue
        if len(txt) > 150: break
        zone.append(txt)
        if len(zone) >= 5: break
    return _dedup(_parse_names(" ".join(zone)))[:20]


def _abstract_linear(lines, full_text) -> str:
    in_abs, parts = False, []
    for l in lines:
        txt = l["text"].strip()
        if not txt: continue
        if re.match(r"^abstract\s*$", txt, re.IGNORECASE):
            in_abs = True; continue
        if in_abs:
            if _is_section_heading(txt): break
            if _is_venue(txt): continue
            parts.append(txt)
    result = " ".join(parts)
    if len(result) > 100:
        return _clean(re.sub(r"\s+", " ", result))
    return _abstract_regex(full_text)


# ── FIX 1: pdfplumber section hierarchy ───────────────────────────────────────

def _sections_linear(lines, body_size, max_size) -> List[Dict]:
    """
    Build hierarchical section list from pdfplumber lines.
    Numbered subsections (2.1, 3.2.1 etc.) are nested under their parent.
    """
    sections: List[Dict] = []
    current_section: Optional[Dict] = None
    current_sub: Optional[Dict] = None

    def _flush_sub():
        nonlocal current_sub
        if current_sub and current_section is not None:
            content = _clean(current_sub.get("_buf", "").strip())
            if content:
                current_section["subsections"].append({
                    "title":   current_sub["title"],
                    "content": content[:40000],
                })
        current_sub = None

    def _flush_section():
        nonlocal current_section
        if current_section and _is_valid_section_title(current_section["title"]):
            body  = _clean(current_section.get("_buf", "").strip())
            total = body + "".join(s["content"] for s in current_section["subsections"])
            if len(total) > 30:
                current_section["content"] = body[:40000]
                del current_section["_buf"]
                sections.append(current_section)
        current_section = None

    current_section = {
        "title": "Introduction", "content": "", "_buf": "",
        "level": 1, "subsections": [],
    }

    for l in lines:
        txt = l["text"].strip()
        if not txt:
            continue

        ratio      = l["size"] / body_size if body_size > 0 else 1.0
        is_heading = (l["bold"] or ratio >= 1.08) and _is_section_heading(txt)

        if not is_heading:
            if current_sub is not None:
                current_sub["_buf"] = current_sub.get("_buf", "") + " " + txt
            elif current_section is not None:
                current_section["_buf"] = current_section.get("_buf", "") + " " + txt
            continue

        tl = txt.lower().lstrip("0123456789. ")
        if any(tl.startswith(bm) for bm in BACK_MATTER):
            _flush_sub()
            _flush_section()
            break

        # Detect depth: "2.1 Foo" has a dot in the number → subsection
        is_subsection = bool(re.match(r"^\d+\.\d+", txt)) or \
                        bool(re.match(r"^[A-Z]\.\d+", txt))

        if is_subsection:
            _flush_sub()
            current_sub = {"title": txt, "_buf": ""}
        else:
            _flush_sub()
            _flush_section()
            current_section = {
                "title": txt, "content": "", "_buf": "",
                "level": 1, "subsections": [],
            }

    _flush_sub()
    _flush_section()

    return [s for s in sections if _is_valid_section_title(s["title"])][:40]


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _abstract_regex(full_text: str) -> str:
    for pat in [
        r"[Aa]bstract\s*[\n\r]+\s*(.*?)(?=\n\s*(?:\d+[\.\\s]|[Ii]ntroduction|[Kk]eywords|[Ii]ndex [Tt]erms))",
        r"ABSTRACT\s*[\n\r]+\s*(.*?)(?=\n\s*(?:\d+|INTRODUCTION|KEYWORDS))",
        r"[Aa]bstract[:.]?\s*(.*?)(?=\n\s*(?:\d+[\.\\s]|introduction|keywords))",
    ]:
        m = re.search(pat, full_text, re.DOTALL | re.IGNORECASE)
        if m:
            a = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(a) > 100:
                return a
    return ""


def _is_section_heading(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 120 or len(t.split()) > 14:
        return False
    tl = t.lower().lstrip("0123456789. ")
    for w in KNOWN_SECTIONS:
        if tl.startswith(w):
            return True
    if re.match(r"^(\d+(\.\d+)*|[A-Z](\.\d+)*)\.?\s+[A-Z]", t) and len(t.split()) <= 10:
        return True
    if t.isupper() and 2 <= len(t.split()) <= 6 and len(t) > 3:
        return True
    return False


def _join_words(words: List) -> str:
    if not words: return ""
    parts = [words[0]["text"]]
    for i in range(1, len(words)):
        prev, curr = words[i-1], words[i]
        x1_prev = prev.get("x1", prev["x0"] + prev["size"] * len(prev["text"]) * 0.5)
        gap = curr["x0"] - x1_prev
        if gap > prev["size"] * 0.18 or curr["fontname"] != prev["fontname"]:
            parts.append(" ")
        parts.append(curr["text"])
    return _fix_text("".join(parts))


def _fix_text(t: str) -> str:
    t = re.sub(r"([a-zA-Z])- *\n([a-z])", r"\1\2", t)
    t = re.sub(r"([a-z])([A-Z])", r"\1 \2", t)
    return t


def _size_stats(lines: List[Dict]) -> Tuple[float, float]:
    if not lines: return 10.0, 16.0
    sizes = Counter(l["size"] for l in lines if "size" in l)
    if not sizes: return 10.0, 16.0
    return sizes.most_common(1)[0][0], max(l["size"] for l in lines if "size" in l)


def _clean(text: str) -> str:
    if not text: return ""
    subs = [
        (r"Proceedings of[^\n]{0,200}", ""),
        (r"Association for Computational Linguistics[^\n]*", ""),
        (r"Annual Meeting of[^\n]{0,150}", ""),
        (r"c\s*[©\u00a9]\s*20\d\d[^\n]*", ""),
        (r"arXiv:\S+\s*\[[^\]]+\][^\n]*", ""),
        (r"https?://doi\.org/\S+", ""),
        (r"(?:Received|Accepted|Published)[^\n]{0,80}", ""),
        (r"\S+@\S+\.\S+", ""),
        (r"\n{3,}", "\n\n"),
    ]
    for pat, repl in subs:
        try: text = re.sub(pat, repl, text, flags=re.MULTILINE | re.IGNORECASE)
        except: pass
    return text.strip()


def _clean_inline(text: str) -> str:
    text = re.sub(r"[†‡*∗¶§⋆⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _is_venue(text: str) -> bool:
    t = text.strip().lower()
    return any(re.search(p, t) for p in VENUE_RE)


def _is_valid_section_title(title: str) -> bool:
    t = (title or "").strip()
    if not t or len(t) < 2: return False
    if re.match(r"^abstract\s*$", t, re.IGNORECASE): return False
    if re.match(r"^(Box|P\.?O\.?\s*Box)\s", t, re.IGNORECASE): return False
    if re.match(r"^\d{5}", t): return False
    if re.search(r"https?://|\bURL\b", t): return False
    if re.match(r"^references?\s*$", t, re.IGNORECASE): return False
    return True


def _dedup(items):
    seen, out = set(), []
    for x in items:
        k = x.lower().replace(" ", "")
        if k not in seen: seen.add(k); out.append(x)
    return out


def _parse_names(text: str) -> List[str]:
    text = re.sub(r"[†‡*∗¶§⋆⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", text)
    text = re.sub(r"(?<=[a-zA-ZÀ-ÿ])\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"[,;]|\band\b|\n", text, flags=re.IGNORECASE)
    authors = []
    for part in parts:
        name = re.sub(r"\s+", " ", part.strip().strip(".,;()")).strip()
        if not name or len(name) < 3: continue
        wds = name.split()
        if not (1 <= len(wds) <= 5): continue
        ok = all(
            bool(re.match(r"^[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝŁŠŽŒ]", w))
            or w in ("-", "van", "de", "von", "der", "bin", "el")
            for w in wds if len(w) > 1
        )
        if not ok: continue
        if any(kw in name.lower() for kw in [
            "university", "institute", "laboratory", "department", "research",
            "center", "school", "college", "lab",
        ]): continue
        authors.append(name)
    return authors


def _fallback(file_path: str) -> Dict[str, Any]:
    try:
        with pdfplumber.open(file_path) as pdf:
            text = _fix_text("\n".join(p.extract_text() or "" for p in pdf.pages))
    except Exception:
        text = ""
    return {
        "title":   Path(file_path).stem.replace("_", " ").title(),
        "authors": [],
        "abstract": text[:800],
        "sections": [{"title": "Full Text", "content": _clean(text)[:40000],
                      "level": 1, "subsections": []}],
        "tables_and_figures": [],
        "full_text": text,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Chunking (public API — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def chunk_text(text, paper_id, section_name="body", chunk_size=800, overlap=100):
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur, idx = [], "", 0
    for s in sents:
        if len(cur) + len(s) <= chunk_size:
            cur += (" " if cur else "") + s
        else:
            if cur:
                chunks.append({"paper_id": paper_id, "section": section_name,
                                "chunk_index": idx, "text": cur.strip()})
                idx += 1
                cur = " ".join(cur.split()[-max(1, overlap//5):]) + " " + s
            else:
                cur = s
    if cur.strip():
        chunks.append({"paper_id": paper_id, "section": section_name,
                        "chunk_index": idx, "text": cur.strip()})
    return chunks


def create_chunks_from_sections(sections, paper_id):
    chunks = []
    for s in sections:
        chunks.extend(chunk_text(s["content"], paper_id, s["title"]))
        for sub in s.get("subsections", []):
            chunks.extend(chunk_text(sub["content"], paper_id,
                                     f"{s['title']} > {sub['title']}"))
    return chunks