"""Syllabus verification for StudyMate.

The student loads the official VTU syllabus PDF for a semester once. For every selected
subject the app then:
  * searches the PDF for the subject (name or course code),
  * shows the text it found for Module 1-5,
  * shows the real PDF page (with the subject / module heading highlighted) so the
    student can confirm the module is correct,
  * can fill the "Topic" column from the syllabus.

    pip install pymupdf

Detection is best-effort (PDF layouts differ), which is exactly why the page viewer and
the "Verified" tick exist: the student always has the last word.
"""
from __future__ import annotations

import hashlib
import re
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st

OFFICIAL_URL = "https://vtu.ac.in/b-e-scheme-syllabus/"
CACHE_DIR = Path("syllabus_cache")

DEFAULT_EDIT = {"Topic (optional)": "", "Difficulty": 3, "Importance": 3,
                "Hours": 2.0, "Deadline": pd.NaT, "Verified": False}

_STOP = {"for", "and", "the", "of", "to", "in", "an", "course"}
_END_OF_SUBJECT = re.compile(r"(?i)suggested\s+learning\s+resources")
_END_OF_MODULES = re.compile(r"(?i)course\s+outcomes?|question\s+paper\s+pattern|assessment\s+details|"
                             r"text\s*books?|reference\s+books?|suggested\s+learning")


def _module_re(n: int) -> re.Pattern:
    return re.compile(rf"(?i)\bmodule\s*[-\u2013\u2014:]?\s*{n}\b")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _sha(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()[:16]


# ----------------------------------------------------------------------------
# Loading the PDF (upload, link, or previously saved copy)
# ----------------------------------------------------------------------------
def _make_entry(name: str, raw: bytes) -> dict:
    import fitz  # pymupdf

    if not raw.startswith(b"%PDF"):
        raise ValueError("That file is not a PDF.")
    with fitz.open(stream=raw, filetype="pdf") as doc:
        pages = doc.page_count
    return {"name": name, "bytes": raw, "id": _sha(raw), "pages": pages}


def _path_for(ck: str) -> Path:
    return CACHE_DIR / (re.sub(r"[^A-Za-z0-9]+", "_", ck).strip("_") + ".pdf")


def _save(ck: str, name: str, raw: bytes) -> None:
    entry = _make_entry(name, raw)
    CACHE_DIR.mkdir(exist_ok=True)
    _path_for(ck).write_bytes(raw)
    st.session_state.setdefault("syllabus", {})[ck] = entry


def render_loader(ck: str, label: str) -> dict | None:
    """Upload / link box. Returns the loaded syllabus entry or None."""
    try:
        import fitz  # noqa: F401
    except ImportError:
        st.error("Syllabus checking needs PyMuPDF. Run:  pip install pymupdf")
        return None

    store = st.session_state.setdefault("syllabus", {})
    if ck not in store and _path_for(ck).exists():  # copy saved from an earlier session
        try:
            store[ck] = _make_entry(_path_for(ck).name, _path_for(ck).read_bytes())
        except Exception:
            pass
    data = store.get(ck)

    st.markdown(f"#### 📄 Syllabus check · {label}")
    if data:
        st.success(f"Syllabus loaded: {data['name']} ({data['pages']} pages). Open a subject below to verify its modules.")
    else:
        st.info("Load the official VTU syllabus PDF for this semester to verify your modules against it.")

    with st.expander("Upload or link the syllabus PDF", expanded=data is None):
        st.link_button("🌐 Open VTU scheme & syllabus page", OFFICIAL_URL)
        up = st.file_uploader("Syllabus PDF", type="pdf", key=f"up|{ck}")
        if up is not None:
            raw = up.getvalue()
            if not data or data["id"] != _sha(raw):
                try:
                    _save(ck, up.name, raw)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        url = st.text_input("...or paste a direct PDF link", key=f"url|{ck}", placeholder="https://vtu.ac.in/pdf/...")
        if st.button("Download PDF", key=f"dl|{ck}") and url.strip():
            try:
                req = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=40) as resp:
                    raw = resp.read()
                _save(ck, url.strip().split("/")[-1] or "syllabus.pdf", raw)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not download the file: {exc}")
    return store.get(ck)


# ----------------------------------------------------------------------------
# Searching the PDF
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _page_texts(doc_id: str, _pdf_bytes: bytes) -> list[str]:
    import fitz

    with fitz.open(stream=_pdf_bytes, filetype="pdf") as doc:
        return [page.get_text("text") for page in doc]


@st.cache_data(show_spinner="Searching the syllabus...", max_entries=300)
def analyse_subject(doc_id: str, term: str, _pdf_bytes: bytes) -> dict | None:
    """Find a subject in the PDF and read its Module 1-5 text. None if not found."""
    texts = _page_texts(doc_id, _pdf_bytes)
    nterm = _norm(term)
    if not nterm:
        return None
    words = [w for w in nterm.split() if len(w) >= 3 and w not in _STOP]

    scores = []
    for text in texts:
        norm = _norm(text)
        if nterm in norm:
            hits = norm.count(nterm) + 2
        elif words:
            have = set(norm.split())
            frac = sum(w in have for w in words) / len(words)
            hits = frac if frac >= 0.75 else 0
        else:
            hits = 0
        scores.append(hits * 3 + (5 if hits and _module_re(1).search(text) else 0))
    if not any(scores):
        return None
    start = max(range(len(scores)), key=lambda i: (scores[i], -i))

    # text of this page and the next few, remembering where each page begins
    joined, offsets = "", []
    for i in range(start, min(start + 6, len(texts))):
        offsets.append((len(joined), i + 1))
        joined += texts[i] + "\n"

    def page_at(pos: int) -> int:
        return [pg for off, pg in offsets if off <= pos][-1]

    result = {"start_page": start + 1, "modules": {}}
    first = _module_re(1).search(joined)
    if not first:
        return result
    end_match = _END_OF_SUBJECT.search(joined, first.end())
    end = end_match.start() if end_match else len(joined)

    heads, pos = [], first.start()
    for n in range(1, 6):
        m = _module_re(n).search(joined, pos, end)
        if not m:
            break
        heads.append((n, m))
        pos = m.end()

    for i, (n, m) in enumerate(heads):
        stop = heads[i + 1][1].start() if i + 1 < len(heads) else end
        body = joined[m.end():stop]
        cut = _END_OF_MODULES.search(body)
        body = body[:cut.start()] if cut else body
        clean = " ".join(body.split())
        lead = re.sub(r"(?i)^\W*\d+\s*hours?\W*", "", clean)
        topic = re.split(r"[:.;]", lead, maxsplit=1)[0].strip()[:80]
        result["modules"][n] = {"page": page_at(m.start()), "text": clean[:600], "topic": topic}
    return result


@st.cache_data(show_spinner=False, max_entries=64)
def render_page(doc_id: str, page_no: int, terms: tuple, _pdf_bytes: bytes) -> bytes:
    """PNG of one PDF page with the given words highlighted."""
    import fitz

    with fitz.open(stream=_pdf_bytes, filetype="pdf") as doc:
        page = doc[page_no - 1]
        for term in terms:
            if term.strip():
                for rect in page.search_for(term):
                    page.add_highlight_annot(rect)
        return page.get_pixmap(dpi=110).tobytes("png")


# ----------------------------------------------------------------------------
# Per-subject checker UI
# ----------------------------------------------------------------------------
def render_subject_check(ck: str, subject: str, data: dict, edits: dict) -> None:
    key = f"{ck}|{subject}"
    term = st.text_input("Search text (subject name or course code, e.g. BCS302)", value=subject, key=f"term|{key}")
    info = analyse_subject(data["id"], term, data["bytes"])
    if not info:
        st.warning("This subject was not found in the PDF. Try the course code or a shorter name in the box above, "
                   "or check that you loaded the right semester.")
        return

    mods = info["modules"]
    st.caption(f"Found around page {info['start_page']} · {len(mods)} of 5 modules detected")
    missing = [f"Module {n}" for n in range(1, 6) if n not in mods]
    if mods and missing:
        st.warning(f"Not detected in the PDF: {', '.join(missing)}. Some subjects (labs, projects) have fewer modules "
                   "- check the page below.")
    if not mods:
        st.warning("No module headings detected. Use the page viewer below to check the syllabus by eye.")
    for n, m in mods.items():
        st.markdown(f"**Module {n}** · page {m['page']}  \n{m['text'][:300]}{'...' if len(m['text']) > 300 else ''}")

    topics = {f"Module {n}": m["topic"] for n, m in mods.items() if m["topic"]}
    if topics and st.button("✍️ Fill topic names from this syllabus", key=f"fill|{key}"):
        for label, topic in topics.items():
            edits.setdefault((subject, label), dict(DEFAULT_EDIT))["Topic (optional)"] = topic
        st.session_state["topic_apply_n"] = st.session_state.get("topic_apply_n", 0) + 1
        st.rerun()

    options = {"Subject start": info["start_page"], **{f"Module {n}": m["page"] for n, m in mods.items()}}
    pick = st.selectbox("Show in PDF", list(options), key=f"pick|{key}")
    page = st.number_input("Page", 1, data["pages"], value=int(options[pick]), key=f"pg|{key}|{pick}")
    terms = (term,) if pick == "Subject start" else (term, f"Module-{pick.split()[-1]}", f"Module {pick.split()[-1]}")
    st.image(render_page(data["id"], int(page), terms, data["bytes"]),
             caption=f"{data['name']} · page {int(page)} of {data['pages']}", use_container_width=True)
    st.caption("Looks right? Tick **✔ Verified** for these modules in the table below.")