"""
Verishelf - a shelf of verified answers, drawn only from what you hand it.
A Streamlit front end over the multi-agent RAG pipeline in agents/, retriever/
and document_processor/. Run locally with: streamlit run app.py
"""
import hashlib
import html
import os
import re
import tempfile
import uuid
from pathlib import Path
from types import SimpleNamespace

import streamlit as st
from PIL import Image, ImageDraw


def _make_favicon() -> Image.Image:
    """A small geometric ink-stamp mark - drawn, not an emoji, so the browser
    tab carries the same signature as the verification stamps in the app."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ink = (86, 86, 86, 255)  # #565656 - kept dark regardless of app theme so the
    # favicon stays visible against a typical (light) browser tab bar
    draw.ellipse([3, 3, size - 3, size - 3], outline=ink, width=5)
    draw.ellipse([12, 12, size - 12, size - 12], outline=ink, width=2)
    draw.line([(20, 33), (28, 42), (45, 20)], fill=ink, width=5, joint="curve")
    return img


st.set_page_config(
    page_title="Verishelf",
    page_icon=_make_favicon(),
    layout="wide",
    initial_sidebar_state="expanded",
)

# Imports that touch config.settings (and therefore require OPENROUTER_API_KEY) are
# wrapped so a missing key produces a friendly message instead of a raw traceback.
try:
    from config import constants
    from document_processor.file_handler import DocumentProcessor
    from retriever.builder import RetrieverBuilder
    from agents.workflow import AgentWorkflow
    from openai import RateLimitError, APIError
except Exception as exc:
    # Two distinct failure modes land here and shouldn't be conflated: a missing/blank
    # OPENROUTER_API_KEY (pydantic ValidationError, mentions the field by name) versus
    # any other import-time failure (e.g. a missing system library a dependency needs).
    # Only show the API-key remediation steps when the exception actually names it.
    if "OPENROUTER_API_KEY" in str(exc):
        st.error("Configuration error")
        st.markdown(
            "Verishelf can't start because **`OPENROUTER_API_KEY`** isn't set.\n\n"
            "1. Copy `.env.example` to `.env`\n"
            "2. Add your key from [openrouter.ai/keys](https://openrouter.ai/keys) (free to create)\n"
            "3. Restart the app"
        )
    else:
        st.error("Startup error")
        st.markdown(
            "Verishelf failed to start due to an unexpected error during setup "
            "(not the API key). See the technical details below."
        )
    with st.expander("Technical details", expanded=True):
        st.exception(exc)
    st.stop()

EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"],
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"],
    },
}


# --------------------------------------------------------------------------
# Cached singletons - heavy objects (embedding model, docling converter, LLM
# client) built once per server process and reused across every user session.
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the reading room (first load only)...")
def get_processor() -> DocumentProcessor:
    return DocumentProcessor()


@st.cache_resource(show_spinner="Loading the indexing model (first load only)...")
def get_retriever_builder() -> RetrieverBuilder:
    return RetrieverBuilder()


@st.cache_resource(show_spinner=False)
def get_workflow() -> AgentWorkflow:
    return AgentWorkflow()


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
defaults = {
    "messages": [],
    "retriever": None,
    "file_hashes": frozenset(),
    "session_id": f"session-{uuid.uuid4().hex}",
    "doc_source": None,
    "doc_chunk_count": 0,
    "pending_question": None,
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("uploader_nonce", 0)


# --------------------------------------------------------------------------
# Styling - an editorial/verification-ledger identity: paper, ink, a rotated
# ink-stamp for the one thing this app actually does (check claims against a
# source before you see them). IBM Plex Mono is a small nod to where the
# backend pipeline originated.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /*
      Base palette, fonts, radius, and borders are set natively in
      .streamlit/config.toml ([theme]) so every built-in Streamlit widget -
      buttons, selects, expanders, alerts, the file uploader - stays visually
      consistent automatically. The rules below only add the bespoke pieces
      config.toml can't express: the stamp, the manuscript layout, and a
      couple of hover/motion details.
    */
    /* Palette ported from the Lovable-generated design
       (github.com/Pouya-Shaterzadeh/verishelf-proof-ledger), converted from its
       OKLCH tokens to hex: paper #F8F5F0, paper-deep #F1ECE4, ink #18110C,
       graphite #5C5751, rule #CCC6BF, verified #315833, flagged #8E2C1C,
       accent #1E3A56.

       Variables are named by ROLE (--bg, --text, --hover-bg...), not by which literal
       color currently fills that role - so flipping between a light and dark theme is
       just new values here, not a hunt through every rule for "which one was the dark
       one again". --hover-* are separate/explicit so a hover state is never
       accidentally the same color as the surface it's on. */
    :root {
        --bg: #F8F5F0;
        --bg-raised: #F1ECE4;
        --text: #18110C;
        --text-soft: #5C5751;
        --verified: #315833;
        --flagged: #8E2C1C;
        --accent: #1E3A56;
        --rule: #CCC6BF;
        --hover-bg: #18110C;
        --hover-text: #F8F5F0;
    }

    /* Sidebar locked to a fixed width - no drag-resize - but collapsing is still
       allowed via Streamlit's own collapse/reopen buttons. The width rule is
       scoped to [aria-expanded="true"] only: Streamlit toggles that attribute
       itself to animate collapse, and an unconditional !important width here
       would fight that same animation (this caused the collapse button's icon
       to get stuck/duplicated when tried unconditionally). Two resize
       affordances are neutralized regardless of expanded state: the drag handle
       on the right edge (emotion "target" class eelgd2m3 in this pinned
       Streamlit version - see requirements.txt), and a second, wider invisible
       hit-area for that same handle that carries its cursor as an inline style
       rather than a class, caught here by attribute selector so it doesn't
       depend on a hashed/target class name. */
    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 452px !important;
        min-width: 452px !important;
        max-width: 452px !important;
    }
    .eelgd2m3,
    [data-testid="stSidebar"] [style*="resize"] {
        display: none !important;
        pointer-events: none !important;
    }

    [data-testid="stMainBlockContainer"] { max-width: 860px; padding-top: 2.5rem; }

    .vs-mono, .vs-eyebrow, .vs-stamp, .vs-masthead,
    .vs-rail__label, .vs-entry__qnum, .vs-entry__qtext, .vs-meta,
    .vs-ledger-row, .vs-citation, .vs-stamp-meta, .vs-kv {
        font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
    }

    /* ---- Title page (hero) ---- */
    .vs-titlepage { padding: 0.5rem 0 0.25rem; }
    .vs-masthead {
        font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--text-soft); margin-bottom: 1.4rem; font-weight: 600;
    }
    .vs-titlepage__title {
        font-family: 'Newsreader', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 3.1rem;
        letter-spacing: -0.01em;
        color: var(--text);
        margin: 0 0 0.6rem;
        line-height: 1.05;
    }
    .vs-titlepage__mission {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
        color: var(--text-soft);
        max-width: 46ch;
        line-height: 1.6;
    }

    .vs-rail { display: flex; margin: 2.4rem 0 0.75rem; border-top: 1px solid var(--rule); }
    .vs-rail__step { flex: 1; padding: 1.1rem 1.5rem 0 0; border-left: 1px solid var(--rule); padding-left: 1.5rem; }
    .vs-rail__step:first-child { border-left: none; padding-left: 0; }
    .vs-rail__label {
        display: block; font-size: 0.7rem; letter-spacing: 0.16em; text-transform: uppercase;
        color: var(--accent); margin-bottom: 0.5rem; font-weight: 600;
    }
    .vs-rail__label .vs-rail__num { color: var(--text); margin-right: 0.5rem; }
    .vs-rail__step p { margin: 0; font-size: 0.97rem; color: var(--text-soft); line-height: 1.55; }
    @media (max-width: 640px) {
        .vs-rail { flex-direction: column; }
        .vs-rail__step { border-left: none; padding-left: 0; border-top: 1px solid var(--rule); padding-top: 1rem; margin-top: 1rem; }
        .vs-rail__step:first-child { border-top: none; margin-top: 0; padding-top: 0; }
    }

    /* ---- Sidebar ---- */
    .vs-eyebrow {
        font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-soft); margin: 1.2rem 0 0.5rem; font-weight: 600;
    }
    .vs-eyebrow--lead { margin-top: 0.4rem; font-size: 0.8rem; }
    .vs-howitworks { list-style: none; margin: 0 0 1.5rem; padding: 0; }
    .vs-howitworks li {
        display: grid; grid-template-columns: 2rem 1fr; gap: 0.4rem;
        margin-bottom: 0.85rem;
    }
    .vs-howitworks .vs-num {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; letter-spacing: 0.08em;
        color: var(--accent); padding-top: 0.15rem;
    }
    .vs-howitworks span:last-child {
        font-family: 'Newsreader', serif; font-style: italic; font-size: 1.02rem;
        color: var(--text-soft); line-height: 1.55;
    }
    .vs-meta { font-size: 0.74rem; color: var(--text-soft); line-height: 1.6; }
    .vs-meta strong { color: var(--text); }
    .vs-meta a { color: var(--accent) !important; }
    .vs-kv { display: flex; justify-content: space-between; gap: 0.75rem; }
    .vs-kv + .vs-kv { margin-top: 0.2rem; }
    .vs-kv dt { color: var(--text-soft); }
    .vs-kv dd { color: var(--text); margin: 0; text-align: right; }

    /* ---- Verification ledger + citations ---- */
    .vs-ledger-row {
        display: grid; grid-template-columns: 9ch 1fr; gap: 1rem;
        padding: 0.35rem 0; border-bottom: 1px dotted var(--rule);
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; color: var(--text);
    }
    .vs-ledger-row span:first-child { color: var(--text-soft); }
    .vs-citation {
        display: flex; justify-content: space-between; gap: 1rem;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; color: var(--text);
        border-bottom: 1px dotted var(--rule); padding: 0.3rem 0;
    }
    .vs-citation .vs-citation__ref { color: var(--text-soft); margin-right: 0.4rem; }
    .vs-citation .vs-citation__score { color: var(--text-soft); white-space: nowrap; }

    /* ---- Manuscript entries (question + answer + stamp) ---- */
    div[class*="st-key-vs-entry-"] { padding: 1.5rem 0 1.7rem; border-bottom: 1px solid var(--rule); }
    div[class*="st-key-vs-entry-"]:first-of-type { padding-top: 0; }
    .vs-entry__q { margin-bottom: 1rem; }
    .vs-entry__qnum {
        font-size: 0.72rem; letter-spacing: 0.08em; color: var(--text-soft); margin-bottom: 0.35rem;
    }
    .vs-entry__qnum .vs-mark { color: var(--accent); margin-right: 0.4rem; font-size: 0.95rem; }
    .vs-entry__qtext {
        font-size: 0.82rem; letter-spacing: 0.03em; text-transform: uppercase; color: var(--text);
    }
    div[class*="st-key-vs-entry-"] [data-testid="stMarkdownContainer"] p,
    div[class*="st-key-vs-entry-"] [data-testid="stMarkdownContainer"] li {
        font-size: 1.12rem; line-height: 1.7;
    }
    div[class*="st-key-vs-entry-"] [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] p,
    div[class*="st-key-vs-entry-"] [data-testid="stExpanderDetails"] [data-testid="stMarkdownContainer"] li {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.83rem; line-height: 1.7; color: var(--text-soft);
    }

    /* ---- The stamp: the one signature element ---- */
    .vs-stamp {
        display: inline-block; position: relative;
        font-weight: 600; font-size: 0.72rem; letter-spacing: 0.14em;
        padding: 0.32rem 0.7rem; border: 1.5px solid currentColor; border-radius: 2px;
        transform: rotate(-2.5deg); margin: 0.7rem 0 0.3rem;
        animation: vs-stamp-settle 260ms ease-out;
    }
    .vs-stamp::before {
        content: ""; position: absolute; inset: 3px;
        border: 1px solid currentColor; border-radius: 1px; opacity: 0.55;
    }
    .vs-stamp--verified { color: var(--verified); }
    .vs-stamp--flagged { color: var(--flagged); }
    .vs-stamp-row { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; }
    .vs-stamp-meta {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; color: var(--text-soft);
    }
    @keyframes vs-stamp-settle {
        0% { opacity: 0; transform: rotate(-2.5deg) scale(1.5); }
        100% { opacity: 1; transform: rotate(-2.5deg) scale(1); }
    }
    @media (prefers-reduced-motion: reduce) {
        .vs-stamp { animation: none; }
    }

    /* ---- A few bespoke interaction details on top of the native theme ---- */
    [data-testid="stButton"] button {
        font-family: 'IBM Plex Mono', monospace;
        transition: background-color 120ms ease, color 120ms ease;
    }
    [data-testid="stButton"] button:hover {
        background: var(--hover-bg); color: var(--hover-text); border-color: var(--hover-bg);
    }
    [data-testid="stChatInputTextArea"] { font-size: 1.05rem; }

    /* Chat input's wrapping container has a hardcoded focus border/box-shadow with
       no theme hook. Pin it to the neutral rule color in every state (base AND
       focus-within) so clicking into the field produces no border change at all. */
    div:has(> div > textarea[data-testid="stChatInputTextArea"]),
    div:has(> div > div > textarea[data-testid="stChatInputTextArea"]),
    div:has(textarea[data-testid="stChatInputTextArea"]),
    div:has(textarea[data-testid="stChatInputTextArea"]):focus-within,
    div:has(textarea[data-testid="stChatInputTextArea"]):focus {
        border-color: var(--rule) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    [data-testid="stChatInputTextArea"]:focus,
    [data-testid="stChatInputTextArea"]:focus-visible {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Selectbox (baseweb combobox) paints the same hardcoded focus border/box-shadow.
       Pin its wrapper to the neutral rule color in every state so opening the sample
       dropdown produces no border-color change. */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div:focus-within,
    div:has(> div > input[role="combobox"]),
    div:has(> div > input[role="combobox"]):focus-within {
        border-color: var(--rule) !important;
        box-shadow: none !important;
        outline: none !important;
    }
    input[role="combobox"]:focus,
    input[role="combobox"]:focus-visible {
        box-shadow: none !important;
        outline: none !important;
    }

    /* Keep the accessible focus ring for real keyboard navigation elsewhere, but
       exclude the chat input and the combobox (both handled above) so they stay
       borderless on click. */
    *:focus-visible:not([data-testid="stChatInputTextArea"]):not(input[role="combobox"]) { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def materialize_uploads(uploaded_files):
    """Write Streamlit's in-memory UploadedFile objects to real temp paths,
    since DocumentProcessor/Docling both expect a filesystem path."""
    materialized, tmp_paths = [], []
    for uf in uploaded_files:
        suffix = Path(uf.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_paths.append(tmp.name)
        materialized.append(SimpleNamespace(name=tmp.name, display_name=uf.name))
    return materialized, tmp_paths


def index_documents(materialized_files, tmp_paths, label: str) -> bool:
    processor = get_processor()
    retriever_builder = get_retriever_builder()
    try:
        with st.spinner(f"Reading {label} - parsing, chunking, and indexing..."):
            chunks = processor.process(materialized_files)
            if not chunks:
                st.error(
                    "Couldn't find any text in that file. Make sure it's a real "
                    f"{'/'.join(constants.ALLOWED_TYPES)} file and isn't empty or corrupted."
                )
                return False
            retriever = retriever_builder.build_hybrid_retriever(
                chunks, collection_name=st.session_state.session_id
            )
    finally:
        for p in tmp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass

    st.session_state.retriever = retriever
    st.session_state.doc_chunk_count = len(chunks)
    st.session_state.doc_source = label
    return True


def supported_from_report(report: str):
    if not report:
        return None
    return "Supported:** YES" in report


_REPORT_LINE = re.compile(r"\*\*(.+?):\*\*\s*(.*)")
_LEDGER_LABELS = {
    "Supported": "Supported",
    "Unsupported claims": "Unsupported",
    "Contradictions": "Contradictions",
    "Relevant": "Relevant",
    "Additional details": "Notes",
}


def render_report(report: str, citations: list):
    """Render the verification report and its citations as compact monospace
    ledger rows instead of a wall of raw markdown bold text."""
    rows = []
    for line in report.strip().split("\n"):
        m = _REPORT_LINE.match(line.strip())
        if not m:
            continue
        raw_label, value = m.group(1).strip(), (m.group(2) or "None").strip()
        label = _LEDGER_LABELS.get(raw_label, raw_label)
        rows.append(
            f'<div class="vs-ledger-row"><span>{html.escape(label)}</span>'
            f"<span>{html.escape(value)}</span></div>"
        )
    st.markdown("".join(rows), unsafe_allow_html=True)

    if citations:
        st.markdown('<div class="vs-eyebrow">Citations</div>', unsafe_allow_html=True)
        cite_rows = []
        for i, c in enumerate(citations, start=1):
            ref = f"[{i:02d}] {html.escape(c['source'])}"
            excerpt = html.escape(c["excerpt"])
            cite_rows.append(
                '<div class="vs-citation">'
                f'<span><span class="vs-citation__ref">{ref}</span>{excerpt}&hellip;</span>'
                f'<span class="vs-citation__score">score {c["score"]:.2f}</span>'
                "</div>"
            )
        st.markdown("".join(cite_rows), unsafe_allow_html=True)


def render_entry(idx, role: str, content: str, verification: str = "", supported=None,
                  citations=None, passages_consulted=0, re_researched=False):
    with st.container(key=f"vs-entry-{idx}"):
        if role == "user":
            query_number = idx // 2 + 1
            st.markdown(
                '<div class="vs-entry__q">'
                f'<div class="vs-entry__qnum"><span class="vs-mark">&sect;</span>Query &#8470;{query_number:03d}</div>'
                f'<div class="vs-entry__qtext">{html.escape(content)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(content)
            if verification:
                stamp_class = "vs-stamp--verified" if supported else "vs-stamp--flagged"
                stamp_text = "Verified" if supported else "Flagged"
                meta_bits = [f"{passages_consulted} passages consulted"]
                if re_researched:
                    meta_bits.append("re-researched once")
                st.markdown(
                    f'<div class="vs-stamp-row"><div class="vs-stamp {stamp_class}">'
                    f'&#10022; {stamp_text}</div>'
                    f'<div class="vs-stamp-meta">{" &middot; ".join(meta_bits)}</div></div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Read the verification report"):
                    render_report(verification, citations or [])


def ask(question: str):
    idx = len(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    render_entry(idx, "user", question)

    with st.spinner("Checking relevance, researching, and verifying the answer..."):
        try:
            result = get_workflow().full_pipeline(
                question=question, retriever=st.session_state.retriever
            )
            answer = result["draft_answer"]
            verification = result.get("verification_report", "")
            citations = result.get("citations", [])
            passages_consulted = result.get("passages_consulted", 0)
            re_researched = result.get("re_researched", False)
        except RateLimitError:
            answer = (
                "OpenRouter's free-tier rate limit was hit (20 requests/minute, 50/day "
                "without added credit). Wait a bit and try again, or add $10 of OpenRouter "
                "credit to raise the daily cap to 1000."
            )
            verification, citations, passages_consulted, re_researched = "", [], 0, False
        except APIError as e:
            answer = f"The model provider returned an error: {e}"
            verification, citations, passages_consulted, re_researched = "", [], 0, False
        except Exception as e:
            answer = f"Something went wrong while answering: {e}"
            verification, citations, passages_consulted, re_researched = "", [], 0, False

    supported = supported_from_report(verification)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "verification": verification, "supported": supported,
        "citations": citations, "passages_consulted": passages_consulted, "re_researched": re_researched,
    })
    render_entry(idx + 1, "assistant", answer, verification, supported,
                 citations, passages_consulted, re_researched)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="vs-eyebrow vs-eyebrow--lead">How it works</div>
        <ol class="vs-howitworks">
            <li><span class="vs-num">01</span><span>ChromaDB stores your document as embeddings for millisecond similarity search</span></li>
            <li><span class="vs-num">02</span><span>A RelevanceChecker agent decides your question is answerable before anything else runs</span></li>
            <li><span class="vs-num">03</span><span>A ResearchAgent drafts an answer, then a VerificationAgent cross-checks it against the source</span></li>
            <li><span class="vs-num">04</span><span>LangGraph's StateGraph wires these agents together, looping back to re-research once if verification fails</span></li>
        </ol>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="vs-eyebrow">Deposit documents</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Add a document",
        type=[t.lstrip(".") for t in constants.ALLOWED_TYPES],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help=f"Accepted formats: {', '.join(constants.ALLOWED_TYPES)} · up to "
        f"{constants.MAX_TOTAL_SIZE // 1024 // 1024}MB total",
        key=f"uploader_{st.session_state.uploader_nonce}",
    )

    if uploaded_files:
        current_hashes = frozenset(hash_bytes(uf.getvalue()) for uf in uploaded_files)
        if current_hashes != st.session_state.file_hashes:
            materialized, tmp_paths = materialize_uploads(uploaded_files)
            label = ", ".join(uf.name for uf in uploaded_files)
            if index_documents(materialized, tmp_paths, label):
                st.session_state.file_hashes = current_hashes
                st.toast(f"Indexed {st.session_state.doc_chunk_count} passages from {len(uploaded_files)} file(s).")

    with st.expander("Try a sample dossier", expanded=not st.session_state.messages and not uploaded_files):
        example_name = st.selectbox(
            "Sample document", options=list(EXAMPLES.keys()), index=None, placeholder="Choose one...",
            label_visibility="collapsed",
        )
        if st.button("Open sample", disabled=example_name is None, use_container_width=True):
            ex = EXAMPLES[example_name]
            materialized = [SimpleNamespace(name=p) for p in ex["file_paths"] if os.path.exists(p)]
            if materialized and index_documents(materialized, [], example_name):
                st.session_state.file_hashes = frozenset({f"example:{example_name}"})
                st.session_state.pending_question = ex["question"]
                st.rerun()
            elif not materialized:
                st.error("Sample file not found on disk.")

    if st.session_state.doc_source:
        st.markdown('<div class="vs-eyebrow">On the shelf</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="vs-meta"><strong>{html.escape(st.session_state.doc_source)}</strong><br>'
            f"{st.session_state.doc_chunk_count} passages indexed</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr style="border-color: var(--rule); margin: 1.4rem 0 1rem;">', unsafe_allow_html=True)
    if st.button("New dossier", use_container_width=True):
        st.session_state.uploader_nonce += 1
        for key in defaults:
            st.session_state.pop(key, None)
        st.rerun()

    st.markdown(
        f"""
        <dl class="vs-meta" style="margin-top: 1.4rem;">
            <div class="vs-kv"><dt>Retriever</dt><dd>bm25 &oplus; chroma</dd></div>
            <div class="vs-kv"><dt>Rate</dt><dd>20/min &middot; 50/day</dd></div>
        </dl>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="vs-titlepage">
            <div class="vs-masthead">{constants.APP_MASTHEAD}</div>
            <h1 class="vs-titlepage__title">{constants.APP_NAME}</h1>
            <p class="vs-titlepage__mission">{constants.APP_TAGLINE}</p>
        </div>
        <div class="vs-rail">
            <div class="vs-rail__step">
                <span class="vs-rail__label"><span class="vs-rail__num">01</span>Upload</span>
                <p>ChromaDB indexes your file as embeddings, ready for hybrid BM25 and vector search.</p>
            </div>
            <div class="vs-rail__step">
                <span class="vs-rail__label"><span class="vs-rail__num">02</span>Ask</span>
                <p>A RelevanceChecker agent confirms you're in scope, then a ResearchAgent drafts an answer from what's retrieved.</p>
            </div>
            <div class="vs-rail__step">
                <span class="vs-rail__label"><span class="vs-rail__num">03</span>Verify</span>
                <p>A VerificationAgent cross-checks the draft against the source, re-researching once if it doesn't hold up.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for i, msg in enumerate(st.session_state.messages):
    render_entry(
        i, msg["role"], msg["content"], msg.get("verification", ""), msg.get("supported"),
        msg.get("citations", []), msg.get("passages_consulted", 0), msg.get("re_researched", False),
    )

if st.session_state.retriever:
    st.markdown(
        '<div class="vs-kv" style="max-width: 860px; margin: 0 auto;">'
        '<dt>Answers are drafted from your corpus only.</dt><dd>20 req/min &middot; 50/day</dd>'
        "</div>",
        unsafe_allow_html=True,
    )

pending = st.session_state.pop("pending_question", None)
question = st.chat_input(
    "Pose a query about your document(s)..." if st.session_state.retriever else "Add a document to begin...",
    disabled=st.session_state.retriever is None,
)
if pending:
    question = pending

if question:
    ask(question)
