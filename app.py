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

# Imports wrapped so any setup failure (e.g. a missing system library) shows a friendly
# message instead of a raw traceback. There is no server-side API key: each visitor
# brings their own OpenAI key at runtime, so nothing here depends on a configured key.
try:
    from config import constants
    from config.settings import settings
    from document_processor.file_handler import DocumentProcessor
    from retriever.builder import RetrieverBuilder
    from agents.workflow import AgentWorkflow
    from agents.llm_client import make_client, validate_key, pick_default_model
    from openai import RateLimitError, APIError
except Exception as exc:
    st.error("Startup error")
    st.markdown(
        "Verishelf failed to start due to an unexpected error during setup. "
        "See the technical details below."
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
def get_workflow(api_key: str, model: str) -> AgentWorkflow:
    """One multi-agent workflow per (key, model). Cached so the LangGraph isn't rebuilt
    on every message; keyed by api_key so each visitor's workflow uses their own key.
    The key lives only in this cached object in server memory - never persisted."""
    return AgentWorkflow(make_client(api_key), model)


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
    # Bring-your-own-key: the visitor's OpenAI key + validation state, session-only.
    "api_key": "",
    "key_ok": False,
    "key_msg": "",
    "models": [],  # discovered from the user's key at runtime
    "model": "",   # set from the user's models once a key is validated
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

    /* Base font size bumped from the 16px default so every rem-based size in the app
       scales up proportionally to a more comfortable reading size, without having to
       touch each element. */
    html { font-size: 17.5px; }

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
    /* The 452px lock is desktop-only. This pinned Streamlit version does NOT turn
       the sidebar into an overlay on small screens (verified against vanilla
       Streamlit) - the expanded sidebar stays in the flex row and shoves the main
       column off-screen. So below the breakpoint we take over and make the sidebar
       a fixed overlay ourselves (see the max-width: 640px block further down). */
    @media (min-width: 641px) {
        [data-testid="stSidebar"][aria-expanded="true"] {
            width: 452px !important;
            min-width: 452px !important;
            max-width: 452px !important;
        }
    }
    .eelgd2m3,
    [data-testid="stSidebar"] [style*="resize"] {
        display: none !important;
        pointer-events: none !important;
    }

    /* Mobile: turn the sidebar into a fixed overlay drawer. position:fixed pulls it
       out of the flex row so the main column reclaims full width (otherwise it's
       squeezed to a sliver and its text runs off-screen). The high z-index is
       already set by Streamlit (sidebarMobile). Width is a comfortable share of the
       viewport, capped so it never gets absurdly wide on a large phone/tablet. */
    @media (max-width: 640px) {
        [data-testid="stSidebar"][aria-expanded="true"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100% !important;
            width: 85vw !important;
            max-width: 22rem !important;
            box-shadow: 0 0 0 100vmax rgba(24, 17, 12, 0.35);
        }
        /* Collapsed: fully off-canvas so it never contributes layout width. */
        [data-testid="stSidebar"][aria-expanded="false"] {
            position: fixed !important;
            width: 0 !important;
            min-width: 0 !important;
        }
    }

    /* Pull the sidebar content up by trimming Streamlit's default header/top padding,
       so "How it works" sits near the top. */
    [data-testid="stSidebarHeader"] { padding: 0.3rem 0.5rem 0 !important; min-height: 0 !important; height: 2.2rem !important; }
    [data-testid="stSidebarUserContent"] { padding-top: 0 !important; padding-bottom: 1rem !important; }

    /* Hide the sidebar scrollbar but keep it scrollable. With the compacted spacing the
       content fits without a bar on a normal-height screen; on a shorter one it can
       still be scrolled (no visible bar, and nothing gets clipped off-screen). */
    [data-testid="stSidebar"] *::-webkit-scrollbar { width: 0 !important; height: 0 !important; display: none !important; }
    [data-testid="stSidebar"] * { scrollbar-width: none !important; }

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
        /* Fluid instead of a fixed 3.1rem - scales down on phones without a
           separate breakpoint, never gets larger than the desktop size. */
        font-size: clamp(2rem, 6vw + 1rem, 3.1rem);
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

        /* Citation rows and footer key/value pairs go from a tight side-by-side
           layout to stacked - at 452px sidebar width these already have room, but
           the main-column citations under each answer don't below ~640px. */
        .vs-citation, .vs-kv { flex-direction: column; gap: 0.15rem; }
        .vs-citation .vs-citation__score, .vs-kv dd { text-align: left; }

        /* Enough top padding to clear Streamlit's floating top header (collapse +
           menu), so the masthead isn't tucked underneath it. */
        [data-testid="stMainBlockContainer"] { padding-top: 3rem; }

        /* Tighten the masthead so the whole line fits a narrow screen and wraps
           cleanly instead of being clipped. */
        .vs-masthead { font-size: 0.6rem; letter-spacing: 0.07em; overflow-wrap: anywhere; }
    }

    /* ---- Sidebar ---- */
    .vs-eyebrow {
        font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
        color: var(--text-soft); margin: 1.2rem 0 0.5rem; font-weight: 600;
    }
    .vs-eyebrow--lead { margin-top: 0; font-size: 0.8rem; }
    .vs-howitworks { list-style: none; margin: 0 0 0.9rem; padding: 0; }
    .vs-howitworks li {
        display: grid; grid-template-columns: 2rem 1fr; gap: 0.4rem;
        margin-bottom: 0.5rem;
    }
    .vs-howitworks .vs-num {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; letter-spacing: 0.08em;
        color: var(--accent); padding-top: 0.1rem;
    }
    .vs-howitworks span:last-child {
        font-family: 'Newsreader', serif; font-style: italic; font-size: 0.98rem;
        color: var(--text-soft); line-height: 1.4;
    }
    .vs-meta { font-size: 0.74rem; color: var(--text-soft); line-height: 1.6; }
    .vs-meta strong { color: var(--text); }
    .vs-meta a { color: var(--accent) !important; }
    .vs-kv { display: flex; justify-content: space-between; gap: 0.75rem; }
    .vs-kv + .vs-kv { margin-top: 0.2rem; }
    .vs-kv dt { color: var(--text-soft); }
    .vs-kv dd { color: var(--text); margin: 0; text-align: right; }

    /* Bring-your-own-key gate */
    .vs-key { font-size: 0.8rem; font-weight: 600; margin: 0.5rem 0 0.15rem; letter-spacing: 0.01em; }
    .vs-key--ok { color: var(--verified); }
    .vs-key--bad { color: var(--flagged); }
    .vs-key-hint { font-size: 0.72rem; color: var(--text-soft); margin: 0.45rem 0 0; }
    .vs-key-hint kbd {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--text);
        background: var(--bg-raised); border: 1px solid var(--rule); border-radius: 3px;
        padding: 0.05rem 0.32rem; margin: 0 0.05rem;
    }
    .vs-key-note { font-size: 0.68rem; color: var(--text-soft); line-height: 1.5; margin-top: 0.35rem; }
    .vs-key-note a { color: var(--accent) !important; white-space: nowrap; }

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

    /* Text inputs (the OpenAI key field) and the selectbox/combobox use Streamlit's
       newer emotion/react-aria structure (no longer data-baseweb), whose default focus
       state paints a harsh dark border. Replace it with a smooth, rounded accent focus:
       a soft border + gentle glow, never a black ring. Targeted by stable testid/role. */
    [data-testid="stTextInputRootElement"],
    div[role="group"]:has(> input[role="combobox"]) {
        border-radius: 7px !important;
        transition: border-color 130ms ease, box-shadow 130ms ease;
    }
    [data-testid="stTextInputRootElement"]:focus-within,
    div[role="group"]:has(> input[role="combobox"]):focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 16%, transparent) !important;
        outline: none !important;
    }
    /* The inner <input> must not add its own outline/ring on top of the container's. */
    [data-testid="stTextInputRootElement"] input:focus,
    [data-testid="stTextInputRootElement"] input:focus-visible,
    input[role="combobox"]:focus,
    input[role="combobox"]:focus-visible {
        outline: none !important;
        box-shadow: none !important;
    }

    /* Hide the "Press Enter to apply" hint - it overlaps the password show/hide (eye)
       icon, and the field commits on Enter OR blur anyway, so it's redundant. */
    [data-testid="InputInstructions"] { display: none !important; }

    /* Keep the accessible focus ring for keyboard nav on buttons/links, but exclude the
       chat input and all form inputs - those get the smooth container focus handled
       above, so they never show the default black/accent double outline. */
    *:focus-visible:not([data-testid="stChatInputTextArea"]):not(input):not(textarea) { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
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
            result = get_workflow(st.session_state.api_key, st.session_state.model).full_pipeline(
                question=question, retriever=st.session_state.retriever
            )
            answer = result["draft_answer"]
            verification = result.get("verification_report", "")
            citations = result.get("citations", [])
            passages_consulted = result.get("passages_consulted", 0)
            re_researched = result.get("re_researched", False)
        except RateLimitError:
            answer = (
                "Your OpenAI key hit a rate or quota limit. Wait a moment and try again, "
                "or check your usage and billing at platform.openai.com."
            )
            verification, citations, passages_consulted, re_researched = "", [], 0, False
        except APIError as e:
            answer = f"OpenAI returned an error: {e}"
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
    # --- Bring-your-own-key gate -------------------------------------------------
    # The visitor supplies their own OpenAI key. It lives only in this session's
    # memory (never written to disk or logs), so usage is billed to them and the app
    # is safe to share publicly without our own key being drained.
    st.markdown('<div class="vs-eyebrow vs-eyebrow--lead">Your OpenAI key</div>', unsafe_allow_html=True)
    _key_in = st.text_input(
        "OpenAI API key", type="password", placeholder="sk-...",
        label_visibility="collapsed", key="api_key_input",
        help="Used only in your browser session — never stored, logged, or sent anywhere except OpenAI.",
    )
    # Validate once per distinct entry (models.list costs no tokens). The same call
    # discovers the models this key can use. Store the attempt so an invalid key doesn't
    # re-trigger validation on every rerun.
    if _key_in != st.session_state.api_key:
        st.session_state.api_key = _key_in
        if _key_in:
            with st.spinner("Verifying key…"):
                ok, msg, models = validate_key(_key_in)
            st.session_state.key_ok, st.session_state.key_msg, st.session_state.models = ok, msg, models
            # Auto-pick a sensible default from the user's own models, unless their
            # current choice is still available.
            if ok and models and st.session_state.model not in models:
                st.session_state.model = pick_default_model(models)
        else:
            st.session_state.key_ok, st.session_state.key_msg, st.session_state.models = False, "", []

    if st.session_state.key_ok:
        st.markdown('<div class="vs-key vs-key--ok">✓ Key active</div>', unsafe_allow_html=True)
        if st.session_state.models:
            _models = st.session_state.models
            _idx = _models.index(st.session_state.model) if st.session_state.model in _models else 0
            st.session_state.model = st.selectbox(
                "Model", _models, index=_idx, label_visibility="collapsed",
                help="Models your key can access. A cost-efficient one is picked by default.",
            )
        else:
            # Key valid but the models list came back empty - let them type a model ID.
            st.session_state.model = st.text_input(
                "Model", value=st.session_state.model or "gpt-4o-mini",
                label_visibility="collapsed", help="Type a chat model ID your key can use.",
            )
    else:
        # Not yet active: show any validation error, plus a clear "how to submit" hint
        # in its own line below the field (Streamlit's built-in hint overlaps the eye
        # icon, so it's hidden via CSS and replaced by this).
        if st.session_state.key_msg:
            st.markdown(f'<div class="vs-key vs-key--bad">✕ {html.escape(st.session_state.key_msg)}</div>', unsafe_allow_html=True)
        st.markdown('<div class="vs-key-hint">Paste your key, then press <kbd>Enter</kbd> to apply.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="vs-key-note">Stored only in this browser session — never saved or logged. '
        '<a href="https://platform.openai.com/api-keys" target="_blank">Get a key ↗</a></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color: var(--rule); margin: 1.1rem 0 1rem;">', unsafe_allow_html=True)

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
        # Reset the documents/conversation but KEEP the visitor's key + model so they
        # don't have to re-enter it for every new dossier.
        _keep = {"api_key", "key_ok", "key_msg", "models", "model"}
        for key in defaults:
            if key not in _keep:
                st.session_state.pop(key, None)
        st.rerun()

    # Only show the model once a key is active - before that the model is unknown
    # (discovered from the user's key), so a placeholder name would be misleading.
    _model_row = (
        f'<div class="vs-kv"><dt>Model</dt><dd>{html.escape(st.session_state.model)}</dd></div>'
        if st.session_state.key_ok and st.session_state.model else ""
    )
    st.markdown(
        f"""
        <dl class="vs-meta" style="margin-top: 1.4rem;">
            <div class="vs-kv"><dt>Retriever</dt><dd>bm25 &oplus; chroma</dd></div>
            {_model_row}
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
        '<dt>Answers are drafted from your corpus only.</dt><dd>billed to your OpenAI key</dd>'
        "</div>",
        unsafe_allow_html=True,
    )

# The chat is gated on a valid OpenAI key (answering makes LLM calls). Indexing a
# document uses only the local embedding model, so it works without a key.
_ready = st.session_state.key_ok and st.session_state.retriever is not None
if not st.session_state.key_ok:
    _placeholder = "Add your OpenAI key to begin…"
elif st.session_state.retriever is None:
    _placeholder = "Add a document to begin…"
else:
    _placeholder = "Ask about your document(s)…"

pending = st.session_state.pop("pending_question", None)
question = st.chat_input(_placeholder, disabled=not _ready)
if pending and _ready:
    question = pending
elif pending and not st.session_state.key_ok:
    # A sample was opened without a key set - hold the question until the key is in.
    st.session_state.pending_question = pending
    st.info("Enter your OpenAI API key in the sidebar, then your sample question will run.")

if question:
    ask(question)
