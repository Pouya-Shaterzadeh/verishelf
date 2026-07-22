# **Verishelf** 🔎

🚀 **AI-powered Multi-Agent RAG system for intelligent document querying with fact verification**

> **Naming note:** "Verishelf" is a working title, swapped in to get this project off the ground with an original name. It lives in exactly one place — `config/constants.py` (`APP_NAME`) — so renaming later is a one-line change, no hunting through the codebase.

---

## **📌 Overview**

**Verishelf** is a **multi-agent Retrieval-Augmented Generation (RAG) system** that helps you query **long, complex documents** and get **accurate, fact-verified answers**. Unlike a plain chatbot, which can hallucinate or struggle with structured data, Verishelf **retrieves, verifies, and self-corrects** before it shows you an answer.

💡 **Key Features:**
- ✅ **Multi-Agent System** — a Research Agent drafts answers, a Verification Agent fact-checks them against the source document, with a bounded self-correction loop if the check fails.
- ✅ **Hybrid Retrieval** — BM25 keyword search + local vector embeddings (no embedding API key needed).
- ✅ **Handles multiple documents** — selects the most relevant content even across several uploaded files.
- ✅ **Scope detection** — rejects out-of-scope questions instead of making something up.
- ✅ **A real web UI** — a Streamlit chat interface with document upload, streaming-style chat, and an inline verification report per answer.

This project started from the IBM Skills Network "DocChat" hands-on lab and keeps its backend architecture — LangGraph multi-agent workflow, Docling document parsing, hybrid BM25 + vector retrieval. What changed:

| Layer | Lab version | This project |
|---|---|---|
| LLMs | IBM watsonx.ai, sandbox-only `project_id="skills-network"` | Free-tier models with multi-provider failover (NVIDIA · Groq · OpenRouter), your own API keys |
| Embeddings | IBM watsonx embeddings (needs the same sandbox) | Local `sentence-transformers` model — free, no key, runs on CPU |
| Frontend | Gradio | Streamlit, custom-designed |

---

## **🛠️ How it works**

1. **Query processing & scope analysis** — Verishelf checks whether your question is answerable from the uploaded document(s) before doing anything else.
2. **Multi-agent research & retrieval** — Docling converts documents to structured Markdown; a hybrid BM25 + vector retriever (Chroma, in-memory per session) finds the relevant chunks.
3. **Answer generation & verification** — a Research Agent drafts an answer; a Verification Agent checks it against the retrieved text and flags unsupported claims or contradictions. If it fails, the system re-researches once more before giving up gracefully.
4. **Response finalization** — you see the answer plus an expandable verification report (✅ verified / ⚠️ flagged).

---

## **📦 Installation**

### 1. Clone and enter the repo
```bash
git clone <your-fork-url> verishelf
cd verishelf
```

### 2. Set up a virtual environment
```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
This pulls in Docling + PyTorch (CPU build) + a local embedding model, so the first install can take a few minutes and a few GB of disk.

### 4. Get free LLM API keys (one or more)
Verishelf uses **multi-provider failover** across three free, OpenAI-compatible LLM providers — each LLM call is tried against them in order until one succeeds, so a rate limit or outage on one silently falls over to the next. Set **at least one** key; set all three for full resilience.

| Provider | Get a free key | Env var |
|---|---|---|
| NVIDIA (build.nvidia.com) | [build.nvidia.com](https://build.nvidia.com) | `NVIDIA_API_KEY` |
| Groq | [console.groq.com](https://console.groq.com) | `GROQ_API_KEY` |
| OpenRouter | [openrouter.ai/keys](https://openrouter.ai/keys) | `OPENROUTER_API_KEY` |

Copy `.env.example` to `.env` and paste in whichever keys you have:
```bash
cp .env.example .env
```

**Free-tier limits to know about:** each provider has its own per-minute cap (roughly 30–60 req/min), and one shared key serves *all* visitors of a deployed app — the failover across three providers is what keeps it resilient under load. Each question burns 2–5 LLM calls (relevance check, research, verification, and up to one retry). Default models are fast, non-reasoning instruct models (`config/settings.py`); avoid "reasoning" models (they return empty content on small token budgets).

### 5. Run it
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Upload document(s)** in the sidebar (PDF, DOCX, TXT, Markdown), or load one of the built-in examples.
2. **Ask a question** in the chat box.
3. Verishelf retrieves, drafts, and verifies an answer — the verification report is one click away under each answer.
4. Out-of-scope questions get an honest "I can't answer that from these documents" instead of a fabricated response.

---

## ☁️ Deploying for free (Streamlit Community Cloud)

1. Push this repo to your own GitHub account (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, and click **New app**.
3. Pick this repo/branch, set **Main file path** to `app.py`.
4. In **Advanced settings → Secrets**, add at least one provider key (all three recommended):
   ```
   NVIDIA_API_KEY = "your-nvidia-key"
   GROQ_API_KEY = "your-groq-key"
   OPENROUTER_API_KEY = "your-openrouter-key"
   ```
5. Deploy — first build takes a few minutes (Docling + PyTorch + the embedding model download).

**Resource notes for the free tier (~1 GB RAM, shared vCPU):**
- OCR is **off by default** (`ENABLE_OCR=false`) — it's the single biggest memory/time cost in Docling's pipeline, and most PDFs have a real text layer that doesn't need it. Only turn it on (as a secret/env override) if you specifically need scanned-document support.
- This tier is noticeably tighter than a local machine — expect slower cold starts and possible memory pressure on large documents. If you hit persistent OOM errors, a paid host with more RAM (Hugging Face Spaces PRO, a small VPS, Google Cloud Run) will fix it directly.
- The document cache (`document_cache/`) and any temp files are ephemeral — they won't survive a restart, which is expected.
- Each visitor gets an isolated in-memory vector store (keyed per browser session), so concurrent users never see each other's documents.
- Free apps sleep after a period of inactivity and cold-start on the next visit — the "warming up" spinners in the sidebar are exactly for that first load.

If you outgrow the free tier, the same `requirements.txt` runs unmodified on Hugging Face Spaces (Docker SDK, requires a PRO subscription for cpu-basic), Render, a small VPS, or Google Cloud Run — just run `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`.

---

## 🤝 Contributing

- **Fork the repo**
- **Create a new branch** (`feature-xyz`)
- **Commit your changes**
- **Submit a PR**

---

## Credits

Backend architecture originally built for the IBM Skills Network "DocChat" lab, authored by Hailey Quach, with contributions from Ricky Shi and Wojciech "Victor" Fulmyk. This fork keeps that backend design and replaces the model provider and the entire frontend.
