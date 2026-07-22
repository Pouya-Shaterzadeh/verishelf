# Container image for Google Cloud Run (also works on any Docker host).
FROM python:3.11-slim

# System libraries Docling/opencv need at runtime (libGL etc.). python:3.11-slim is
# Debian bookworm, where these are the correct (non-t64) package names.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the models INTO the image so a Cloud Run cold start doesn't re-fetch
# them on the first request (Cloud Run scales to zero and has no persistent disk, so
# without this every cold start would re-download hundreds of MB). Baked into the HF
# cache under /root/.cache, which the app reads at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# Docling's layout models (best-effort - the helper's name has varied across versions,
# so don't fail the build if it's absent; worst case they download on first parse).
RUN python -c "from docling.utils.model_downloader import download_models; download_models()" || true

COPY . .

# Cloud Run injects $PORT (default 8080); Streamlit must bind it on 0.0.0.0. CORS/XSRF
# are disabled because Streamlit sits behind Cloud Run's proxy - with XSRF on, the file
# uploader (core to this app) fails behind the proxy.
ENV PORT=8080
EXPOSE 8080
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
