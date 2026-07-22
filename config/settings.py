from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # --- LLM (bring-your-own-key) ---
    # Each visitor supplies their own OpenAI API key at runtime in the app (see app.py),
    # so there is NO server-side LLM key here - usage is billed to the visitor's own key
    # and their own rate limits, which is what makes this safe to share publicly.
    # OPENAI_BASE_URL is overridable for OpenAI-compatible providers, but the app is
    # designed around OpenAI keys.
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    # Cost-effective, fast, strong at the strict relevance-label / verification formats
    # these prompts need. Users can pick a model in the UI (see OPENAI_MODELS).
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_MODELS: list = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]

    # Embeddings run locally (free, no API key needed - the user's key is only for the LLM)
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Optional settings with defaults
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # Document processing
    ENABLE_OCR: bool = False  # scanned-PDF text extraction; costs a lot of memory/time to enable

    # Retrieval settings
    VECTOR_SEARCH_K: int = 10
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]

    # Run the ResearchAgent's first draft concurrently with the RelevanceChecker
    # instead of waiting for the relevance verdict first. Research only needs the
    # retrieved passages, so on in-scope questions (the common case) this removes the
    # relevance call's latency from the critical path. The cost: on a rare out-of-scope
    # question we've paid for one research call we then discard.
    SPECULATIVE_RESEARCH: bool = True

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # Cache settings
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Ignore unrecognized env vars instead of crashing on them.
        extra = "ignore"

settings = Settings()
