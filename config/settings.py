from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # Required settings
    LLM_API_KEY: str

    # LLM provider connection. Provider-neutral names (LLM_*) because we've swapped
    # providers before - any OpenAI-compatible endpoint works by overriding these two
    # in .env. Default is NVIDIA's build.nvidia.com (NIM) hosted catalog: free for
    # development with a ~40 requests/minute rate limit and, crucially, NO daily quota
    # (the OpenRouter free tier's 50/day cap was the recurring "rate limit" failure).
    LLM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"

    # Model routing. meta/llama-3.1-8b-instruct: a plain instruct model (NOT a
    # "reasoning" model - those spend the token budget on hidden chain-of-thought and
    # return empty content on the small max_tokens these prompts use). Chosen for
    # SPEED: on NVIDIA's free tier the big 70B model is heavily queued (~15-25s to
    # first byte), while the 8B answers in well under a second and still follows the
    # strict relevance-label and verification formats cleanly. Browse the catalog at
    # https://build.nvidia.com/models and override any of these via .env - just avoid
    # reasoning models. For higher quality at the cost of latency, meta/llama-3.3-70b-instruct.
    RESEARCH_MODEL: str = "meta/llama-3.1-8b-instruct"
    VERIFICATION_MODEL: str = "meta/llama-3.1-8b-instruct"
    RELEVANCE_MODEL: str = "meta/llama-3.1-8b-instruct"

    # Embeddings run locally (free, no API key needed)
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
    # question we've paid for one research call we then discard. Set False to trade that
    # speed for strictly minimal free-tier quota usage.
    SPECULATIVE_RESEARCH: bool = True

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # Cache settings
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # Ignore unrecognized env vars (e.g. a leftover key from a previous provider)
        # instead of crashing on them - only the fields declared above are consumed.
        extra = "ignore"

settings = Settings()
