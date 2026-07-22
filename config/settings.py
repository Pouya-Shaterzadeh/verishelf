from pydantic_settings import BaseSettings
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # Required settings
    OPENROUTER_API_KEY: str

    # OpenRouter connection
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # Optional attribution headers OpenRouter uses for its public rankings - safe to leave as-is.
    OPENROUTER_SITE_URL: str = "https://github.com"
    OPENROUTER_SITE_NAME: str = "Verishelf"

    # Model routing - defaults are free-tier models on OpenRouter.
    # Deliberately NOT a "reasoning" model (e.g. the nvidia/nemotron-*:free or
    # openai/gpt-oss-*:free families): those burn the whole max_tokens budget on
    # hidden chain-of-thought before ever emitting the actual answer, which comes
    # back as content=None on the small token budgets these prompts use.
    # Free models rotate over time; check https://openrouter.ai/models?max_price=0
    # and override via .env if this gets deprecated - verify a candidate isn't a
    # reasoning model first (check for a populated `reasoning` field with content=None).
    RESEARCH_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    VERIFICATION_MODEL: str = "google/gemma-4-26b-a4b-it:free"
    RELEVANCE_MODEL: str = "google/gemma-4-26b-a4b-it:free"

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

settings = Settings()
