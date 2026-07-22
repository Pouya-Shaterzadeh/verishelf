from pydantic_settings import BaseSettings
from pydantic import model_validator
from .constants import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES

class Settings(BaseSettings):
    # --- LLM providers (multi-provider fallback) ---
    # Each LLM call is tried against the configured providers in PROVIDER_ORDER until
    # one succeeds, so a rate limit / timeout / outage on one silently fails over to
    # the next. A provider is "active" only if its API key is set; at least one is
    # required (enforced below). All are OpenAI-compatible, so the client is uniform -
    # only the base URL and model ID differ per provider.
    GEMINI_API_KEY: str = ""
    CEREBRAS_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    CEREBRAS_BASE_URL: str = "https://api.cerebras.ai/v1"
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # One model per provider (the same request, but each provider names models
    # differently). All are fast, non-reasoning instruct models validated to follow
    # the strict relevance-label / verification formats these prompts need. (Reasoning
    # models spend the token budget on hidden chain-of-thought and return empty content
    # on the small max_tokens used here.) OpenRouter has no free Llama, so it uses the
    # free Gemma we validated there.
    GEMINI_MODEL: str = "gemini-2.0-flash"
    CEREBRAS_MODEL: str = "llama3.1-8b"
    NVIDIA_MODEL: str = "meta/llama-3.1-8b-instruct"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENROUTER_MODEL: str = "google/gemma-4-26b-a4b-it:free"

    # Failover order, most-generous first so the primary absorbs the most load before
    # failing over: Gemini (~60 rpm) -> Cerebras (1M tokens/day, very fast) -> NVIDIA
    # (no daily cap) -> Groq (fast but low token/min cap) -> OpenRouter (50/day, last
    # resort). Reorder via .env to reprioritize.
    PROVIDER_ORDER: list = ["gemini", "cerebras", "nvidia", "groq", "openrouter"]

    @model_validator(mode="after")
    def _require_a_provider(self):
        if not self.active_providers():
            raise ValueError(
                "No LLM provider configured. Set at least one of GEMINI_API_KEY, "
                "CEREBRAS_API_KEY, NVIDIA_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY "
                "in .env."
            )
        return self

    def _provider_cfg(self) -> dict:
        """name -> (label, base_url, api_key, model) for every provider."""
        return {
            "gemini": ("Gemini", self.GEMINI_BASE_URL, self.GEMINI_API_KEY, self.GEMINI_MODEL),
            "cerebras": ("Cerebras", self.CEREBRAS_BASE_URL, self.CEREBRAS_API_KEY, self.CEREBRAS_MODEL),
            "nvidia": ("Nvidia", self.NVIDIA_BASE_URL, self.NVIDIA_API_KEY, self.NVIDIA_MODEL),
            "groq": ("Groq", self.GROQ_BASE_URL, self.GROQ_API_KEY, self.GROQ_MODEL),
            "openrouter": ("OpenRouter", self.OPENROUTER_BASE_URL, self.OPENROUTER_API_KEY, self.OPENROUTER_MODEL),
        }

    def active_providers(self) -> list:
        """Configured providers in failover order - only those whose key is set."""
        cfg = self._provider_cfg()
        active = []
        for name in self.PROVIDER_ORDER:
            if name not in cfg:
                continue
            label, base, key, model = cfg[name]
            if key:
                active.append({"name": name, "base_url": base, "api_key": key, "model": model})
        return active

    def all_providers(self) -> list:
        """Every provider (with display label), regardless of whether a key is set - for
        the status UI, in failover order. active_providers() is the subset actually used."""
        cfg = self._provider_cfg()
        return [
            {"name": n, "label": cfg[n][0], "base_url": cfg[n][1], "api_key": cfg[n][2], "model": cfg[n][3]}
            for n in self.PROVIDER_ORDER if n in cfg
        ]

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
