from openai import OpenAI
from config.settings import settings


def get_client() -> OpenAI:
    """Shared LLM client. The provider is any OpenAI-compatible endpoint (default:
    NVIDIA build.nvidia.com / NIM) configured via LLM_BASE_URL + LLM_API_KEY.

    max_retries is raised above the SDK default of 2: NVIDIA's free tier is ~40
    req/min shared across the key, so a burst (each question is 3-5 calls, and the
    speculative-research feature fires two at once) can briefly trip a 429. The SDK
    retries 429/5xx/timeouts with exponential backoff and honors any Retry-After
    header, so more retries absorb these transient limits silently instead of
    surfacing an error. timeout caps a hung request so a slow call can't stall the UI.
    """
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        max_retries=5,
        timeout=30.0,
    )
