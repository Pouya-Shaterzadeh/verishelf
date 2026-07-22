from openai import OpenAI
from config.settings import settings


def get_client() -> OpenAI:
    """Shared OpenRouter client (OpenAI-compatible API) used by all agents."""
    return OpenAI(
        base_url=settings.OPENROUTER_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": settings.OPENROUTER_SITE_URL,
            "X-Title": settings.OPENROUTER_SITE_NAME,
        },
    )
