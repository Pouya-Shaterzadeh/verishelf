from openai import OpenAI
from config.settings import settings


def get_client() -> OpenAI:
    """Shared LLM client. The provider is any OpenAI-compatible endpoint (default:
    NVIDIA build.nvidia.com / NIM) configured via LLM_BASE_URL + LLM_API_KEY."""
    return OpenAI(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
    )
