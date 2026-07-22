from openai import OpenAI, AuthenticationError, PermissionDeniedError
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


def make_client(api_key: str) -> OpenAI:
    """Build an OpenAI client from the user's own API key (bring-your-own-key).

    The key comes from the visitor at runtime and lives only in their Streamlit
    session - it is never read from the server environment, persisted, or logged.
    """
    return OpenAI(
        api_key=api_key,
        base_url=settings.OPENAI_BASE_URL,
        max_retries=2,
        timeout=60.0,
    )


def validate_key(api_key: str) -> tuple[bool, str]:
    """Cheaply verify a key authenticates, without spending any tokens.

    Returns (ok, message). Uses the models endpoint - it requires a valid key but
    costs nothing. Never logs the key itself.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Enter your OpenAI API key to begin."
    try:
        make_client(api_key).models.list()
        return True, "Key verified — you're ready to go."
    except (AuthenticationError, PermissionDeniedError):
        return False, "That key was rejected. Double-check it and try again."
    except Exception as e:  # network hiccup, etc.
        logger.info(f"Key validation could not complete: {type(e).__name__}")
        return False, f"Couldn't verify the key right now ({type(e).__name__}). Try again."
