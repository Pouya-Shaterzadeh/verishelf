from openai import OpenAI, AuthenticationError, PermissionDeniedError
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Substrings that mark a NON-chat gpt-* model (image/audio/etc.) - excluded from the
# picker so every listed model actually works with this app's chat prompts.
_NON_CHAT = ("embedding", "tts", "whisper", "dall-e", "audio", "realtime",
             "transcribe", "image", "moderation", "search", "codex")


def _is_usable_chat_model(model_id: str) -> bool:
    """True for gpt-* chat models only. Deliberately excludes the o-series 'reasoning'
    models: they spend the token budget on hidden reasoning and return empty content on
    the tiny max_tokens these prompts use (e.g. the 10-token relevance label), which
    would silently break the app - so we don't offer them."""
    m = model_id.lower()
    if not m.startswith("gpt"):
        return False
    return not any(x in m for x in _NON_CHAT)


def pick_default_model(models: list) -> str:
    """Soft default from the user's own models: prefer a cost-efficient '*-mini' tier,
    else the first (list is sorted newest-first). No hardcoded name to go stale."""
    if not models:
        return ""
    minis = [m for m in models if "mini" in m]
    return (minis or models)[0]


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


def validate_key(api_key: str) -> tuple[bool, str, list]:
    """Verify a key and, from the SAME call, discover the chat models it can use.

    Returns (ok, message, models). The models list is fetched from /v1/models - so the
    app always reflects OpenAI's current lineup and each key's own access, with no
    hardcoded model names to maintain. Costs no tokens. Never logs the key.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return False, "Enter your OpenAI API key to begin.", []
    try:
        resp = make_client(api_key).models.list()
        models = sorted(
            (m.id for m in resp.data if _is_usable_chat_model(m.id)),
            reverse=True,  # newest-ish first (e.g. gpt-5* before gpt-4*)
        )
        return True, "Key verified — you're ready to go.", models
    except (AuthenticationError, PermissionDeniedError):
        return False, "That key was rejected. Double-check it and try again.", []
    except Exception as e:  # network hiccup, etc.
        logger.info(f"Key validation could not complete: {type(e).__name__}")
        return False, f"Couldn't verify the key right now ({type(e).__name__}). Try again.", []
