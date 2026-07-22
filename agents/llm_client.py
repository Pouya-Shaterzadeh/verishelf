from openai import OpenAI, RateLimitError, APIError, APITimeoutError, APIConnectionError
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

# Errors that mean "this provider is unavailable right now" - fail over to the next
# one rather than surfacing to the user. A genuine bug (e.g. a bad request) is not in
# this list, so it still propagates.
_FALLBACK_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError, APIError)


class MultiProviderLLM:
    """Chat-completions client that fails over across OpenAI-compatible providers.

    Tries each configured provider (settings.active_providers(), in failover order)
    until one returns, so a rate limit / timeout / outage on one silently drops to the
    next. Each provider carries its own model ID. Per-provider max_retries is low (1)
    so a rate-limited primary fails over quickly instead of waiting out its backoff -
    the other providers are the real safety net. Only if ALL providers fail is the
    last error raised (so the UI's RateLimitError/APIError handlers still fire).
    """

    def __init__(self, providers):
        if not providers:
            raise RuntimeError(
                "No LLM provider configured. Set at least one of NVIDIA_API_KEY, "
                "GROQ_API_KEY, or OPENROUTER_API_KEY."
            )
        self._providers = [
            {
                **p,
                "client": OpenAI(
                    base_url=p["base_url"], api_key=p["api_key"],
                    max_retries=1, timeout=30.0,
                ),
            }
            for p in providers
        ]
        logger.info(f"LLM failover order: {[p['name'] for p in self._providers]}")

    def create(self, *, messages, max_tokens, temperature):
        """OpenAI-style chat completion with cross-provider failover. Returns the raw
        completion object (callers read .choices[0].message.content as usual)."""
        last_err = None
        for p in self._providers:
            try:
                return p["client"].chat.completions.create(
                    model=p["model"], messages=messages,
                    max_tokens=max_tokens, temperature=temperature,
                )
            except _FALLBACK_ERRORS as e:
                logger.warning(
                    f"LLM provider '{p['name']}' unavailable "
                    f"({type(e).__name__}); failing over to next."
                )
                last_err = e
        # Every provider failed - re-raise the last error so the UI can react to it.
        raise last_err


def get_client() -> MultiProviderLLM:
    """Shared multi-provider LLM client (see MultiProviderLLM)."""
    return MultiProviderLLM(settings.active_providers())


def probe_provider(base_url: str, api_key: str, model: str, timeout: float = 12.0) -> bool:
    """Liveness check for one provider: a 1-token chat completion. Returns True iff the
    provider answers right now - i.e. key valid, reachable, and not rate-limited. Uses
    the same call path as real usage (NVIDIA's /models endpoint is public and wouldn't
    catch a bad key or a rate limit, so a real completion is the only honest signal).
    max_retries=0 so a down provider is reported promptly rather than retried."""
    if not api_key:
        return False
    try:
        client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True
    except Exception as e:
        logger.info(f"Provider probe failed for {base_url}: {type(e).__name__}")
        return False
