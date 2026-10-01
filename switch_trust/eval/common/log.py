import logging
import os
import re

_REDACT_PATTERNS = re.compile(
    r"("
    r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"
    r"|sk-[A-Za-z0-9]{10,}"
    r"|key-[A-Za-z0-9]{10,}"
    r"|pat-[A-Za-z0-9]{10,}"
    r"|AIza[A-Za-z0-9_\-]{35,}"
    r"|xox[bspra]-[A-Za-z0-9\-]+"
    r"|ghp_[A-Za-z0-9]{36,}"
    r"|-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"
    r"[\s\S]*?-----END"
    r")",
    re.IGNORECASE,
)

_REPLACEMENT = "[REDACTED]"


def redact(text: str) -> str:
    """Replace any known secret pattern in ``text`` with ``[REDACTED]``."""
    return _REDACT_PATTERNS.sub(_REPLACEMENT, text)


class RedactingFilter(logging.Filter):
    """Scrub secrets from log records before they reach handlers."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
        return True


class RedactingFormatter(logging.Formatter):
    """Wrap another formatter and scrub secrets from its rendered output.

    A handler-level :class:`RedactingFilter` only sees ``record.msg`` and
    ``record.args``, so secrets that surface elsewhere — an exception message or
    traceback (e.g. an aiohttp connect error carrying a key in the URL), or a
    stray ``extra=`` field — reach the handler unredacted. Redacting the fully
    formatted string closes that gap regardless of how the wrapped formatter
    renders exceptions.
    """

    def __init__(self, inner: logging.Formatter) -> None:
        super().__init__()
        self._inner = inner

    def format(self, record: logging.LogRecord) -> str:
        return redact(self._inner.format(record))


_NOISY_LOGGERS = [
    "anthropic",
    "google.genai",
    "google.adk",
    "google_adk",
    "google.auth",
    "google.api_core",
    "google_genai.models",
    "httpcore",
    "httpx",
    "urllib3",
    "opentelemetry",
    "litellm",
    "LiteLLM",
    "openai",
    "asyncio",
]


def silence_noisy_loggers() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.captureWarnings(True)

    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("DATASETS_VERBOSITY", "error")


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s"


def setup_file_logging(log_path: str) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    file_handler.addFilter(RedactingFilter())
    root.addHandler(file_handler)

    os.environ["LITELLM_LOG"] = "CRITICAL"
    try:
        import litellm  # noqa: PLC0415

        litellm.suppress_debug_info = True
    except ImportError:
        pass
    silence_noisy_loggers()
