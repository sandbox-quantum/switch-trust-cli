import logging
import os
import re

# Kept in sync by hand with servicelib.redact's VALUE_SHAPE_PATTERN and
# KEY_NAME_PATTERN (common/py/servicelib), the same patterns roi's
# platform/log.py imports directly — this file can't do the same because
# cli_exporter/export.py copies eval/common/ verbatim into the standalone
# flintai-cli PyPI package, which cannot depend on servicelib. Update both when
# adding a new secret shape.
#
# Two patterns, applied in this order — not one: re.sub never rescans text it
# has already consumed, so if the key-name pattern ran first against
# "api_key: -----BEGIN RSA PRIVATE KEY-----\n<body>", its `\S+` would stop at
# the first space and swallow only "api_key: -----BEGIN", hiding the rest of
# the private key. Running the value-shape pattern first redacts the whole
# block before the key-name pattern ever sees it.
_VALUE_SHAPE_PATTERN = re.compile(
    r"("
    r"Bearer\s+[A-Za-z0-9\-_.~+/]+=*"
    r"|sk-[A-Za-z0-9\-_]{8,}"
    r"|key-[A-Za-z0-9]{10,}"
    r"|pat-[A-Za-z0-9]{10,}"
    r"|AIza[A-Za-z0-9_\-]{35,}"
    r"|xox[bspra]-[A-Za-z0-9\-]+"
    r"|ghp_[A-Za-z0-9]{36,}"
    r"|-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"
    # Bounded at 8192 chars: unbounded [\s\S]*? is quadratic when a BEGIN
    # header has no matching -----END later in the text, and this runs on
    # every log record.
    r"[\s\S]{0,8192}?-----END"
    r")",
    re.IGNORECASE,
)

# By key name, for a value VALUE_SHAPE_PATTERN didn't already catch. The
# optional scheme word before \S+ covers "Bearer", "Basic", or any other auth
# scheme, so "Authorization: Basic dXNlcjpwYXNz" redacts in full.
_KEY_NAME_PATTERN = re.compile(
    r"("
    r"authorization\s*[:=]\s*(?:[A-Za-z]+\s+)?\S+"
    r"|api[_-]?key\s*[:=]\s*\S+"
    r")",
    re.IGNORECASE,
)

_REPLACEMENT = "[REDACTED]"


def redact(text: str) -> str:
    """Replace any known secret pattern in ``text`` with ``[REDACTED]``."""
    return _KEY_NAME_PATTERN.sub(
        _REPLACEMENT, _VALUE_SHAPE_PATTERN.sub(_REPLACEMENT, text)
    )


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
