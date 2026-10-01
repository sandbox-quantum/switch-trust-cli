"""Anonymize hardcoded secrets in source code text.

Replaces secret values with '*' characters of matching length, preserving
the surrounding code structure (variable names, quotes, line numbers).
"""

from __future__ import annotations

import re

_SECRET_KEYWORDS = (
    r"API_?KEY|TOKEN|SECRET|PASSWORD|PASS|CREDENTIAL|DATABASE_URL|"
    r"AUTH|PRIVATE_KEY|ACCESS_KEY|SECRET_KEY|CLIENT_SECRET"
)

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(\b\w*(?:" + _SECRET_KEYWORDS + r")\w*"  # variable name containing a keyword
    r"""\s*=\s*)(["'])"""  # = and opening quote
    r"(.+?)"  # the secret value (non-greedy)
    r"(\2)",  # matching closing quote
    re.IGNORECASE,
)

_STANDALONE_SECRET_RE = re.compile(
    r"(?<=[\"' =])"  # preceded by quote, space, or equals
    r"("
    r"sk-[A-Za-z0-9\-_]{10,}"
    r"|AIza[A-Za-z0-9_\-]{35,}"
    r"|key-[A-Za-z0-9]{10,}"
    r"|pat-[A-Za-z0-9]{10,}"
    r"|ghp_[A-Za-z0-9]{30,}"
    r"|gho_[A-Za-z0-9]{30,}"
    r"|glpat-[A-Za-z0-9\-_]{20,}"
    r"|xox[bpras]-[A-Za-z0-9\-]{10,}"
    r")",
    re.IGNORECASE,
)

_BEARER_RE = re.compile(
    r"(Bearer\s+)([A-Za-z0-9\-_.~+/]+=*)",
    re.IGNORECASE,
)


def _mask_assignment(m: re.Match) -> str:
    prefix = m.group(1)  # e.g. 'API_KEY = '
    quote = m.group(2)  # " or '
    value = m.group(3)  # the secret
    return prefix + quote + "*" * len(value) + quote


def _mask_standalone(m: re.Match) -> str:
    return "*" * len(m.group(1))


def _mask_bearer(m: re.Match) -> str:
    return m.group(1) + "*" * len(m.group(2))


def anonymize_secrets(text: str) -> str:
    """Replace hardcoded secret values with '*' characters of matching length.

    Applies two strategies:
    1. Assignment-based: VAR_NAME = "secret" where the variable name contains
       keywords like KEY, TOKEN, SECRET, PASSWORD, etc.
    2. Standalone tokens: Known secret prefixes (sk-*, AIza*, key-*, pat-*, etc.)
       appearing in any context.
    """
    result = _SECRET_ASSIGNMENT_RE.sub(_mask_assignment, text)
    result = _STANDALONE_SECRET_RE.sub(_mask_standalone, result)
    result = _BEARER_RE.sub(_mask_bearer, result)
    return result
