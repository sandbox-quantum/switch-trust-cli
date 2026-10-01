"""Lazy imports for the optional ``full`` dependency extra.

The published ``switch-trust-cli`` package ships ``transformers[torch]`` and
``garak`` as an optional extra (``pip install switch-trust-cli[full]``) rather than
as core dependencies, because they are large (``torch`` alone is ~2 GB and
on Linux also pulls in Cuda). The helpers here import those packages on demand
and raise a friendly ``ImportError`` pointing at the extra when they are
missing, instead of the bare import error a user would otherwise see.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any

_INSTALL_HINT = "Install it with: pip install switch-trust-cli[full]"


def require_transformers_pipeline() -> Callable[..., Any]:
    """Return ``transformers.pipeline``.

    Raises ``ImportError`` with an install hint if ``transformers`` (part of
    the optional ``full`` extra) is not installed.
    """
    try:
        from transformers import pipeline
    except ImportError as e:
        raise ImportError(
            "transformers is required for HuggingFace models and toxicity "
            f"detection. {_INSTALL_HINT}"
        ) from e
    return pipeline


def require_garak() -> ModuleType:
    """Return the top-level ``garak`` module.

    The submodules the eval engine reaches into (``garak.attempt``,
    ``garak._plugins``, ``garak.generators.base``, ...) are imported here so
    they are attribute-accessible on the returned module — a bare
    ``import garak`` does not guarantee that.

    Raises ``ImportError`` with an install hint if ``garak`` (part of the
    optional ``full`` extra) is not installed.
    """
    try:
        import garak
        import garak._config  # noqa: F401
        import garak._plugins  # noqa: F401
        import garak.attempt  # noqa: F401
        import garak.generators.base  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"garak is required for garak probes and detectors. {_INSTALL_HINT}"
        ) from e
    return garak
