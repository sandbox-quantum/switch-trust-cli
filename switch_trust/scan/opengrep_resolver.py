"""
opengrep_resolver.py — Locate the OpenGrep binary.
"""

import shutil


def find_opengrep_binary() -> str | None:
    """
    Returns the path to the binary, or None if not found.
    """
    return shutil.which("opengrep")
