"""constants.py — Shared constants for the agent scanner."""

REQUIREMENT_FILE_NAMES = (
    "requirements.txt",
    "requirements_lock.txt",
    "pyproject.toml",
)

PYTHON_FILE_EXTENSION = ".py"

CONFIG_FILE_EXTENSIONS = (".toml", ".cfg")

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3, "Info": 4}

VALID_SEVERITIES = {"critical", "high", "medium", "low", "info"}
