import ast
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

REQUIREMENTS_FILE = "requirements.txt"
PYTHON_EXTENSIONS = ".py"
PYTHON_IGNORES = "__init__.py"

# Import module root -> Framework name
FRAMEWORK_ROOTS: dict[str, str] = {
    "google.adk": "Google ADK",
    "google.genai": "Google GenAI",
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "agents": "OpenAI Agents SDK",
    "langgraph": "LangGraph",
    "crewai": "CrewAI",
    "autogen": "AutoGen",
    "transformers": "HuggingFace Transformers",
    "smolagents": "HuggingFace smolagents",
}


class FileType(Enum):
    PYTHON = "python"
    REQUIREMENTS = "requirements"
    OTHER = "other"


@dataclass
class RelevantFile:
    path: str
    type: FileType
    evidence: str | None = None
    framework: str | None = None


def _detect_relevant_file(
    abs_file_path: str,
) -> RelevantFile | None:
    """Detects relevant files for scanning."""
    if os.path.basename(abs_file_path).lower() == REQUIREMENTS_FILE:
        logger.info("File in scope: %s", abs_file_path)
        return RelevantFile(path=abs_file_path, type=FileType.REQUIREMENTS)
    elif abs_file_path.endswith(PYTHON_EXTENSIONS) and not abs_file_path.endswith(
        PYTHON_IGNORES
    ):
        match = _detect_framework_in_file(abs_file_path)
        if match:
            logger.info("File in scope: %s", abs_file_path)
            return match
    return None


def find_relevant_files(path: str) -> list[RelevantFile]:
    """`path` might be a file or a directory.
    If it is a file, detect its type, if it is a directory,
    recursively find relevant files in the given directory."""
    relevant_files = []

    path = os.path.abspath(path)
    if os.path.isfile(path):
        relevant_file = _detect_relevant_file(path)
        if relevant_file:
            relevant_files.append(relevant_file)
    else:
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                relevant_file = _detect_relevant_file(os.path.join(dirpath, filename))
                if relevant_file:
                    relevant_files.append(relevant_file)

    return relevant_files


def is_relevant_file(file_path: str) -> bool:
    file_name = os.path.basename(file_path).lower()

    if file_name == REQUIREMENTS_FILE:
        return True

    if file_path.endswith(PYTHON_EXTENSIONS) and not file_path.endswith(PYTHON_IGNORES):
        return is_relevant_python_file(file_path)

    return file_path.endswith(".py")


def is_relevant_python_file(file_path: str) -> bool:
    content = ""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

    except Exception as e:
        logger.warning("Failed to read file %s: %s", file_path, e)
        return True

    return has_relevant_imports(content)


def has_relevant_imports(content: str) -> bool:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _matches_framework(alias.name):
                    return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and _matches_framework(node.module):
                return True

    return False


def _detect_framework_in_file(file_path: str) -> RelevantFile | None:
    """Read a Python file and return a RelevantFile if it imports a known framework."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        logger.warning("Failed to read file %s: %s", file_path, e)
        return RelevantFile(path=file_path, type=FileType.PYTHON)

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                framework = _get_framework_name(alias.name)
                if framework:
                    return RelevantFile(
                        path=file_path,
                        type=FileType.PYTHON,
                        evidence=ast.get_source_segment(content, node)
                        or f"import {alias.name}",
                        framework=framework,
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                framework = _get_framework_name(node.module)
                if framework:
                    return RelevantFile(
                        path=file_path,
                        type=FileType.PYTHON,
                        evidence=ast.get_source_segment(content, node)
                        or f"from {node.module} import ...",
                        framework=framework,
                    )

    return None


def _matches_framework(module: str) -> bool:
    return _get_framework_name(module) is not None


def _get_framework_name(module: str) -> str | None:
    for root, name in FRAMEWORK_ROOTS.items():
        if module == root or module.startswith(root + "."):
            return name
    return None
