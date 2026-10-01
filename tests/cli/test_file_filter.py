"""
Tests for file_filter.py — framework-aware file filtering.
"""

import os
import tempfile
import unittest

from switch_trust.cli.file_filter import (
    FRAMEWORK_ROOTS,
    FileType,
    RelevantFile,
    _detect_framework_in_file,
    _get_framework_name,
    _matches_framework,
    find_relevant_files,
    has_relevant_imports,
    is_relevant_file,
    is_relevant_python_file,
)


class TestFrameworkRoots(unittest.TestCase):
    def test_is_dict(self):
        self.assertIsInstance(FRAMEWORK_ROOTS, dict)

    def test_has_expected_entries(self):
        self.assertIn("openai", FRAMEWORK_ROOTS)
        self.assertIn("anthropic", FRAMEWORK_ROOTS)
        self.assertIn("google.adk", FRAMEWORK_ROOTS)
        self.assertIn("crewai", FRAMEWORK_ROOTS)

    def test_values_are_human_readable(self):
        for root, name in FRAMEWORK_ROOTS.items():
            with self.subTest(root=root):
                self.assertIsInstance(name, str)
                self.assertGreater(len(name), 0)


class TestGetFrameworkName(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(_get_framework_name("openai"), "OpenAI")
        self.assertEqual(_get_framework_name("anthropic"), "Anthropic")
        self.assertEqual(_get_framework_name("google.adk"), "Google ADK")

    def test_subpackage_match(self):
        self.assertEqual(_get_framework_name("openai.agents"), "OpenAI")
        self.assertEqual(_get_framework_name("google.adk.tools"), "Google ADK")
        self.assertEqual(_get_framework_name("langgraph.graph"), "LangGraph")

    def test_no_match(self):
        self.assertIsNone(_get_framework_name("os"))
        self.assertIsNone(_get_framework_name("google.cloud"))
        self.assertIsNone(_get_framework_name("openai_utils"))


class TestMatchesFramework(unittest.TestCase):
    def test_exact_match(self):
        for name in [
            "openai",
            "anthropic",
            "langgraph",
            "agents",
            "google.adk",
            "google.genai",
        ]:
            with self.subTest(name=name):
                self.assertTrue(_matches_framework(name))

    def test_subpackage_match(self):
        cases = [
            "openai.agents",
            "anthropic.types.message",
            "langgraph.graph",
            "google.adk.agents",
            "google.genai.types",
            "agents.run",
        ]
        for name in cases:
            with self.subTest(name=name):
                self.assertTrue(_matches_framework(name))

    def test_no_match(self):
        for name in ["os", "google.cloud", "google", "agentsmith", "openai_utils"]:
            with self.subTest(name=name):
                self.assertFalse(_matches_framework(name))


class TestHasRelevantImports(unittest.TestCase):
    def test_simple_imports(self):
        for code in [
            "import openai",
            "import anthropic",
            "import langgraph",
            "import agents",
            "import google.adk",
            "import google.genai",
        ]:
            with self.subTest(code=code):
                self.assertTrue(has_relevant_imports(code))

    def test_from_imports(self):
        for code in [
            "from openai import ChatCompletion",
            "from anthropic import Anthropic",
            "from langgraph.graph import StateGraph",
            "from google.adk.agents import Agent",
        ]:
            with self.subTest(code=code):
                self.assertTrue(has_relevant_imports(code))

    def test_comment_not_matched(self):
        self.assertFalse(has_relevant_imports("# import openai"))

    def test_string_not_matched(self):
        self.assertFalse(has_relevant_imports('x = "import openai"'))

    def test_no_relevant_imports(self):
        self.assertFalse(has_relevant_imports("import os\nimport json"))

    def test_empty_content(self):
        self.assertFalse(has_relevant_imports(""))

    def test_syntax_error_returns_false(self):
        self.assertFalse(has_relevant_imports("def broken("))


class TestDetectFrameworkInFile(unittest.TestCase):
    def test_detects_openai(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import openai\nclient = openai.Client()\n")
            path = f.name
        try:
            result = _detect_framework_in_file(path)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, RelevantFile)
            self.assertEqual(result.path, path)
            self.assertEqual(result.framework, "OpenAI")
            self.assertIn("openai", result.evidence)
        finally:
            os.unlink(path)

    def test_detects_from_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("from anthropic import Anthropic\n")
            path = f.name
        try:
            result = _detect_framework_in_file(path)
            self.assertIsNotNone(result)
            self.assertEqual(result.framework, "Anthropic")
            self.assertIn("anthropic", result.evidence)
        finally:
            os.unlink(path)

    def test_detects_google_adk(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("from google.adk.agents import Agent\n")
            path = f.name
        try:
            result = _detect_framework_in_file(path)
            self.assertIsNotNone(result)
            self.assertEqual(result.framework, "Google ADK")
        finally:
            os.unlink(path)

    def test_returns_none_for_no_framework(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import os\nimport json\n")
            path = f.name
        try:
            result = _detect_framework_in_file(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_returns_none_for_syntax_error(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def broken(\n")
            path = f.name
        try:
            result = _detect_framework_in_file(path)
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_returns_relevant_file_for_unreadable(self):
        result = _detect_framework_in_file("/nonexistent/agent.py")
        self.assertIsNotNone(result)
        self.assertIsNone(result.framework)


class TestIsRelevantFile(unittest.TestCase):
    def test_requirements_txt(self):
        self.assertTrue(is_relevant_file("requirements.txt"))

    def test_non_python_file(self):
        self.assertFalse(is_relevant_file("readme.md"))

    def test_python_file(self):
        self.assertTrue(is_relevant_file("agent.py"))


class TestIsRelevantPythonFile(unittest.TestCase):
    def test_with_relevant_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import openai\n")
            path = f.name
        try:
            self.assertTrue(is_relevant_python_file(path))
        finally:
            os.unlink(path)

    def test_without_relevant_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import os\n")
            path = f.name
        try:
            self.assertFalse(is_relevant_python_file(path))
        finally:
            os.unlink(path)

    def test_nonexistent_file_returns_true(self):
        self.assertTrue(is_relevant_python_file("/nonexistent/path.py"))


class TestFindRelevantFiles(unittest.TestCase):
    def test_finds_python_files_with_imports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "agent.py")
            with open(py_path, "w") as f:
                f.write("import openai\nx = 1\n")
            result = find_relevant_files(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], RelevantFile)
            self.assertEqual(result[0].path, py_path)
            self.assertEqual(result[0].framework, "OpenAI")
            self.assertIn("openai", result[0].evidence)

    def test_finds_requirements_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("flask==2.0\n")
            result = find_relevant_files(tmpdir)
            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0], RelevantFile)
            self.assertIsNone(result[0].framework)

    def test_skips_irrelevant_python(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "utils.py")
            with open(py_path, "w") as f:
                f.write("import os\n")
            result = find_relevant_files(tmpdir)
            self.assertEqual(len(result), 0)

    def test_skips_init_py(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init_path = os.path.join(tmpdir, "__init__.py")
            with open(init_path, "w") as f:
                f.write("import openai\n")
            result = find_relevant_files(tmpdir)
            self.assertEqual(len(result), 0)

    def test_single_python_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "agent.py")
            with open(py_path, "w") as f:
                f.write("import openai\nx = 1\n")
            result = find_relevant_files(py_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].path, py_path)
            self.assertEqual(result[0].type, FileType.PYTHON)
            self.assertEqual(result[0].framework, "OpenAI")

    def test_single_requirements_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = os.path.join(tmpdir, "requirements.txt")
            with open(req_path, "w") as f:
                f.write("openai==1.0\n")
            result = find_relevant_files(req_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].path, req_path)
            self.assertEqual(result[0].type, FileType.REQUIREMENTS)
            self.assertIsNone(result[0].framework)

    def test_single_python_file_without_framework_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            py_path = os.path.join(tmpdir, "utils.py")
            with open(py_path, "w") as f:
                f.write("import os\n")
            self.assertEqual(find_relevant_files(py_path), [])

    def test_single_non_python_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            md_path = os.path.join(tmpdir, "readme.md")
            with open(md_path, "w") as f:
                f.write("import openai\n")
            self.assertEqual(find_relevant_files(md_path), [])

    def test_single_init_py_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            init_path = os.path.join(tmpdir, "__init__.py")
            with open(init_path, "w") as f:
                f.write("import openai\n")
            self.assertEqual(find_relevant_files(init_path), [])

    def test_relative_path_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "agent.py"), "w") as f:
                f.write("import openai\n")
            nested = os.path.join(tmpdir, "sub", "deeper")
            os.makedirs(nested)
            cwd = os.getcwd()
            os.chdir(nested)
            try:
                result = find_relevant_files("../../agent.py")
            finally:
                os.chdir(cwd)
            self.assertEqual(len(result), 1)
            self.assertTrue(os.path.isabs(result[0].path))
            self.assertEqual(os.path.basename(result[0].path), "agent.py")

    def test_multiple_files_different_frameworks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "a.py"), "w") as f:
                f.write("import openai\n")
            with open(os.path.join(tmpdir, "b.py"), "w") as f:
                f.write("from anthropic import Anthropic\n")
            result = find_relevant_files(tmpdir)
            self.assertEqual(len(result), 2)
            frameworks = {r.framework for r in result}
            self.assertEqual(frameworks, {"OpenAI", "Anthropic"})


if __name__ == "__main__":
    unittest.main()
