import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from switch_trust.eval.core.optional_deps import (
    require_garak,
    require_transformers_pipeline,
)


class TestRequireTransformersPipeline(unittest.TestCase):
    def test_missing_raises_with_install_hint(self):
        # Setting the module to None in sys.modules forces ``import
        # transformers`` to raise ImportError, simulating the package being
        # absent (i.e. the ``full`` extra not installed).
        with patch.dict(sys.modules, {"transformers": None}):
            with self.assertRaises(ImportError) as ctx:
                require_transformers_pipeline()
        self.assertIn("switch-trust-cli[full]", str(ctx.exception))

    def test_present_returns_pipeline(self):
        self.assertTrue(callable(require_transformers_pipeline()))


class TestRequireGarak(unittest.TestCase):
    def test_missing_raises_with_install_hint(self):
        with patch.dict(sys.modules, {"garak": None}):
            with self.assertRaises(ImportError) as ctx:
                require_garak()
        self.assertIn("switch-trust-cli[full]", str(ctx.exception))

    def test_present_returns_module(self):
        garak = require_garak()
        self.assertIsInstance(garak, ModuleType)
        self.assertTrue(callable(garak._plugins.load_plugin))
        self.assertTrue(callable(garak._plugins.enumerate_plugins))


if __name__ == "__main__":
    unittest.main()
