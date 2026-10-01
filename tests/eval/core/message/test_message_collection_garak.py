import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.core.message.message_collection_garak import (
    GarakMessageCollection,
)


class TestGarakMessageCollection(unittest.TestCase):
    @patch("garak._plugins.load_plugin")
    def test_load_returns_messages(self, mock_load):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt 1", "prompt 2"]
        mock_load.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        messages = collection.load()

        self.assertEqual(len(messages), 2)
        self.assertEqual(
            messages[0].content.parts[0].text,
            "prompt 1",
        )
        mock_load.assert_called_once_with(
            "probes.test.Probe",
        )

    @patch("garak._plugins.load_plugin")
    def test_load_caches_messages(self, mock_load):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt"]
        mock_load.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        collection.load()
        collection.load()

        mock_load.assert_called_once()

    @patch("garak._plugins.load_plugin")
    def test_get_existing(self, mock_load):
        mock_probe = MagicMock()
        mock_probe.prompts = ["prompt"]
        mock_load.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        messages = collection.load()
        msg = collection.get(messages[0].id)

        self.assertEqual(
            msg.content.parts[0].text,
            "prompt",
        )

    @patch("garak._plugins.load_plugin")
    def test_get_missing_raises(self, mock_load):
        mock_probe = MagicMock()
        mock_probe.prompts = []
        mock_load.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        with self.assertRaises(KeyError):
            collection.get("no-such-id")

    @patch("garak._plugins.load_plugin")
    def test_save_raises(self, mock_load):
        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        with self.assertRaises(NotImplementedError):
            collection.save([])

    @patch("garak._plugins.load_plugin")
    def test_size_returns_count(self, mock_load):
        mock_probe = MagicMock()
        mock_probe.prompts = ["a", "b", "c"]
        mock_load.return_value = mock_probe

        collection = GarakMessageCollection(
            probe_name="probes.test.Probe",
        )
        self.assertEqual(collection.size(), 3)


if __name__ == "__main__":
    unittest.main()
