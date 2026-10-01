import unittest
from unittest.mock import patch

from switch_trust.eval.db.base.message.message_collection_helpers import (
    create_message_collection,
)
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionType,
)


class TestCreateGarak(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.message.message_collection_garak"
        ".GarakMessageCollection",
    )
    def test_creates_with_probe_name(self, MockGarak):
        db = DbMessageCollection(
            type=MessageCollectionType.GARAK,
            name="test",
            probe_name="encoding.InjectBase64",
        )
        result = create_message_collection(db)

        MockGarak.assert_called_once_with(
            probe_name="encoding.InjectBase64",
        )
        self.assertEqual(result, MockGarak.return_value)

    def test_requires_probe_name(self):
        db = DbMessageCollection(
            type=MessageCollectionType.GARAK,
            name="test",
        )
        with self.assertRaises(ValueError):
            create_message_collection(db)


class TestCreateUnknownType(unittest.TestCase):
    def test_raises_value_error(self):
        db = DbMessageCollection(
            type=MessageCollectionType.IN_MEMORY,
            name="test",
        )
        db.type = "unknown_type"  # type: ignore[assignment]
        with self.assertRaises(ValueError):
            create_message_collection(db)


if __name__ == "__main__":
    unittest.main()
