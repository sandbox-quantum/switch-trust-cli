import unittest
from unittest.mock import patch

from switch_trust.eval.db.base.message.message_collection_helpers import (
    create_message_collection,
)
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionType,
)


class TestCreateCSV(unittest.TestCase):
    @patch(
        "switch_trust.eval.core.message.message_collection_csv.CsvMessageCollection",
    )
    def test_creates_with_filename_and_column(self, MockCSV):
        db = DbMessageCollection(
            type=MessageCollectionType.CSV,
            name="test",
            filename="data.csv",
            column="text",
        )
        result = create_message_collection(db)

        MockCSV.assert_called_once_with(
            filename="data.csv",
            column="text",
        )
        self.assertEqual(result, MockCSV.return_value)

    @patch(
        "switch_trust.eval.core.message.message_collection_csv.CsvMessageCollection",
    )
    def test_defaults_column_to_prompt(self, MockCSV):
        db = DbMessageCollection(
            type=MessageCollectionType.CSV,
            name="test",
            filename="data.csv",
            column="prompt",
        )
        create_message_collection(db)

        MockCSV.assert_called_once_with(
            filename="data.csv",
            column="prompt",
        )

    def test_requires_filename(self):
        db = DbMessageCollection(
            type=MessageCollectionType.CSV,
            name="test",
        )
        with self.assertRaises(ValueError):
            create_message_collection(db)


if __name__ == "__main__":
    unittest.main()
