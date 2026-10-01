import unittest

from switch_trust.eval.core.message.message_collection_memory import (
    InMemoryMessageCollection,
)
from switch_trust.eval.db.base.message.message_collection_helpers import (
    create_message_collection,
)
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionType,
)


class TestCreateInMemoryWithPrompts(unittest.TestCase):
    def test_creates_from_strings(self):
        db = DbMessageCollection(
            type=MessageCollectionType.IN_MEMORY,
            name="test",
            prompts=["hello", "world"],
        )
        result = create_message_collection(db)

        self.assertIsInstance(result, InMemoryMessageCollection)
        loaded = result.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "hello",
        )


class TestCreateInMemoryWithMessages(unittest.TestCase):
    def test_creates_from_message_dicts(self):
        msg_dict = {
            "id": "m1",
            "content": {
                "role": "user",
                "parts": [
                    {"part_type": "text", "text": "hi"},
                ],
            },
        }
        db = DbMessageCollection(
            type=MessageCollectionType.IN_MEMORY,
            name="test",
            messages=[msg_dict],
        )
        result = create_message_collection(db)

        self.assertIsInstance(result, InMemoryMessageCollection)
        loaded = result.load()
        self.assertEqual(len(loaded), 1)


class TestCreateInMemoryEmpty(unittest.TestCase):
    def test_creates_empty_collection(self):
        db = DbMessageCollection(
            type=MessageCollectionType.IN_MEMORY,
            name="test",
        )
        result = create_message_collection(db)

        self.assertIsInstance(result, InMemoryMessageCollection)
        self.assertEqual(len(result.load()), 0)


if __name__ == "__main__":
    unittest.main()
