import unittest

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.message.message_collection_memory import (
    InMemoryMessageCollection,
)


class TestInMemoryMessageCollection(unittest.TestCase):
    def test_load_returns_messages(self):
        messages = [
            Message(content=Content.text(Role.USER, "hello")),
            Message(content=Content.text(Role.USER, "world")),
        ]
        collection = InMemoryMessageCollection(messages=messages)

        loaded = collection.load()

        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "hello",
        )

    def test_save_replaces_messages(self):
        collection = InMemoryMessageCollection()
        self.assertEqual(len(collection.load()), 0)

        new_messages = [
            Message(content=Content.text(Role.USER, "saved")),
        ]
        collection.save(new_messages)

        loaded = collection.load()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "saved",
        )

    def test_from_strings(self):
        collection = InMemoryMessageCollection()
        result = collection.from_strings(
            prompts=["prompt 1", "prompt 2"],
        )

        loaded = result.load()
        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "prompt 1",
        )
        self.assertEqual(loaded[0].content.role, Role.USER)

    def test_get_returns_matching_message(self):
        messages = [
            Message(content=Content.text(Role.USER, "hello")),
            Message(content=Content.text(Role.USER, "world")),
        ]
        collection = InMemoryMessageCollection(messages=messages)
        target_id = messages[1].id

        result = collection.get(target_id)

        self.assertEqual(result.id, target_id)
        self.assertEqual(
            result.content.parts[0].text,
            "world",
        )

    def test_get_raises_for_missing_id(self):
        collection = InMemoryMessageCollection(
            messages=[
                Message(content=Content.text(Role.USER, "hi")),
            ],
        )
        with self.assertRaises(KeyError):
            collection.get("nonexistent-id")

    def test_size_returns_correct_count(self):
        collection = InMemoryMessageCollection()
        self.assertEqual(collection.size(), 0)

        messages = [
            Message(content=Content.text(Role.USER, "a")),
            Message(content=Content.text(Role.USER, "b")),
            Message(content=Content.text(Role.USER, "c")),
        ]
        collection = InMemoryMessageCollection(messages=messages)
        self.assertEqual(collection.size(), 3)


if __name__ == "__main__":
    unittest.main()
