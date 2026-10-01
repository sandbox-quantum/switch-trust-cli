import csv
import os
import tempfile
import unittest

from switch_trust.eval.common.schema import Role
from switch_trust.eval.core.message.message_collection_csv import CsvMessageCollection


class TestCsvMessageCollection(unittest.TestCase):
    def _write_csv(
        self,
        rows: list[list[str]],
        header: list[str] | None = None,
    ) -> str:
        fd, path = tempfile.mkstemp(suffix=".csv")
        os.close(fd)
        self._temp_files.append(path)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header:
                writer.writerow(header)
            writer.writerows(rows)
        return path

    def setUp(self):
        self._temp_files: list[str] = []

    def tearDown(self):
        for path in self._temp_files:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_returns_messages(self):
        path = self._write_csv(
            [["hello"], ["world"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, column="prompt")

        loaded = collection.load()

        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "hello",
        )
        self.assertEqual(
            loaded[1].content.parts[0].text,
            "world",
        )

    def test_messages_have_user_role(self):
        path = self._write_csv(
            [["a prompt"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, "prompt")

        loaded = collection.load()

        self.assertEqual(loaded[0].content.role, Role.USER)

    def test_custom_column_name(self):
        path = self._write_csv(
            [["hello", "extra"]],
            header=["text", "other"],
        )
        collection = CsvMessageCollection(path, column="text")

        loaded = collection.load()

        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            "hello",
        )

    def test_size(self):
        path = self._write_csv(
            [["a"], ["b"], ["c"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, column="prompt")

        self.assertEqual(collection.size(), 3)

    def test_get_by_id(self):
        path = self._write_csv(
            [["hello"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, column="prompt")
        loaded = collection.load()
        msg_id = loaded[0].id

        found = collection.get(msg_id)

        self.assertEqual(
            found.content.parts[0].text,
            "hello",
        )

    def test_get_missing_id_raises(self):
        path = self._write_csv(
            [["hello"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, "prompt")

        with self.assertRaises(KeyError):
            collection.get("nonexistent-id")

    def test_missing_column_raises(self):
        path = self._write_csv(
            [["hello"]],
            header=["other_column"],
        )
        collection = CsvMessageCollection(
            path,
            column="prompt",
        )

        with self.assertRaises(ValueError):
            collection.load()

    def test_missing_file_raises(self):
        collection = CsvMessageCollection("/nonexistent/path.csv", "prompt")

        with self.assertRaises(FileNotFoundError):
            collection.load()

    def test_save_raises(self):
        path = self._write_csv(
            [["hello"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, "prompt")

        with self.assertRaises(NotImplementedError):
            collection.save([])

    def test_load_returns_copy(self):
        path = self._write_csv(
            [["hello"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, "prompt")

        loaded = collection.load()
        loaded.clear()

        self.assertEqual(len(collection.load()), 1)

    def test_prompts_with_commas_and_quotes(self):
        path = self._write_csv(
            [['She said, "hello"'], ["a, b, c"]],
            header=["prompt"],
        )
        collection = CsvMessageCollection(path, "prompt")

        loaded = collection.load()

        self.assertEqual(len(loaded), 2)
        self.assertEqual(
            loaded[0].content.parts[0].text,
            'She said, "hello"',
        )
        self.assertEqual(
            loaded[1].content.parts[0].text,
            "a, b, c",
        )


if __name__ == "__main__":
    unittest.main()
