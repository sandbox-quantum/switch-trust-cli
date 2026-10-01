import csv

from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.message.message_collection import MessageCollection


class CsvMessageCollection(MessageCollection):
    """A message collection that reads prompts from a CSV file."""

    def __init__(self, filename: str, column: str):
        self._filename = filename
        self._column = column
        self._messages: list[Message] | None = None

    def _ensure_loaded(self) -> list[Message]:
        if self._messages is None:
            with open(self._filename, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if self._column not in (reader.fieldnames or []):
                    raise ValueError(
                        f"Column {self._column!r} not found in "
                        f"{self._filename!r}. "
                        f"Available: {reader.fieldnames}"
                    )
                self._messages = [
                    Message(content=Content.text(Role.USER, row[self._column]))
                    for row in reader
                ]
        return self._messages

    def get(self, id: str) -> Message:
        for message in self._ensure_loaded():
            if message.id == id:
                return message
        raise KeyError(f"Message '{id}' does not exist")

    def load(self) -> list[Message]:
        return list(self._ensure_loaded())

    def size(self) -> int:
        return len(self._ensure_loaded())

    def save(self, messages: list[Message]) -> None:
        raise NotImplementedError("CsvMessageCollection is read-only")
