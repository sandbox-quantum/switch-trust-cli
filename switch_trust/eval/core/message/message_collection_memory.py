from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.message.message_collection import MessageCollection


class InMemoryMessageCollection(MessageCollection):
    """A message collection backed by an in-memory list."""

    def __init__(
        self,
        messages: list[Message] | None = None,
    ):
        self._messages: list[Message] = messages or []

    def get(self, id: str) -> Message:
        for message in self._messages:
            if message.id == id:
                return message
        raise KeyError(f"Message '{id}' does not exist")

    def load(self) -> list[Message]:
        return list(self._messages)

    def size(self) -> int:
        return len(self._messages)

    def save(self, messages: list[Message]) -> None:
        self._messages = list(messages)

    def from_strings(
        self,
        prompts: list[str],
    ) -> "InMemoryMessageCollection":
        messages = [Message(content=Content.text(Role.USER, p)) for p in prompts]
        return InMemoryMessageCollection(messages)
