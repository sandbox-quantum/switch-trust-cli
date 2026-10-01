from switch_trust.eval.common.schema import Content, Message, Role
from switch_trust.eval.core.message.message_collection import MessageCollection
from switch_trust.eval.core.optional_deps import require_garak


def _garak_prompt_to_message(prompt) -> Message:
    garak = require_garak()
    if isinstance(prompt, str):
        text = prompt
    elif isinstance(prompt, garak.attempt.Message):
        text = prompt.text
    elif isinstance(prompt, garak.attempt.Conversation):
        parts = [turn.content.text for turn in prompt.turns]
        text = "\n".join(parts)
    else:
        text = str(prompt)
    return Message(content=Content.text(Role.USER, text))


class GarakMessageCollection(MessageCollection):
    """A message collection backed by a garak probe's prompts."""

    def __init__(self, probe_name: str):
        self._probe_name = probe_name
        self._messages: list[Message] | None = None

    def _ensure_loaded(self) -> list[Message]:
        if self._messages is None:
            probe = require_garak()._plugins.load_plugin(self._probe_name)
            self._messages = [_garak_prompt_to_message(p) for p in probe.prompts]
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
        raise NotImplementedError("GarakMessageCollection is read-only")
