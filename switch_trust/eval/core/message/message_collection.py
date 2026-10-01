"""
Abstract interface for message collections.

A MessageCollection produces a list of Message items (typically user-role
text prompts) that can be fed to a Model for evaluation.
"""

from abc import ABC, abstractmethod

from switch_trust.eval.common.schema import Message


class MessageCollection(ABC):
    """A collection of messages that can be loaded from a backing store."""

    @abstractmethod
    def get(self, id: str) -> Message:
        """Return the message with the given id.

        Raises KeyError if the message does not exist.
        """
        pass

    @abstractmethod
    def load(self) -> list[Message]:
        """Load and return all messages in the collection."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return the number of messages in the collection."""
        pass

    @abstractmethod
    def save(self, messages: list[Message]) -> None:
        """Persist a list of messages to the backing store."""
        pass
