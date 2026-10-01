from switch_trust.eval.common.schema import Message
from switch_trust.eval.core.message.message_collection import MessageCollection
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionType,
)


def create_message_collection(db_collection: DbMessageCollection) -> MessageCollection:
    """Create a MessageCollection instance from a DbMessageCollection."""
    if db_collection.type == MessageCollectionType.IN_MEMORY:
        from switch_trust.eval.core.message.message_collection_memory import (  # noqa: PLC0415 - mirrors the eval/db original
            InMemoryMessageCollection,
        )

        if db_collection.prompts:
            return InMemoryMessageCollection().from_strings(
                db_collection.prompts,
            )

        messages = None
        if db_collection.messages:
            messages = [Message.from_dict(m) for m in db_collection.messages]
        return InMemoryMessageCollection(messages=messages)

    elif db_collection.type == MessageCollectionType.CSV:
        from switch_trust.eval.core.message.message_collection_csv import (  # noqa: PLC0415 - mirrors the eval/db original
            CsvMessageCollection,
        )

        if not db_collection.filename:
            raise ValueError("filename is required for csv collections")
        return CsvMessageCollection(
            filename=db_collection.filename,
            column=db_collection.column or "prompt",
        )

    elif db_collection.type == MessageCollectionType.GARAK:
        from switch_trust.eval.core.message.message_collection_garak import (  # noqa: PLC0415 - mirrors the eval/db original
            GarakMessageCollection,
        )

        if not db_collection.probe_name:
            raise ValueError("probe_name is required for garak collections")
        return GarakMessageCollection(
            probe_name=db_collection.probe_name,
        )

    else:
        raise ValueError(f"unknown collection type: {db_collection.type}")
