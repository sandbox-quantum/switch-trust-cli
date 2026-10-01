"""
Read-only JSON-backed repositories for CLI evaluation runs.

Reads a single JSON configuration file and provides repository
implementations for models, evaluations, detectors, message
collections, and model evaluations. All write operations raise
NotImplementedError.
"""

from __future__ import annotations

import json
import os

from switch_trust.eval.common.reference import Reference
from switch_trust.eval.common.utils import strip_nulls
from switch_trust.eval.core.message.message_collection import MessageCollection
from switch_trust.eval.db.base.detectors.detector_repository import DetectorRepository
from switch_trust.eval.db.base.detectors.detector_types import (
    DbDetector,
    DetectorListView,
    DetectorSortOrder,
)
from switch_trust.eval.db.base.eval.eval_repository import EvaluationRepository
from switch_trust.eval.db.base.eval.eval_types import (
    DbEvaluation,
    EvaluationListView,
    EvaluationSortOrder,
)
from switch_trust.eval.db.base.eval.model_eval_repository import (
    ModelEvaluationRepository,
)
from switch_trust.eval.db.base.eval.model_eval_types import (
    DbModelEvaluation,
    ModelEvaluationListItem,
    ModelEvaluationListView,
    ModelEvaluationSortOrder,
)
from switch_trust.eval.db.base.message.message_collection_helpers import (
    create_message_collection,
)
from switch_trust.eval.db.base.message.message_collection_repository import (
    MessageCollectionRepository,
)
from switch_trust.eval.db.base.message.message_collection_types import (
    DbMessageCollection,
    MessageCollectionListView,
    MessageCollectionSortOrder,
    MessageCollectionType,
)
from switch_trust.eval.db.base.models.model_repository import ModelRepository
from switch_trust.eval.db.base.models.model_types import (
    DbModel,
    DbModelListView,
    ModelSortOrder,
)

_READ_ONLY = NotImplementedError("JSON store is read-only")


def _filter_by_query(
    items: list,
    query: str | None,
    name_attr: str = "name",
) -> list:
    if not query:
        return items
    q = query.lower()
    return [item for item in items if q in getattr(item, name_attr).lower()]


class JsonModelRepository(ModelRepository):
    def __init__(self, models: list[DbModel]):
        self._models = models
        self._by_id = {m.id: m for m in models}

    def list(self) -> list[DbModel]:
        return list(self._models)

    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: ModelSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> DbModelListView:
        items = _filter_by_query(self._models, query)
        if types:
            items = [m for m in items if m.type.value in types]
        total = len(items)
        items = items[offset : offset + limit]
        return DbModelListView(items=items, total=total)

    def get(self, id: str) -> DbModel:
        if id not in self._by_id:
            raise KeyError(f"Model {id!r} not found")
        return self._by_id[id]

    def create(self, config: DbModel) -> DbModel:
        raise _READ_ONLY

    def update(self, config: DbModel) -> DbModel:
        raise _READ_ONLY

    def delete(self, id: str) -> None:
        raise _READ_ONLY

    def clear(self) -> None:
        raise _READ_ONLY


class JsonEvaluationRepository(EvaluationRepository):
    def __init__(self, evaluations: list[DbEvaluation]):
        self._evaluations = evaluations
        self._by_id = {e.id: e for e in evaluations}

    def list(self) -> list[DbEvaluation]:
        return list(self._evaluations)

    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: EvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> EvaluationListView:
        items = _filter_by_query(self._evaluations, query)
        if types:
            items = [e for e in items if e.type.value in types]
        total = len(items)
        items = items[offset : offset + limit]
        return EvaluationListView(items=items, total=total)

    def get(self, id: str) -> DbEvaluation:
        if id not in self._by_id:
            raise KeyError(
                f"Evaluation {id!r} not found",
            )
        return self._by_id[id]

    def create(self, config: DbEvaluation) -> DbEvaluation:
        raise _READ_ONLY

    def update(self, config: DbEvaluation) -> DbEvaluation:
        raise _READ_ONLY

    def delete(self, id: str) -> None:
        raise _READ_ONLY

    def clear(self) -> None:
        raise _READ_ONLY


class JsonDetectorRepository(DetectorRepository):
    def __init__(self, detectors: list[DbDetector]):
        self._detectors = detectors
        self._by_id = {d.id: d for d in detectors}

    def list(self) -> list[DbDetector]:
        return list(self._detectors)

    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: DetectorSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> DetectorListView:
        items = _filter_by_query(self._detectors, query)
        if types:
            items = [d for d in items if d.type.value in types]
        total = len(items)
        items = items[offset : offset + limit]
        return DetectorListView(items=items, total=total)

    def get(self, id: str) -> DbDetector:
        if id not in self._by_id:
            raise KeyError(
                f"Detector {id!r} not found",
            )
        return self._by_id[id]

    def create(self, config: DbDetector) -> DbDetector:
        raise _READ_ONLY

    def update(self, config: DbDetector) -> DbDetector:
        raise _READ_ONLY

    def delete(self, id: str) -> None:
        raise _READ_ONLY

    def clear(self) -> None:
        raise _READ_ONLY


class JsonMessageCollectionRepository(
    MessageCollectionRepository,
):
    def __init__(
        self,
        collections: list[DbMessageCollection],
    ):
        self._collections = collections
        self._by_id = {c.id: c for c in collections}

    def list(self) -> list[DbMessageCollection]:
        return list(self._collections)

    def search(
        self,
        query: str | None = None,
        types: list[str] | None = None,
        order: MessageCollectionSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> MessageCollectionListView:
        items = _filter_by_query(
            self._collections,
            query,
        )
        if types:
            items = [c for c in items if c.type.value in types]
        total = len(items)
        items = items[offset : offset + limit]
        return MessageCollectionListView(
            items=items,
            total=total,
        )

    def get(self, id: str) -> DbMessageCollection:
        if id not in self._by_id:
            raise KeyError(
                f"MessageCollection {id!r} not found",
            )
        return self._by_id[id]

    def get_message_collection(
        self,
        id: str,
    ) -> MessageCollection:

        return create_message_collection(self.get(id))

    def create(
        self,
        db_collection: DbMessageCollection,
    ) -> DbMessageCollection:
        raise _READ_ONLY

    def update(
        self,
        db_collection: DbMessageCollection,
    ) -> DbMessageCollection:
        raise _READ_ONLY

    def delete(self, id: str) -> None:
        raise _READ_ONLY

    def clear(self) -> None:
        raise _READ_ONLY


class JsonModelEvaluationRepository(
    ModelEvaluationRepository,
):
    def __init__(
        self,
        model_evaluations: list[DbModelEvaluation],
        model_repo: JsonModelRepository,
        eval_repo: JsonEvaluationRepository,
    ):
        self._items = model_evaluations
        self._by_id = {me.id: me for me in model_evaluations}
        self._model_repo = model_repo
        self._eval_repo = eval_repo

    def _resolve_refs(
        self,
        me: DbModelEvaluation,
    ) -> ModelEvaluationListItem:
        model_ref: Reference | None = None
        eval_ref: Reference | None = None
        try:
            model_ref = self._model_repo.get(
                me.model_id,
            ).get_ref()
        except KeyError:
            pass
        try:
            eval_ref = self._eval_repo.get(
                me.evaluation_id,
            ).get_ref()
        except KeyError:
            pass
        return ModelEvaluationListItem(
            config=me,
            model_ref=model_ref,
            evaluation_ref=eval_ref,
        )

    def list_by_model(
        self,
        model_id: str,
        order: ModelEvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> ModelEvaluationListView:
        filtered = [me for me in self._items if me.model_id == model_id]
        total = len(filtered)
        page = filtered[offset : offset + limit]
        items = [self._resolve_refs(me) for me in page]
        return ModelEvaluationListView(
            items=items,
            total=total,
        )

    def list_by_evaluation(
        self,
        evaluation_id: str,
        order: ModelEvaluationSortOrder | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> ModelEvaluationListView:
        filtered = [me for me in self._items if me.evaluation_id == evaluation_id]
        total = len(filtered)
        page = filtered[offset : offset + limit]
        items = [self._resolve_refs(me) for me in page]
        return ModelEvaluationListView(
            items=items,
            total=total,
        )

    def get(self, id: str) -> DbModelEvaluation:
        if id not in self._by_id:
            raise KeyError(
                f"ModelEvaluation {id!r} not found",
            )
        return self._by_id[id]

    def list_all(self) -> list[DbModelEvaluation]:
        return list(self._items)

    def add(self, config: DbModelEvaluation) -> bool:
        """Add a model-evaluation assignment.

        Returns True if added, False if a matching
        model_id + evaluation_id pair already exists.
        """
        for existing in self._items:
            if (
                existing.model_id == config.model_id
                and existing.evaluation_id == config.evaluation_id
            ):
                return False
        self._items.append(config)
        self._by_id[config.id] = config
        return True

    def remove(
        self,
        model_id: str | None = None,
        evaluation_id: str | None = None,
    ) -> list[DbModelEvaluation]:
        """Remove assignments matching the given filters.

        At least one of model_id or evaluation_id must be
        provided. Returns the list of removed items.
        """
        if model_id is None and evaluation_id is None:
            return []

        removed: list[DbModelEvaluation] = []
        kept: list[DbModelEvaluation] = []
        for me in self._items:
            match = True
            if model_id is not None and me.model_id != model_id:
                match = False
            if evaluation_id is not None and me.evaluation_id != evaluation_id:
                match = False
            if match:
                removed.append(me)
            else:
                kept.append(me)

        self._items = kept
        self._by_id = {me.id: me for me in kept}
        return removed

    def create(
        self,
        config: DbModelEvaluation,
    ) -> DbModelEvaluation:
        raise _READ_ONLY

    def clear(self) -> None:
        raise _READ_ONLY


class JsonRepository:
    """Read-only store backed by a single JSON config file."""

    def __init__(self, path: str):
        self._path = path
        with open(path) as f:
            data = json.load(f)
        self._init_repos(data)

    def merge(self, other: JsonRepository) -> JsonRepository:
        """Return a new repository with entries from both.

        On ID conflicts, entries from ``self`` take precedence.
        """

        def _combine(self_list: list, other_list: list) -> list:
            self_ids = {item.id for item in self_list}
            combined = list(self_list)
            for item in other_list:
                if item.id not in self_ids:
                    combined.append(item)
            return combined

        merged = object.__new__(JsonRepository)
        merged._path = self._path

        models = _combine(
            self._models_repo.list(),
            other._models_repo.list(),
        )
        evals = _combine(
            self._evaluations_repo.list(),
            other._evaluations_repo.list(),
        )
        detectors = _combine(
            self._detectors_repo.list(),
            other._detectors_repo.list(),
        )
        collections = _combine(
            self._message_collections_repo.list(),
            other._message_collections_repo.list(),
        )
        assignments = _combine(
            self._model_evaluations_repo.list_all(),
            other._model_evaluations_repo.list_all(),
        )

        merged._models_repo = JsonModelRepository(models)
        merged._evaluations_repo = JsonEvaluationRepository(evals)
        merged._detectors_repo = JsonDetectorRepository(detectors)
        merged._message_collections_repo = JsonMessageCollectionRepository(collections)
        merged._model_evaluations_repo = JsonModelEvaluationRepository(
            assignments,
            model_repo=merged._models_repo,
            eval_repo=merged._evaluations_repo,
        )
        return merged

    def _init_repos(self, data: dict) -> None:
        config_dir = os.path.dirname(
            os.path.abspath(self._path),
        )

        self._models_repo = JsonModelRepository(
            [DbModel.from_dict(m) for m in data.get("models", [])]
        )
        self._evaluations_repo = JsonEvaluationRepository(
            [DbEvaluation.from_dict(e) for e in data.get("evaluations", [])]
        )
        detectors: list[DbDetector] = []
        for d_dict in data.get("detectors", []):
            prompt_file = d_dict.pop("prompt_file", None)
            if prompt_file:
                if not os.path.isabs(prompt_file):
                    prompt_file = os.path.join(
                        config_dir,
                        prompt_file,
                    )
                with open(
                    prompt_file,
                    encoding="utf-8",
                ) as f:
                    d_dict["prompt"] = f.read()
            d = DbDetector.from_dict(d_dict)
            detectors.append(d)
        self._detectors_repo = JsonDetectorRepository(detectors)

        collections: list[DbMessageCollection] = []
        for mc_dict in data.get("message_collections", []):
            mc = DbMessageCollection.from_dict(mc_dict)
            if (
                mc.type == MessageCollectionType.CSV
                and mc.filename
                and not os.path.isabs(mc.filename)
            ):
                mc.filename = os.path.join(
                    config_dir,
                    mc.filename,
                )
            collections.append(mc)

        self._message_collections_repo = JsonMessageCollectionRepository(collections)
        self._model_evaluations_repo = JsonModelEvaluationRepository(
            [
                DbModelEvaluation.from_dict(me)
                for me in data.get(
                    "model_evaluations",
                    [],
                )
            ],
            model_repo=self._models_repo,
            eval_repo=self._evaluations_repo,
        )

    def save(self) -> None:
        """Write the current state back to the config file."""
        data: dict = {
            "models": [m.to_dict() for m in self._models_repo.list()],
            "evaluations": [e.to_dict() for e in self._evaluations_repo.list()],
            "detectors": [d.to_dict() for d in self._detectors_repo.list()],
            "message_collections": [
                c.to_dict() for c in self._message_collections_repo.list()
            ],
            "model_evaluations": [
                me.to_dict() for me in self._model_evaluations_repo.list_all()
            ],
        }
        data = strip_nulls(data)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @property
    def path(self) -> str:
        return self._path

    @property
    def models(self) -> JsonModelRepository:
        return self._models_repo

    @property
    def evaluations(self) -> JsonEvaluationRepository:
        return self._evaluations_repo

    @property
    def detectors(self) -> JsonDetectorRepository:
        return self._detectors_repo

    @property
    def message_collections(
        self,
    ) -> JsonMessageCollectionRepository:
        return self._message_collections_repo

    @property
    def model_evaluations(
        self,
    ) -> JsonModelEvaluationRepository:
        return self._model_evaluations_repo
