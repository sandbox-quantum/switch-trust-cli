import json
import os
import tempfile
import unittest

from switch_trust.eval.db.base.eval.model_eval_types import DbModelEvaluation
from switch_trust.eval.db.base.models.model_types import DbModel
from switch_trust.eval.db.json.repository_json import JsonRepository

_SAMPLE_CONFIG = {
    "models": [
        {
            "id": "m1",
            "type": "gemini",
            "name": "test-model",
            "model_name": "gemini-2.5-flash-lite",
            "tags": {"tier": "Fast"},
        },
        {
            "id": "m2",
            "type": "openai",
            "name": "gpt-model",
            "model_name": "gpt-4o-mini",
            "tags": {},
        },
    ],
    "evaluations": [
        {
            "id": "e1",
            "type": "adversarial_probe",
            "name": "Test probe",
            "approach": "Probe",
            "adversarial_goals": ["test goal"],
            "num_prompts": 5,
            "max_turns": 3,
            "tags": {"category": "test"},
        },
    ],
    "detectors": [
        {
            "id": "d1",
            "type": "garak",
            "name": "Test detector",
            "detector_name": "detectors.unsafe_content.ToxicCommentModel",
        },
    ],
    "message_collections": [
        {
            "id": "mc1",
            "type": "in-memory",
            "name": "Test messages",
            "messages": [
                {
                    "content": {
                        "role": "user",
                        "parts": [{"text": "hello"}],
                    },
                },
            ],
        },
    ],
    "model_evaluations": [
        {
            "id": "me1",
            "model_id": "m1",
            "evaluation_id": "e1",
            "name": "test-model / Test probe",
        },
    ],
}


def _write_config(data: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


class TestJsonRepository(unittest.TestCase):
    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_models_list(self):
        models = self.store.models.list()
        self.assertEqual(len(models), 2)
        self.assertIsInstance(models[0], DbModel)
        self.assertEqual(models[0].id, "m1")
        self.assertEqual(models[0].name, "test-model")

    def test_models_get(self):
        model = self.store.models.get("m1")
        self.assertEqual(model.name, "test-model")

    def test_models_get_missing(self):
        with self.assertRaises(KeyError):
            self.store.models.get("missing")

    def test_models_search(self):
        result = self.store.models.search(query="gpt")
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0].name, "gpt-model")

    def test_models_search_by_type(self):
        result = self.store.models.search(
            types=["gemini"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(
            result.items[0].name,
            "test-model",
        )

    def test_models_search_pagination(self):
        result = self.store.models.search(
            offset=1,
            limit=1,
        )
        self.assertEqual(result.total, 2)
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].id, "m2")

    def test_evaluations_list(self):
        evals = self.store.evaluations.list()
        self.assertEqual(len(evals), 1)
        self.assertEqual(evals[0].id, "e1")

    def test_evaluations_get(self):
        e = self.store.evaluations.get("e1")
        self.assertEqual(e.name, "Test probe")

    def test_detectors_list(self):
        detectors = self.store.detectors.list()
        self.assertEqual(len(detectors), 1)
        self.assertEqual(detectors[0].id, "d1")

    def test_message_collections_list(self):
        cols = self.store.message_collections.list()
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0].id, "mc1")

    def test_model_evaluations_list_all(self):
        items = self.store.model_evaluations.list_all()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].id, "me1")

    def test_model_evaluations_get(self):
        me = self.store.model_evaluations.get("me1")
        self.assertEqual(me.name, "test-model / Test probe")

    def test_model_evaluations_list_by_model(self):
        view = self.store.model_evaluations.list_by_model(
            "m1",
        )
        self.assertEqual(view.total, 1)
        self.assertEqual(
            view.items[0].config.id,
            "me1",
        )
        self.assertIsNotNone(view.items[0].model_ref)
        self.assertIsNotNone(view.items[0].evaluation_ref)

    def test_model_evaluations_list_by_model_empty(self):
        view = self.store.model_evaluations.list_by_model(
            "m2",
        )
        self.assertEqual(view.total, 0)

    def test_model_evaluations_list_by_evaluation(self):
        view = self.store.model_evaluations.list_by_evaluation("e1")
        self.assertEqual(view.total, 1)


class TestJsonRepositoryReadOnly(unittest.TestCase):
    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_write_operations_raise(self):
        with self.assertRaises(NotImplementedError):
            self.store.models.create(
                DbModel(
                    type="gemini",
                    name="x",
                    model_name="x",
                ),
            )
        with self.assertRaises(NotImplementedError):
            self.store.evaluations.create(
                self.store.evaluations.get("e1"),
            )
        with self.assertRaises(NotImplementedError):
            self.store.detectors.create(
                self.store.detectors.get("d1"),
            )
        with self.assertRaises(NotImplementedError):
            self.store.message_collections.create(
                self.store.message_collections.get("mc1"),
            )
        with self.assertRaises(NotImplementedError):
            self.store.model_evaluations.create(
                self.store.model_evaluations.get("me1"),
            )


class TestJsonModelEvaluationAdd(unittest.TestCase):
    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_add_new_assignment(self):
        me = DbModelEvaluation(
            model_id="m2",
            evaluation_id="e1",
            name="gpt-model / Test probe",
        )
        added = self.store.model_evaluations.add(me)
        self.assertTrue(added)
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            2,
        )

    def test_add_duplicate_skipped(self):
        me = DbModelEvaluation(
            model_id="m1",
            evaluation_id="e1",
            name="duplicate",
        )
        added = self.store.model_evaluations.add(me)
        self.assertFalse(added)
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            1,
        )

    def test_add_same_model_different_eval(self):
        config = dict(_SAMPLE_CONFIG)
        config["evaluations"] = list(
            _SAMPLE_CONFIG["evaluations"],
        ) + [
            {
                "id": "e2",
                "type": "metric_toxicity",
                "name": "Toxicity",
                "approach": "Metric",
            },
        ]
        path = _write_config(config)
        try:
            store = JsonRepository(path)
            me = DbModelEvaluation(
                model_id="m1",
                evaluation_id="e2",
                name="test-model / Toxicity",
            )
            self.assertTrue(store.model_evaluations.add(me))
            self.assertEqual(
                len(store.model_evaluations.list_all()),
                2,
            )
        finally:
            os.unlink(path)


class TestJsonModelEvaluationRemove(unittest.TestCase):
    def setUp(self):
        config = dict(_SAMPLE_CONFIG)
        config["model_evaluations"] = [
            {
                "id": "me1",
                "model_id": "m1",
                "evaluation_id": "e1",
                "name": "m1/e1",
            },
            {
                "id": "me2",
                "model_id": "m2",
                "evaluation_id": "e1",
                "name": "m2/e1",
            },
        ]
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_remove_by_model_and_eval(self):
        removed = self.store.model_evaluations.remove(
            model_id="m1",
            evaluation_id="e1",
        )
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].id, "me1")
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            1,
        )

    def test_remove_by_model_only(self):
        removed = self.store.model_evaluations.remove(
            model_id="m1",
        )
        self.assertEqual(len(removed), 1)
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            1,
        )

    def test_remove_by_eval_only(self):
        removed = self.store.model_evaluations.remove(
            evaluation_id="e1",
        )
        self.assertEqual(len(removed), 2)
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            0,
        )

    def test_remove_no_match(self):
        removed = self.store.model_evaluations.remove(
            model_id="missing",
        )
        self.assertEqual(len(removed), 0)
        self.assertEqual(
            len(self.store.model_evaluations.list_all()),
            2,
        )

    def test_remove_no_filters_returns_empty(self):
        removed = self.store.model_evaluations.remove()
        self.assertEqual(len(removed), 0)


class TestJsonRepositorySave(unittest.TestCase):
    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_save_persists_added_assignment(self):
        me = DbModelEvaluation(
            id="me-new",
            model_id="m2",
            evaluation_id="e1",
            name="gpt-model / Test probe",
        )
        self.store.model_evaluations.add(me)
        self.store.save()

        reloaded = JsonRepository(self._path)
        self.assertEqual(
            len(reloaded.model_evaluations.list_all()),
            2,
        )
        self.assertIsNotNone(
            reloaded.model_evaluations.get("me-new"),
        )

    def test_save_persists_removed_assignment(self):
        self.store.model_evaluations.remove(model_id="m1")
        self.store.save()

        reloaded = JsonRepository(self._path)
        self.assertEqual(
            len(reloaded.model_evaluations.list_all()),
            0,
        )


class TestJsonRepositoryEmptyConfig(unittest.TestCase):
    def setUp(self):
        self._path = _write_config({})
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_empty_lists(self):
        self.assertEqual(self.store.models.list(), [])
        self.assertEqual(self.store.evaluations.list(), [])
        self.assertEqual(self.store.detectors.list(), [])
        self.assertEqual(
            self.store.message_collections.list(),
            [],
        )
        self.assertEqual(
            self.store.model_evaluations.list_all(),
            [],
        )


class TestJsonModelRepositoryReadOnly(unittest.TestCase):
    """All write operations on JsonModelRepository raise NotImplementedError."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_update_raises(self):
        model = self.store.models.get("m1")
        with self.assertRaises(NotImplementedError):
            self.store.models.update(model)

    def test_delete_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.models.delete("m1")

    def test_clear_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.models.clear()


class TestJsonEvaluationRepositoryReadOnly(unittest.TestCase):
    """All write operations on JsonEvaluationRepository raise NotImplementedError."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_update_raises(self):
        ev = self.store.evaluations.get("e1")
        with self.assertRaises(NotImplementedError):
            self.store.evaluations.update(ev)

    def test_delete_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.evaluations.delete("e1")

    def test_clear_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.evaluations.clear()


class TestJsonDetectorRepositoryReadOnly(unittest.TestCase):
    """All write operations on JsonDetectorRepository raise NotImplementedError."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_update_raises(self):
        det = self.store.detectors.get("d1")
        with self.assertRaises(NotImplementedError):
            self.store.detectors.update(det)

    def test_delete_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.detectors.delete("d1")

    def test_clear_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.detectors.clear()


class TestJsonMessageCollectionRepositoryReadOnly(unittest.TestCase):
    """All write operations on JsonMessageCollectionRepository raise NotImplementedError."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_update_raises(self):
        mc = self.store.message_collections.get("mc1")
        with self.assertRaises(NotImplementedError):
            self.store.message_collections.update(mc)

    def test_delete_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.message_collections.delete("mc1")

    def test_clear_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.message_collections.clear()


class TestJsonModelEvaluationRepositoryReadOnly(unittest.TestCase):
    """Write operations on JsonModelEvaluationRepository raise NotImplementedError."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_clear_raises(self):
        with self.assertRaises(NotImplementedError):
            self.store.model_evaluations.clear()


class TestJsonEvaluationSearch(unittest.TestCase):
    """Test search with type filter and get for evaluations."""

    def setUp(self):
        config = dict(_SAMPLE_CONFIG)
        config["evaluations"] = [
            {
                "id": "e1",
                "type": "adversarial_probe",
                "name": "Test probe",
                "approach": "Probe",
            },
            {
                "id": "e2",
                "type": "metric_toxicity",
                "name": "Toxicity",
                "approach": "Metric",
            },
        ]
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_search_by_type(self):
        result = self.store.evaluations.search(
            types=["metric_toxicity"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0].id, "e2")

    def test_search_by_type_no_match(self):
        result = self.store.evaluations.search(
            types=["garak_probe"],
        )
        self.assertEqual(result.total, 0)

    def test_search_by_query_and_type(self):
        result = self.store.evaluations.search(
            query="tox",
            types=["metric_toxicity"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0].name, "Toxicity")

    def test_get_existing(self):
        ev = self.store.evaluations.get("e1")
        self.assertEqual(ev.name, "Test probe")

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.evaluations.get("missing")


class TestJsonDetectorSearch(unittest.TestCase):
    """Test search with type filter and get for detectors."""

    def setUp(self):
        config = dict(_SAMPLE_CONFIG)
        config["detectors"] = [
            {
                "id": "d1",
                "type": "garak",
                "name": "Garak detector",
                "detector_name": "detectors.foo",
            },
            {
                "id": "d2",
                "type": "model",
                "name": "Model detector",
            },
        ]
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_search_by_type(self):
        result = self.store.detectors.search(
            types=["garak"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0].id, "d1")

    def test_search_by_type_no_match(self):
        result = self.store.detectors.search(
            types=["pii"],
        )
        self.assertEqual(result.total, 0)

    def test_get_existing(self):
        det = self.store.detectors.get("d1")
        self.assertEqual(det.name, "Garak detector")

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.detectors.get("missing")


class TestJsonMessageCollectionSearch(unittest.TestCase):
    """Test search with type filter, get, and get_message_collection."""

    def setUp(self):
        config = dict(_SAMPLE_CONFIG)
        config["message_collections"] = [
            {
                "id": "mc1",
                "type": "in-memory",
                "name": "Memory messages",
                "prompts": ["hello", "world"],
            },
            {
                "id": "mc2",
                "type": "garak",
                "name": "Garak messages",
                "probe_name": "probes.test.TestProbe",
            },
        ]
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_search_by_type(self):
        result = self.store.message_collections.search(
            types=["in-memory"],
        )
        self.assertEqual(result.total, 1)
        self.assertEqual(result.items[0].id, "mc1")

    def test_search_by_type_no_match(self):
        result = self.store.message_collections.search(
            types=["postgres"],
        )
        self.assertEqual(result.total, 0)

    def test_get_existing(self):
        mc = self.store.message_collections.get("mc1")
        self.assertEqual(mc.name, "Memory messages")

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.message_collections.get("missing")

    def test_get_message_collection_in_memory(self):
        from switch_trust.eval.core.message.message_collection import (
            MessageCollection,
        )

        mc = self.store.message_collections.get_message_collection("mc1")
        self.assertIsInstance(mc, MessageCollection)


class TestJsonModelEvaluationResolveRefs(unittest.TestCase):
    """Test _resolve_refs handles missing model/eval refs gracefully."""

    def setUp(self):
        config = {
            "models": [
                {
                    "id": "m1",
                    "type": "openai",
                    "name": "existing-model",
                    "model_name": "gpt-4o-mini",
                },
            ],
            "evaluations": [
                {
                    "id": "e1",
                    "type": "adversarial_probe",
                    "name": "existing-eval",
                    "approach": "Probe",
                },
            ],
            "detectors": [],
            "message_collections": [],
            "model_evaluations": [
                {
                    "id": "me1",
                    "model_id": "m1",
                    "evaluation_id": "e1",
                    "name": "valid refs",
                },
                {
                    "id": "me2",
                    "model_id": "missing-model",
                    "evaluation_id": "e1",
                    "name": "missing model ref",
                },
                {
                    "id": "me3",
                    "model_id": "m1",
                    "evaluation_id": "missing-eval",
                    "name": "missing eval ref",
                },
                {
                    "id": "me4",
                    "model_id": "missing-model",
                    "evaluation_id": "missing-eval",
                    "name": "both refs missing",
                },
            ],
        }
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_valid_refs_resolved(self):
        view = self.store.model_evaluations.list_by_model("m1")
        item = [i for i in view.items if i.config.id == "me1"][0]
        self.assertIsNotNone(item.model_ref)
        self.assertIsNotNone(item.evaluation_ref)

    def test_missing_model_ref(self):
        view = self.store.model_evaluations.list_by_evaluation("e1")
        item = [i for i in view.items if i.config.id == "me2"][0]
        self.assertIsNone(item.model_ref)
        self.assertIsNotNone(item.evaluation_ref)

    def test_missing_eval_ref(self):
        view = self.store.model_evaluations.list_by_model("m1")
        item = [i for i in view.items if i.config.id == "me3"][0]
        self.assertIsNotNone(item.model_ref)
        self.assertIsNone(item.evaluation_ref)

    def test_both_refs_missing(self):
        me = self.store.model_evaluations.get("me4")
        resolved = self.store.model_evaluations._resolve_refs(me)
        self.assertIsNone(resolved.model_ref)
        self.assertIsNone(resolved.evaluation_ref)


class TestJsonModelEvaluationGetMissing(unittest.TestCase):
    """Test get raises KeyError for missing model evaluation."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_get_missing_raises(self):
        with self.assertRaises(KeyError):
            self.store.model_evaluations.get("nonexistent")


class TestJsonRepositoryMerge(unittest.TestCase):
    """Test merge combines entries and self takes precedence on ID conflicts."""

    def setUp(self):
        self._config_a = {
            "models": [
                {
                    "id": "m1",
                    "type": "openai",
                    "name": "model-from-a",
                    "model_name": "gpt-4o-mini",
                },
            ],
            "evaluations": [
                {
                    "id": "e1",
                    "type": "adversarial_probe",
                    "name": "eval-from-a",
                    "approach": "Probe",
                },
            ],
            "detectors": [
                {
                    "id": "d1",
                    "type": "garak",
                    "name": "detector-from-a",
                    "detector_name": "detectors.foo",
                },
            ],
            "message_collections": [
                {
                    "id": "mc1",
                    "type": "in-memory",
                    "name": "collection-from-a",
                    "prompts": ["a"],
                },
            ],
            "model_evaluations": [
                {
                    "id": "me1",
                    "model_id": "m1",
                    "evaluation_id": "e1",
                    "name": "assignment-from-a",
                },
            ],
        }
        self._config_b = {
            "models": [
                {
                    "id": "m1",
                    "type": "openai",
                    "name": "model-from-b-CONFLICT",
                    "model_name": "gpt-4o",
                },
                {
                    "id": "m2",
                    "type": "gemini",
                    "name": "model-from-b-unique",
                    "model_name": "gemini-2.5-flash-lite",
                },
            ],
            "evaluations": [
                {
                    "id": "e2",
                    "type": "metric_toxicity",
                    "name": "eval-from-b-unique",
                    "approach": "Metric",
                },
            ],
            "detectors": [
                {
                    "id": "d2",
                    "type": "model",
                    "name": "detector-from-b-unique",
                },
            ],
            "message_collections": [
                {
                    "id": "mc2",
                    "type": "in-memory",
                    "name": "collection-from-b-unique",
                    "prompts": ["b"],
                },
            ],
            "model_evaluations": [
                {
                    "id": "me2",
                    "model_id": "m2",
                    "evaluation_id": "e2",
                    "name": "assignment-from-b-unique",
                },
            ],
        }
        self._path_a = _write_config(self._config_a)
        self._path_b = _write_config(self._config_b)

    def tearDown(self):
        os.unlink(self._path_a)
        os.unlink(self._path_b)

    def test_merge_combines_unique_entries(self):
        repo_a = JsonRepository(self._path_a)
        repo_b = JsonRepository(self._path_b)
        merged = repo_a.merge(repo_b)

        self.assertEqual(len(merged.models.list()), 2)
        self.assertEqual(len(merged.evaluations.list()), 2)
        self.assertEqual(len(merged.detectors.list()), 2)
        self.assertEqual(
            len(merged.message_collections.list()),
            2,
        )
        self.assertEqual(
            len(merged.model_evaluations.list_all()),
            2,
        )

    def test_merge_self_takes_precedence(self):
        repo_a = JsonRepository(self._path_a)
        repo_b = JsonRepository(self._path_b)
        merged = repo_a.merge(repo_b)

        # m1 should keep repo_a's name, not repo_b's
        m1 = merged.models.get("m1")
        self.assertEqual(m1.name, "model-from-a")

    def test_merge_preserves_path(self):
        repo_a = JsonRepository(self._path_a)
        repo_b = JsonRepository(self._path_b)
        merged = repo_a.merge(repo_b)
        self.assertEqual(merged.path, self._path_a)


class TestJsonRepositorySaveRoundtrip(unittest.TestCase):
    """Test save writes JSON and can be reloaded."""

    def setUp(self):
        self._path = _write_config(_SAMPLE_CONFIG)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_save_roundtrip_unchanged(self):
        original_models = len(self.store.models.list())
        original_evals = len(self.store.evaluations.list())
        self.store.save()

        reloaded = JsonRepository(self._path)
        self.assertEqual(
            len(reloaded.models.list()),
            original_models,
        )
        self.assertEqual(
            len(reloaded.evaluations.list()),
            original_evals,
        )

    def test_save_writes_valid_json(self):
        self.store.save()
        with open(self._path) as f:
            data = json.load(f)
        self.assertIn("models", data)
        self.assertIn("evaluations", data)
        self.assertIn("detectors", data)
        self.assertIn("message_collections", data)
        self.assertIn("model_evaluations", data)


class TestJsonRepositoryCsvFilenameResolution(unittest.TestCase):
    """Test that relative CSV filenames are resolved to absolute paths."""

    def setUp(self):
        config = {
            "models": [],
            "evaluations": [],
            "detectors": [],
            "message_collections": [
                {
                    "id": "mc-csv-rel",
                    "type": "csv",
                    "name": "Relative CSV",
                    "filename": "data/prompts.csv",
                    "column": "prompt",
                },
                {
                    "id": "mc-csv-abs",
                    "type": "csv",
                    "name": "Absolute CSV",
                    "filename": "/absolute/path/prompts.csv",
                    "column": "prompt",
                },
                {
                    "id": "mc-inmem",
                    "type": "in-memory",
                    "name": "In-memory (no filename)",
                    "prompts": ["test"],
                },
            ],
            "model_evaluations": [],
        }
        self._path = _write_config(config)
        self.store = JsonRepository(self._path)

    def tearDown(self):
        os.unlink(self._path)

    def test_relative_csv_path_becomes_absolute(self):
        mc = self.store.message_collections.get("mc-csv-rel")
        self.assertTrue(os.path.isabs(mc.filename))
        config_dir = os.path.dirname(
            os.path.abspath(self._path),
        )
        expected = os.path.join(config_dir, "data/prompts.csv")
        self.assertEqual(mc.filename, expected)

    def test_absolute_csv_path_unchanged(self):
        mc = self.store.message_collections.get("mc-csv-abs")
        self.assertEqual(
            mc.filename,
            "/absolute/path/prompts.csv",
        )

    def test_non_csv_type_unaffected(self):
        mc = self.store.message_collections.get("mc-inmem")
        self.assertIsNone(mc.filename)


if __name__ == "__main__":
    unittest.main()
