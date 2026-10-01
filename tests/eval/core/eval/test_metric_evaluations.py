"""Tests for metric evaluation classes.

Covers ConcisenessMetricEvaluation, FactualAccuracyMetricEvaluation,
InstructionAdherenceMetricEvaluation, ToneMetricEvaluation, and
ToxicityMetricEvaluation.
"""

import unittest
from unittest.mock import MagicMock, patch

from switch_trust.eval.core.detectors.detector_model import ModelDetector
from switch_trust.eval.core.eval.evaluation_single_prompt import (
    SinglePromptEvaluation,
)
from switch_trust.eval.core.eval.metric_conciseness import ConcisenessMetricEvaluation
from switch_trust.eval.core.eval.metric_factual_accuracy import (
    FactualAccuracyMetricEvaluation,
)
from switch_trust.eval.core.eval.metric_instruction_adherence import (
    InstructionAdherenceMetricEvaluation,
)
from switch_trust.eval.core.eval.metric_tone import ToneMetricEvaluation
from switch_trust.eval.core.eval.metric_toxicity import ToxicityMetricEvaluation


class TestConcisenessMetricEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    def test_init_stores_judge_model(self):
        judge = MagicMock()
        e = ConcisenessMetricEvaluation(judge_model=judge)
        self.assertIs(e._judge_model, judge)

    async def test_get_children_returns_correct_count(self):
        judge = MagicMock()
        e = ConcisenessMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        self.assertEqual(len(children), 100)

    async def test_children_are_single_prompt_evaluations(self):
        judge = MagicMock()
        e = ConcisenessMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    async def test_children_have_model_detector(self):
        judge = MagicMock()
        e = ConcisenessMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child.detector, ModelDetector)

    async def test_children_have_prompts(self):
        judge = MagicMock()
        e = ConcisenessMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        for child in children:
            self.assertIsNotNone(child.prompt)


class TestFactualAccuracyMetricEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    def test_init_stores_judge_model(self):
        judge = MagicMock()
        e = FactualAccuracyMetricEvaluation(judge_model=judge)
        self.assertIs(e._judge_model, judge)

    async def test_get_children_returns_correct_count(self):
        judge = MagicMock()
        e = FactualAccuracyMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        self.assertEqual(len(children), 100)

    async def test_children_are_single_prompt_evaluations(self):
        judge = MagicMock()
        e = FactualAccuracyMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    async def test_each_child_has_unique_detector(self):
        """Each QA pair should get its own detector with
        the ground-truth answer embedded in the prompt."""
        judge = MagicMock()
        e = FactualAccuracyMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        detectors = [child.detector for child in children]
        # Each child gets its own detector instance because
        # the detector prompt includes the ground-truth answer
        detector_ids = {id(d) for d in detectors}
        self.assertEqual(len(detector_ids), 100)


class TestInstructionAdherenceMetricEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    def test_init_stores_judge_model(self):
        judge = MagicMock()
        e = InstructionAdherenceMetricEvaluation(
            judge_model=judge,
        )
        self.assertIs(e._judge_model, judge)

    async def test_get_children_returns_correct_count(self):
        judge = MagicMock()
        e = InstructionAdherenceMetricEvaluation(
            judge_model=judge,
        )
        children = await e.get_children()
        self.assertEqual(len(children), 100)

    async def test_children_are_single_prompt_evaluations(self):
        judge = MagicMock()
        e = InstructionAdherenceMetricEvaluation(
            judge_model=judge,
        )
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    async def test_children_share_single_detector(self):
        """All children should share the same detector instance
        since the detector prompt is the same for all."""
        judge = MagicMock()
        e = InstructionAdherenceMetricEvaluation(
            judge_model=judge,
        )
        children = await e.get_children()
        detectors = [child.detector for child in children]
        detector_ids = {id(d) for d in detectors}
        self.assertEqual(len(detector_ids), 1)


class TestToneMetricEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    def test_init_stores_judge_model(self):
        judge = MagicMock()
        e = ToneMetricEvaluation(judge_model=judge)
        self.assertIs(e._judge_model, judge)

    async def test_get_children_returns_correct_count(self):
        judge = MagicMock()
        e = ToneMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        self.assertEqual(len(children), 100)

    async def test_children_are_single_prompt_evaluations(self):
        judge = MagicMock()
        e = ToneMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    async def test_children_share_single_detector(self):
        judge = MagicMock()
        e = ToneMetricEvaluation(judge_model=judge)
        children = await e.get_children()
        detectors = [child.detector for child in children]
        detector_ids = {id(d) for d in detectors}
        self.assertEqual(len(detector_ids), 1)


class TestToxicityMetricEvaluation(
    unittest.IsolatedAsyncioTestCase,
):
    @patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityDetector",
    )
    def test_init_creates_instance(self, mock_cls):
        e = ToxicityMetricEvaluation()
        self.assertIsInstance(e, ToxicityMetricEvaluation)

    @patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityDetector",
    )
    async def test_get_children_returns_correct_count(
        self,
        mock_cls,
    ):
        e = ToxicityMetricEvaluation()
        children = await e.get_children()
        self.assertEqual(len(children), 100)

    @patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityDetector",
    )
    async def test_children_are_single_prompt_evaluations(
        self,
        mock_cls,
    ):
        e = ToxicityMetricEvaluation()
        children = await e.get_children()
        for child in children:
            self.assertIsInstance(child, SinglePromptEvaluation)

    @patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityDetector",
    )
    async def test_children_use_toxicity_detector(
        self,
        mock_cls,
    ):
        e = ToxicityMetricEvaluation()
        children = await e.get_children()
        mock_instance = mock_cls.return_value
        for child in children:
            self.assertIs(child.detector, mock_instance)

    @patch(
        "switch_trust.eval.core.eval.metric_toxicity.ToxicityDetector",
    )
    async def test_children_share_single_detector(
        self,
        mock_cls,
    ):
        e = ToxicityMetricEvaluation()
        await e.get_children()
        # ToxicityDetector() should be called once and shared
        mock_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
