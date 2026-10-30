from __future__ import annotations

from enum import Enum


class EvaluationApproach(str, Enum):
    """Whether an evaluation actively probes the model
    or passively measures a quality metric."""

    PROBE = "Probe"
    METRIC = "Metric"
