"""Backward-compatibility shim — imports from metric.py.

AnalyticalSpace and AnalyticalParams are now MetricSpace and MetricParams.
This module preserves the old import paths for existing code.
"""

from helmlab.spaces.metric import MetricSpace as AnalyticalSpace  # noqa: F401
from helmlab.spaces.metric import MetricParams as AnalyticalParams  # noqa: F401
from helmlab.spaces.metric import oklab_params  # noqa: F401

__all__ = ["AnalyticalSpace", "AnalyticalParams", "oklab_params"]
