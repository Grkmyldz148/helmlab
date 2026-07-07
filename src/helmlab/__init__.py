"""Helmlab — a data-driven analytical color space for perceptual color difference."""

# Single source of truth for the version is pyproject.toml. At runtime we read it
# from the installed package metadata so __init__ never drifts. The literal below
# is only a fallback for a source checkout that hasn't been installed; the
# scripts/bump_version.py tool keeps it in sync and CI guards it.
from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PNF

try:
    __version__ = _pkg_version("helmlab")
except _PNF:  # not installed (e.g. running straight from a source tree)
    __version__ = "0.17.0"  # bump-version-fallback

from helmlab.helmlab import Helmlab
from helmlab.spaces.metric import MetricSpace, MetricParams
from helmlab.spaces.gen import GenSpace, GenParams

__all__ = ["Helmlab", "MetricSpace", "MetricParams", "GenSpace", "GenParams"]
