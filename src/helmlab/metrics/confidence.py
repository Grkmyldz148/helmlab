"""Confidence layer for color-difference predictions (EXPERIMENTAL).

A standard color-difference metric returns one number: how different two colors
are. This layer adds a second number: how much *observers will disagree* about
that difference, predicted from the colors alone. It turns a bare ΔE into a
ΔE *with a calibrated reliability* — the first thing to ask before trusting a
small difference.

Provenance (honest)
-------------------
- Base metric: helmlab's perceptual distance (the v21 MetricSpace, STRESS ≈ 22.7
  on COMBVD).
- Model: ``disagreement = a·de + b·chroma + c`` (in human difference-rating
  units), fit on HumanFB (47 color-pairs × ~74 observers; disagreement = the
  std of the rating across observers). Leave-one-out R² ≈ 0.58.
- Independently validated on a second multi-observer dataset (hong_2025, 8
  subjects × 10,506 colors): the color structure replicates out-of-sample
  (held-out R² ≈ 0.26, same direction — more disagreement at low chroma).

Two validated effects (both confirmed on two independent datasets)
  1. Near-threshold: small differences → more (relative) disagreement.
  2. Low chroma → more disagreement.

Limits
------
EXPERIMENTAL. n=47 training pairs. Calibrated for the small / near-threshold
regime (perceptual ``de`` ≲ 0.15) where reliability actually matters; for large
differences it clamps to a baseline disagreement and reports "reliable". Not a
deep new mechanism — it is the Weber/threshold effect plus a chroma term, made
deployable. The exact coefficients should not be over-read.
"""
import json
import os

import numpy as np

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "confidence_params.json")


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return json.load(f)


class ConfidenceModel:
    """Predict observer disagreement (and a reliability score) for a color difference.

    Parameters
    ----------
    params : dict, optional
        Coefficients ``{a, b, c, scale, dis_floor, de_train_max, ...}``.
        Defaults to the bundled ``confidence_params.json``.
    """

    def __init__(self, params: dict | None = None):
        self.p = params if params is not None else _load_params()

    def disagreement(self, de, chroma):
        """Predicted inter-observer disagreement (std), in human rating units."""
        p = self.p
        d = p["a"] * np.asarray(de, float) + p["b"] * np.asarray(chroma, float) + p["c"]
        return np.maximum(d, p["dis_floor"])

    def assess(self, de, chroma) -> dict:
        """Full reliability assessment for a difference ``de`` at mean ``chroma``.

        Returns a dict with:
          - ``de``           : the perceptual distance passed in
          - ``de_human``     : that distance mapped to human rating units
          - ``disagreement`` : predicted observer disagreement (same units)
          - ``reliability``  : ``de_human / (de_human + disagreement)`` ∈ [0, 1)
          - ``reliable``     : True if the difference exceeds the human noise band
          - ``extrapolated`` : True if ``de`` is beyond the trained range
        """
        de = np.asarray(de, float)
        chroma = np.asarray(chroma, float)
        dis = self.disagreement(de, chroma)
        de_h = de * self.p["scale"]
        out = {
            "de": de,
            "de_human": de_h,
            "disagreement": dis,
            "reliability": de_h / (de_h + dis),
            "reliable": de_h > dis,
            "extrapolated": de > self.p["de_train_max"],
        }
        if de.ndim == 0:
            out = {k: (float(v) if isinstance(v, np.ndarray) and v.ndim == 0
                       else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                   for k, v in out.items()}
        return out
