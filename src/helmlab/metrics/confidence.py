"""Confidence layer for color-difference predictions (EXPERIMENTAL, v2).

A standard color-difference metric returns one number: how different two colors
are. This layer adds a second: how much *observers will disagree* about that
difference, predicted from the colors alone.

Provenance (honest, v2 — ordinal refit 2026-07-05)
---------------------------------------------------
- Base metric: helmlab's perceptual distance (v21 MetricSpace).
- HumanFB's ratings are 5-level CATEGORICAL; v1 used the std of the category
  codes as its target, which turned out to be ANTI-correlated (rho = -0.51)
  with the true latent observer spread (ceiling pile-up artifact). v2 fits an
  ordered-probit model instead: categories are thresholds ``tau`` on a latent
  difference axis; each pair gets a latent mean ``mu`` and observer-spread
  ``sigma`` (log-sigma ridge, converged).
- Runtime model: ``mu = mu_scale*de`` (Spearman 0.957 vs fitted latents) and
  ``sigma = a*de + b*chroma + c`` (LOO-R2 = 0.49, latent units).
- Relative disagreement sigma/mu falls with de (rho = -0.77) and chroma
  (-0.36): small / low-chroma differences are the unreliable regime.
- Cross-domain check (hong_2025, 8 observers): between-observer JND CV vs
  chroma-radius = +0.23, same sign as sigma's chroma term — but threshold and
  suprathreshold domains differ; do not over-read the chroma axis.

Limits
------
EXPERIMENTAL. n=47 training pairs, single uncontrolled collection session,
pairs sampled from helmlab's own zones. Calibrated for the small /
near-threshold regime (de <~ 0.15). Coefficients should not be over-read.
"""

import json
import os

import numpy as np

_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "confidence_params.json")


def _load_params() -> dict:
    with open(_PARAMS_PATH) as f:
        return json.load(f)


def _erf(x):
    """Abramowitz & Stegun 7.1.26 (max err 1.5e-7) — IDENTICAL arithmetic to the
    JS sibling so cross-language parity holds; do not swap for math.erf."""
    x = np.asarray(x, float)
    sign = np.where(x < 0, -1.0, 1.0)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return sign * y


def _phi(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / np.sqrt(2.0)))


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
        """Predicted inter-observer spread sigma (latent difference units)."""
        p = self.p
        d = p["sigma_a"] * np.asarray(de, float) + p["sigma_b"] * np.asarray(chroma, float) + p["sigma_c"]
        return np.maximum(d, p["sigma_floor"])

    def assess(self, de, chroma) -> dict:
        """Latent-model confidence for a perceptual difference ``de``.

        Returns ``de``, ``latent`` (mu), ``disagreement`` (sigma),
        ``reliability`` (mu/(mu+sigma)), ``p_noticeable``
        (P(latent > tau[1]) — chance a random observer rates the pair at
        least 'moderately different'), ``reliable`` (p_noticeable > 0.5)
        and ``extrapolated``.
        """
        de = np.asarray(de, float)
        p = self.p
        mu = p["mu_scale"] * de
        sig = self.disagreement(de, chroma)
        tau2 = p["tau"][1]
        p_not = 1.0 - _phi((tau2 - mu) / sig)
        out = {
            "de": de,
            "latent": mu,
            "disagreement": sig,
            "reliability": mu / (mu + sig),
            "p_noticeable": p_not,
            "reliable": mu > tau2,
            "extrapolated": de > p["de_train_max"],
        }
        if de.ndim == 0:
            out = {k: (float(v) if isinstance(v, np.ndarray) and v.ndim == 0
                       else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                   for k, v in out.items()}
        return out
