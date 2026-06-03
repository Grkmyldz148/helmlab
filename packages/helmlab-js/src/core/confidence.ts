/**
 * Confidence layer for color-difference predictions (EXPERIMENTAL / beta).
 *
 * A standard metric returns one number: how different two colors are. This adds
 * a second: how much observers will disagree about that difference, predicted
 * from the colors alone — a calibrated reliability on the difference.
 *
 * Model: `disagreement = A·de + B·chroma + C` (human rating units), fit on
 * HumanFB (47 color-pairs × ~74 observers; LOO R² ≈ 0.58) and validated
 * out-of-sample on a second multi-observer dataset (hong_2025, held-out
 * R² ≈ 0.26, same direction: more disagreement at low chroma). Mirrors the
 * Python `helmlab.metrics.confidence` model exactly (same coefficients).
 *
 * Limits: EXPERIMENTAL, n=47 training pairs, calibrated for the small /
 * near-threshold regime (de ≲ 0.15) where reliability matters.
 */
const A = -98.243;
const B = -11.538;
const C = 30.716;
const SCALE = 591.15; // maps perceptual `de` → human rating units
const DIS_FLOOR = 4.76;
const DE_TRAIN_MAX = 0.1484;

export interface Confidence {
  /** the perceptual distance itself */
  de: number;
  /** that distance mapped to human rating units */
  deHuman: number;
  /** predicted inter-observer disagreement (std, same units as deHuman) */
  disagreement: number;
  /** deHuman / (deHuman + disagreement), in [0, 1) */
  reliability: number;
  /** true if the difference exceeds the human noise band */
  reliable: boolean;
  /** true if `de` is beyond the trained range (extrapolating) */
  extrapolated: boolean;
}

/** Assess how reliable a difference `de` at mean `chroma` is. */
export function assessConfidence(de: number, chroma: number): Confidence {
  const disagreement = Math.max(A * de + B * chroma + C, DIS_FLOOR);
  const deHuman = de * SCALE;
  return {
    de,
    deHuman,
    disagreement,
    reliability: deHuman / (deHuman + disagreement),
    reliable: deHuman > disagreement,
    extrapolated: de > DE_TRAIN_MAX,
  };
}
