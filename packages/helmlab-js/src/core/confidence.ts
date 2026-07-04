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
// Full-precision coefficients from src/helmlab/data/confidence_params.json —
// rounded copies drifted disagreement ~5e-4 vs Python and put DE_TRAIN_MAX
// slightly high (0.1484 vs 0.14811…), flipping `extrapolated` in between.
const A = -98.2432184776279;
const B = -11.538177805266077;
const C = 30.716489232811135;
const SCALE = 591.1523027745784; // maps perceptual `de` → human rating units
const DIS_FLOOR = 4.759084879353265;
const DE_TRAIN_MAX = 0.14811473587883495;

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
