/**
 * Confidence layer for color-difference predictions (EXPERIMENTAL, v2).
 *
 * v2 = ordered-probit refit of HumanFB (2026-07-05): the 5-level categorical
 * ratings are modelled as thresholds `TAU` on a latent difference axis; each
 * pair gets a latent mean mu and observer-spread sigma. v1's target (std of
 * category codes) was ANTI-correlated (rho = -0.51) with the true latent
 * spread — a ceiling pile-up artifact. Runtime model: mu = MU_SCALE*de
 * (Spearman 0.957), sigma = A*de + B*chroma + C (LOO-R2 0.49, latent units).
 * Relative disagreement sigma/mu falls with de (-0.77) and chroma (-0.36):
 * small / low-chroma differences remain the unreliable regime. Mirrors
 * Python `helmlab.metrics.confidence` exactly (same coefficients, same erf).
 *
 * Limits: EXPERIMENTAL, n=47 pairs, zone-conditional sampling, de <~ 0.15.
 */
const MU_SCALE = 20.827694941963827;
const A = 5.785073154058929;
const B = 1.0148932574050995;
const C = -0.12266288972117914;
const SIGMA_FLOOR = 0.35;
const TAU2 = 0.7421522316160746;
const DE_TRAIN_MAX = 0.14811473587883495;

/** 1 JND in trained-metric de units: TAU2 / MU_SCALE — the de at which the
 * median observer of the ordinal model rates a pair at least "moderately
 * different". Used by hl.metric.jnd(); mirrors Python (tau[1]/mu_scale). */
export const JND_DE = TAU2 / MU_SCALE;
const SQRT2 = Math.sqrt(2.0);

/** Abramowitz & Stegun 7.1.26 — IDENTICAL arithmetic to the Python sibling
 *  (do not swap for a different erf; cross-language parity depends on it). */
function erf(x: number): number {
  const sign = x < 0 ? -1.0 : 1.0;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + 0.3275911 * ax);
  const y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
              - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return sign * y;
}

const phi = (x: number): number => 0.5 * (1.0 + erf(x / SQRT2));

export interface Confidence {
  /** the perceptual distance itself */
  de: number;
  /** latent difference mu = MU_SCALE * de (ordinal-model units) */
  latent: number;
  /** predicted inter-observer spread sigma (latent units) */
  disagreement: number;
  /** mu / (mu + sigma), in [0, 1) */
  reliability: number;
  /** P(latent > TAU2) — chance a random observer rates the pair at least "moderately different" */
  pNoticeable: number;
  /** true when pNoticeable > 0.5 (mu > TAU2) */
  reliable: boolean;
  /** true if `de` is beyond the trained range (extrapolating) */
  extrapolated: boolean;
}

export function assessConfidence(de: number, chroma: number): Confidence {
  const mu = MU_SCALE * de;
  const sigma = Math.max(A * de + B * chroma + C, SIGMA_FLOOR);
  const pNoticeable = 1.0 - phi((TAU2 - mu) / sigma);
  return {
    de,
    latent: mu,
    disagreement: sigma,
    reliability: mu / (mu + sigma),
    pNoticeable,
    reliable: mu > TAU2,
    extrapolated: de > DE_TRAIN_MAX,
  };
}
