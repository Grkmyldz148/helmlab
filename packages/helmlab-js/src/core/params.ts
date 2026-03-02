import { mat3, mat3Inv, type Mat3 } from '../utils/math.js';
import defaultParams from '../data/params.json';
import defaultGenParams from '../data/gen-params.json';

/** All parameters for the analytical color space transform. */
export interface HelmlabParams {
  // Core (24 params)
  M1: number[][];
  gamma: number[];
  M2: number[][];
  hk_weight: number;
  hk_power: number;
  hk_hue_mod: number;

  // Cubic L correction (3 params)
  L_corr_p1: number;
  L_corr_p2: number;
  L_corr_p3: number;

  // Hue-dep chroma scaling (8 params: 4 harmonics)
  cs_cos1: number; cs_sin1: number;
  cs_cos2: number; cs_sin2: number;
  cs_cos3: number; cs_sin3: number;
  cs_cos4: number; cs_sin4: number;

  // L-dep chroma scaling (2 params)
  lc1: number;
  lc2: number;

  // Enhanced H-K harmonics (3 params)
  hk_sin1: number;
  hk_cos2: number;
  hk_sin2: number;

  // Hue correction (8 params: 4 harmonics)
  hue_cos1: number; hue_sin1: number;
  hue_cos2: number; hue_sin2: number;
  hue_cos3: number; hue_sin3: number;
  hue_cos4: number; hue_sin4: number;

  // HLC interaction (4 params)
  hlc_cos1: number; hlc_sin1: number;
  hlc_cos2: number; hlc_sin2: number;

  // Hue-dep lightness (4 params)
  hl_cos1: number; hl_sin1: number;
  hl_cos2: number; hl_sin2: number;

  // Chroma power (4 params)
  cp_cos1: number; cp_sin1: number;
  cp_cos2: number; cp_sin2: number;

  // Dark L compression (3 params)
  lp_dark: number;
  lp_dark_hcos: number;
  lp_dark_hsin: number;

  // Distance metric
  dist_power: number;
  dist_wC: number;
  dist_compress: number;
  dist_linear: number;
  dist_post_power: number;
  dist_sl: number;
  dist_sc: number;

  // Hue-dep L correction (2 params)
  Lh_cos1: number;
  Lh_sin1: number;
}

/** Parsed and cached params ready for compute. */
export interface CompiledParams {
  M1: Mat3;
  M1_inv: Mat3;
  M2: Mat3;
  M2_inv: Mat3;
  gamma: Float64Array;
  inv_gamma: Float64Array;
  raw: HelmlabParams;
}

export function compileParams(p: HelmlabParams): CompiledParams {
  const M1 = mat3(p.M1);
  const M2 = mat3(p.M2);
  return {
    M1, M1_inv: mat3Inv(M1),
    M2, M2_inv: mat3Inv(M2),
    gamma: new Float64Array(p.gamma),
    inv_gamma: new Float64Array(p.gamma.map(g => 1 / g)),
    raw: p,
  };
}

export function getDefaultParams(): HelmlabParams {
  return defaultParams as HelmlabParams;
}

// ── GenParams (generation-optimized space) ────────────────────────

/** Parameters for the generation color space (subset of HelmlabParams). */
export interface GenParams {
  M1: number[][];
  gamma: number[];
  M2: number[][];
  hue_cos1: number; hue_sin1: number;
  hue_cos2: number; hue_sin2: number;
  hue_cos3: number; hue_sin3: number;
  hue_cos4: number; hue_sin4: number;
  L_corr_p1: number; L_corr_p2: number; L_corr_p3: number;
  lp_dark: number; lp_dark_hcos: number; lp_dark_hsin: number;
  lc1: number; lc2: number;
}

/** Compiled gen params ready for compute. */
export interface CompiledGenParams {
  M1: Mat3;
  M1_inv: Mat3;
  M2: Mat3;
  M2_inv: Mat3;
  gamma: Float64Array;
  inv_gamma: Float64Array;
  raw: GenParams;
}

export function compileGenParams(p: GenParams): CompiledGenParams {
  const M1 = mat3(p.M1);
  const M2 = mat3(p.M2);
  return {
    M1, M1_inv: mat3Inv(M1),
    M2, M2_inv: mat3Inv(M2),
    gamma: new Float64Array(p.gamma),
    inv_gamma: new Float64Array(p.gamma.map(g => 1 / g)),
    raw: p,
  };
}

export function getDefaultGenParams(): GenParams {
  return defaultGenParams as GenParams;
}
