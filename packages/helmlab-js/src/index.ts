export const VERSION: string = typeof __HELMLAB_VERSION__ !== 'undefined' ? __HELMLAB_VERSION__ : '0.0.0-dev';
export { Helmlab, Gen, Metric, ContrastError, parseColor } from './helmlab.js';
export type { HelmlabOptions, Gamut, HarmonyKind } from './helmlab.js';
export { Tokens } from './tokens.js';
export type { TokenCssFormat } from './tokens.js';
export type { Lab, XYZ, RGB, Hex, LCh, SemanticScale, WCAGLevel, GenLab, MetricLab } from './types.js';
export type { HelmlabParams, GenParams } from './core/params.js';

// Lower-level exports for advanced usage
export { AnalyticalSpace } from './core/analytical.js';
export type { AnalyticalOptions } from './core/analytical.js';
export { MetricSpace } from './spaces/metric.js';
export type { MetricOptions } from './spaces/metric.js';
export { GenSpace } from './spaces/gen.js';
export type { GenSpaceOptions } from './spaces/gen.js';
export { compileParams, getDefaultParams, compileGenParams, getDefaultGenParams } from './core/params.js';
export {
  hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb,
  xyzToDisplayP3, displayP3ToXyz,
  xyzToRec2020, rec2020ToXyz,
  linearToSrgb, srgbToLinear,
  linearToRec2020, rec2020ToLinear,
  clampRgb, relativeLuminance,
} from './utils/srgb.js';
export { gamutMap, isInGamut, maxChroma, findCusp } from './utils/gamut.js';
export type { GamutMapMethod } from './utils/gamut.js';
export type { SpaceLike } from './utils/gamut.js';
export { contrastRatio } from './utils/contrast.js';
