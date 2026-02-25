/** WCAG contrast ratio utilities. */

import type { RGB } from '../types.js';
import { relativeLuminance } from './srgb.js';

/** WCAG contrast ratio (1:1 to 21:1). */
export function contrastRatio(fg: RGB, bg: RGB): number {
  const L1 = relativeLuminance(fg);
  const L2 = relativeLuminance(bg);
  const lighter = Math.max(L1, L2);
  const darker = Math.min(L1, L2);
  return (lighter + 0.05) / (darker + 0.05);
}
