/**
 * Format a number in scientific notation, e.g. 1.67e-15
 */
export function formatScientific(n: number): string {
  if (n === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(n)));
  const mantissa = n / Math.pow(10, exp);
  return `${mantissa.toFixed(2)}e${exp >= 0 ? "+" : ""}${exp}`;
}

/**
 * Format a number as a percentage, e.g. "37.45%"
 */
export function formatPercent(n: number, decimals: number = 2): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

/**
 * Format a number as degrees, e.g. "27.5°"
 */
export function formatDegrees(n: number, decimals: number = 1): string {
  return `${n.toFixed(decimals)}\u00B0`;
}

/**
 * Format a number as a ratio, e.g. "1.78"
 */
export function formatRatio(n: number): string {
  return n.toFixed(2);
}
