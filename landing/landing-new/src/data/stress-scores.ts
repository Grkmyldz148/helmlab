export interface StressScore {
  readonly metric: string;
  readonly shortName: string;
  readonly combvd: number;
  readonly macadam: number;
  readonly average: number;
  readonly isHelmlab: boolean;
}

export const stressScores = [
  {
    metric: "MetricSpace v21",
    shortName: "MetricSpace",
    combvd: 22.48,
    macadam: 19.51,
    average: 21.75,
    isHelmlab: true,
  },
  {
    metric: "CIE94",
    shortName: "CIE94",
    combvd: 33.37,
    macadam: 19.78,
    average: 37.63,
    isHelmlab: false,
  },
  {
    metric: "CIEDE2000",
    shortName: "CIEDE2000",
    combvd: 29.20,
    macadam: 22.13,
    average: 37.96,
    isHelmlab: false,
  },
  {
    metric: "DIN99",
    shortName: "DIN99",
    combvd: 35.57,
    macadam: 23.31,
    average: 38.44,
    isHelmlab: false,
  },
  {
    metric: "Jzazbz",
    shortName: "Jzazbz",
    combvd: 41.92,
    macadam: 24.14,
    average: 42.60,
    isHelmlab: false,
  },
  {
    metric: "CIE Lab \u0394E76",
    shortName: "CIE Lab",
    combvd: 42.86,
    macadam: 24.53,
    average: 43.24,
    isHelmlab: false,
  },
  {
    metric: "OKLab",
    shortName: "OKLab",
    combvd: 47.35,
    macadam: 32.72,
    average: 45.78,
    isHelmlab: false,
  },
  {
    metric: "CAM16-UCS",
    shortName: "CAM16",
    combvd: 33.47,
    macadam: 18.71,
    average: 36.73,
    isHelmlab: false,
  },
  {
    metric: "CIECAM02-UCS",
    shortName: "CIECAM02",
    combvd: 30.90,
    macadam: 19.21,
    average: 35.98,
    isHelmlab: false,
  },
] as const satisfies readonly StressScore[];

export type StressDataset = "combvd" | "macadam";

export const datasetLabels: Record<StressDataset, string> = {
  combvd: "COMBVD (3813 pairs)",
  macadam: "MacAdam 1974 (128 pairs)",
} as const;

export interface SubDatasetScore {
  readonly name: string;
  readonly n: number;
  readonly metricSpace: number;
  readonly ciede2000: number;
  readonly winner: "metricspace" | "ciede2000" | "tie";
}

export const combvdSubDatasets = [
  { name: "BFD-P (C)", n: 200, metricSpace: 29.08, ciede2000: 29.08, winner: "tie" },
  { name: "BFD-P (D65)", n: 2028, metricSpace: 21.54, ciede2000: 24.09, winner: "metricspace" },
  { name: "BFD-P (M)", n: 548, metricSpace: 21.75, ciede2000: 35.23, winner: "metricspace" },
  { name: "LEEDS", n: 307, metricSpace: 21.84, ciede2000: 19.25, winner: "ciede2000" },
  { name: "RIT-DuPont", n: 312, metricSpace: 21.90, ciede2000: 19.47, winner: "ciede2000" },
  { name: "WITT", n: 418, metricSpace: 30.93, ciede2000: 30.22, winner: "ciede2000" },
] as const satisfies readonly SubDatasetScore[];

export interface SubsetSplitScore {
  readonly name: string;
  readonly n: number;
  /** median in-subset CIELAB ΔE*ab used as the split point */
  readonly medDE: number;
  readonly smallHelm: number;
  readonly smallDE2K: number;
  readonly bigHelm: number;
  readonly bigDE2K: number;
}

/**
 * Where the CIEDE2000 gap lives — each COMBVD subset split at its median
 * CIELAB ΔE*ab into a near-threshold half and a larger-difference half,
 * STRESS computed per half (optimal scale refit per half).
 *
 * Measured 2026-07-05 with the shipped v21 params, Bradford CAT to D65,
 * CIEDE2000 from the same harness. Deterministic; rerun reproduces exactly.
 * Takeaway: the LEEDS/RIT-DuPont deficit is almost entirely near-threshold
 * (ΔE*ab ≲ 1.5) — the tolerance regime CIEDE2000's S-functions were derived
 * from — while MetricSpace wins both halves on the (much larger) BFD-P sets.
 */
export const combvdSubsetSplit = [
  { name: "BFD-P (C)", n: 200, medDE: 0.73, smallHelm: 31.72, smallDE2K: 26.96, bigHelm: 27.49, bigDE2K: 27.13 },
  { name: "BFD-P (D65)", n: 2028, medDE: 2.24, smallHelm: 23.57, smallDE2K: 23.99, bigHelm: 20.98, bigDE2K: 23.91 },
  { name: "BFD-P (M)", n: 548, medDE: 4.49, smallHelm: 29.07, smallDE2K: 34.63, bigHelm: 17.50, bigDE2K: 32.27 },
  { name: "LEEDS", n: 307, medDE: 1.57, smallHelm: 23.15, smallDE2K: 18.94, bigHelm: 18.07, bigDE2K: 16.48 },
  { name: "RIT-DuPont", n: 312, medDE: 1.35, smallHelm: 22.67, smallDE2K: 19.07, bigHelm: 18.48, bigDE2K: 17.93 },
  { name: "WITT", n: 418, medDE: 1.46, smallHelm: 29.83, smallDE2K: 28.71, bigHelm: 29.35, bigDE2K: 29.08 },
] as const satisfies readonly SubsetSplitScore[];
