export interface StressScore {
  readonly metric: string;
  readonly shortName: string;
  readonly combvd: number;
  readonly macadam: number;
  readonly humanFeedback: number;
  readonly average: number;
  readonly isHelmlab: boolean;
}

export const stressScores = [
  {
    metric: "MetricSpace v21",
    shortName: "MetricSpace",
    combvd: 22.48,
    macadam: 19.51,
    humanFeedback: 23.26,
    average: 21.75,
    isHelmlab: true,
  },
  {
    metric: "CIE94",
    shortName: "CIE94",
    combvd: 33.37,
    macadam: 19.78,
    humanFeedback: 59.75,
    average: 37.63,
    isHelmlab: false,
  },
  {
    metric: "CIEDE2000",
    shortName: "CIEDE2000",
    combvd: 29.20,
    macadam: 22.13,
    humanFeedback: 62.54,
    average: 37.96,
    isHelmlab: false,
  },
  {
    metric: "DIN99",
    shortName: "DIN99",
    combvd: 35.57,
    macadam: 23.31,
    humanFeedback: 56.44,
    average: 38.44,
    isHelmlab: false,
  },
  {
    metric: "Jzazbz",
    shortName: "Jzazbz",
    combvd: 41.92,
    macadam: 24.14,
    humanFeedback: 61.74,
    average: 42.60,
    isHelmlab: false,
  },
  {
    metric: "CIE Lab \u0394E76",
    shortName: "CIE Lab",
    combvd: 42.86,
    macadam: 24.53,
    humanFeedback: 62.32,
    average: 43.24,
    isHelmlab: false,
  },
  {
    metric: "OKLab",
    shortName: "OKLab",
    combvd: 47.35,
    macadam: 32.72,
    humanFeedback: 57.27,
    average: 45.78,
    isHelmlab: false,
  },
  {
    metric: "CAM16-UCS",
    shortName: "CAM16",
    combvd: 33.47,
    macadam: 18.71,
    humanFeedback: 58.02,
    average: 36.73,
    isHelmlab: false,
  },
  {
    metric: "CIECAM02-UCS",
    shortName: "CIECAM02",
    combvd: 30.90,
    macadam: 19.21,
    humanFeedback: 57.83,
    average: 35.98,
    isHelmlab: false,
  },
] as const satisfies readonly StressScore[];

export type StressDataset = "combvd" | "macadam" | "humanFeedback";

export const datasetLabels: Record<StressDataset, string> = {
  combvd: "COMBVD (3813 pairs)",
  macadam: "MacAdam 1974 (128 pairs)",
  humanFeedback: "Human Feedback (3552 judgements)",
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
