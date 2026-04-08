// ── Hung & Berns (1995) — Hue Linearity ──

export interface HungBernsHue {
  readonly hue: string;
  readonly oklab: number;
  readonly genspace: number;
  readonly winner: "genspace" | "oklab" | "tie";
}

export const hungBernsPerHue = [
  { hue: "Red", oklab: 2.42, genspace: 2.84, winner: "oklab" },
  { hue: "Red-yellow", oklab: 4.24, genspace: 3.55, winner: "genspace" },
  { hue: "Yellow", oklab: 5.71, genspace: 4.51, winner: "genspace" },
  { hue: "Yellow-green", oklab: 5.48, genspace: 4.69, winner: "genspace" },
  { hue: "Green", oklab: 1.64, genspace: 1.69, winner: "oklab" },
  { hue: "Green-cyan", oklab: 7.13, genspace: 8.77, winner: "oklab" },
  { hue: "Cyan", oklab: 9.37, genspace: 9.94, winner: "oklab" },
  { hue: "Cyan-blue", oklab: 7.01, genspace: 5.58, winner: "genspace" },
  { hue: "Blue", oklab: 4.62, genspace: 3.29, winner: "genspace" },
  { hue: "Blue-magenta", oklab: 4.52, genspace: 3.76, winner: "genspace" },
  { hue: "Magenta", oklab: 3.59, genspace: 3.54, winner: "tie" },
  { hue: "Magenta-red", oklab: 3.78, genspace: 4.43, winner: "oklab" },
] as const satisfies readonly HungBernsHue[];

export const hungBernsSummary = {
  samples: 168,
  hues: 12,
  targetsPerHue: 13,
  observers: 9,
  source: "Zenodo 3367463",
  genspaceWins: 6,
  oklabWins: 5,
  ties: 1,
  cieLab: { meanAngularDeviation: 5.94, maxAngularDeviation: 33.5 },
  oklab: { meanAngularDeviation: 4.96, maxAngularDeviation: 25.5 },
  genspace: { meanAngularDeviation: 4.72, maxAngularDeviation: 25.2 },
  overallWinner: "genspace" as const,
} as const;

// ── Ebner & Fairchild (1998) — Constant Perceived-Hue Surfaces ──

export interface EbnerFairchildResult {
  readonly space: string;
  readonly meanAngularDeviation: number;
  readonly maxAngularDeviation: number;
}

export const ebnerFairchild = [
  { space: "CIE Lab", meanAngularDeviation: 2.95, maxAngularDeviation: 16.0 },
  { space: "OKLab", meanAngularDeviation: 2.23, maxAngularDeviation: 8.1 },
  { space: "GenSpace", meanAngularDeviation: 2.10, maxAngularDeviation: 8.6 },
] as const satisfies readonly EbnerFairchildResult[];

export const ebnerFairchildSummary = {
  samples: 321,
  hues: 15,
  source: "Zenodo 3362536",
  meanWinner: "genspace" as const,
  maxWinner: "oklab" as const,
} as const;

// ── Pointer's Gamut (1980) — Real Surface Colors ──

export interface PointerGamutResult {
  readonly space: string;
  readonly chromaCV: number;
  readonly boundarySmoothness: number;
  readonly hueUniformityCV: number;
}

export const pointerGamut = [
  { space: "CIE Lab", chromaCV: 0.479, boundarySmoothness: 0.144, hueUniformityCV: 0.034 },
  { space: "OKLab", chromaCV: 0.413, boundarySmoothness: 0.132, hueUniformityCV: 0.370 },
  { space: "GenSpace", chromaCV: 0.404, boundarySmoothness: 0.125, hueUniformityCV: 0.262 },
] as const satisfies readonly PointerGamutResult[];

export const pointerGamutSummary = {
  boundaryPoints: 576,
  lLevels: 16,
  hueAngles: 36,
  source: "colour-science",
  chromaCVWinner: "genspace" as const,
  boundarySmoothnessWinner: "genspace" as const,
  hueUniformityCVWinner: "cielab" as const,
  note: "CIE Lab wins hue uniformity because Pointer's data is defined in CIE Lab coordinates.",
} as const;

// ── Independent Benchmark Summary ──

export interface IndependentDatasetSummary {
  readonly dataset: string;
  readonly year: number;
  readonly genspaceWins: number;
  readonly oklabWins: number;
  readonly source: string;
}

export const independentSummary = [
  { dataset: "Hung & Berns", year: 1995, genspaceWins: 2, oklabWins: 0, source: "Zenodo 3367463" },
  { dataset: "Ebner & Fairchild", year: 1998, genspaceWins: 1, oklabWins: 1, source: "Zenodo 3362536" },
  { dataset: "Pointer's Gamut", year: 1980, genspaceWins: 3, oklabWins: 0, source: "colour-science" },
] as const satisfies readonly IndependentDatasetSummary[];

export const independentTotals = {
  genspaceWins: 6,
  oklabWins: 1,
  totalMetrics: 7,
} as const;
