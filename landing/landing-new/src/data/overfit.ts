export interface OverfitEntry {
  readonly model: string;
  readonly description: string;
  readonly dof: number;
  readonly trainStress: number;
  readonly testStress: number;
  readonly gap: number;
  readonly macadam: number;
  readonly humanFeedback: number;
}

export const overfitAnalysis = [
  {
    model: "v20b baseline",
    description: "No optimization (starting point)",
    dof: 0,
    trainStress: 27.72,
    testStress: 27.57,
    gap: -0.15,
    macadam: 20.09,
    humanFeedback: 33.68,
  },
  {
    model: "v21 (full-data)",
    description: "Full-data trained (published result)",
    dof: 72,
    trainStress: 22.14,
    testStress: 23.91,
    gap: 1.77,
    macadam: 19.51,
    humanFeedback: 23.26,
  },
  {
    model: "Phase 1 train-only",
    description: "Train split only, 6 DOF optimization",
    dof: 6,
    trainStress: 25.35,
    testStress: 25.65,
    gap: 0.30,
    macadam: 18.85,
    humanFeedback: 24.76,
  },
  {
    model: "Phase 1+2 train-only",
    description: "Train split only, 48 DOF optimization",
    dof: 48,
    trainStress: 22.78,
    testStress: 24.59,
    gap: 1.82,
    macadam: 19.12,
    humanFeedback: 23.73,
  },
] as const satisfies readonly OverfitEntry[];

export const overfitKeyFindings = [
  {
    label: "Mild overfitting exists",
    detail: "+1.8 STRESS gap between train and held-out test. This is real and acknowledged.",
  },
  {
    label: "Gap is from data variance, not memorization",
    detail: "v21 (full-data) gap = +1.77, train-only gap = +1.82. Nearly identical.",
  },
  {
    label: "Held-out test still beats all competitors",
    detail: "Held-out test STRESS (24.59) vs CIEDE2000 (29.20) on full dataset = 16% better.",
  },
  {
    label: "MacAdam generalizes independently",
    detail: "MacAdam 1974 (never trained on) = 19.12, consistent with v21's 19.51.",
  },
  {
    label: "Low DOF shows near-zero gap",
    detail: "6 DOF shows +0.30 gap. Overfitting scales with parameters but stays controlled.",
  },
] as const;

export const overfitStressReporting = {
  publishedFullData: 22.48,
  crossValidatedEstimate: 24.3,
  note: "Both are still #1 among all tested competitors.",
} as const;

export const trainTestSplit = {
  seed: 42,
  trainRatio: 0.8,
  trainPairs: 3050,
  testPairs: 763,
  totalPairs: 3813,
} as const;
