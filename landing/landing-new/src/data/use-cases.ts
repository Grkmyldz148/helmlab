export interface UseCase {
  readonly useCase: string;
  readonly why: string;
  readonly alternative: string;
  readonly advantage: string;
}

export const metricSpaceUseCases = [
  {
    useCase: "Quality control (print, display, textile)",
    why: "Most accurate \u0394E prediction",
    alternative: "CIEDE2000",
    advantage: "23% lower STRESS on COMBVD",
  },
  {
    useCase: "Color matching tolerance",
    why: "Pair-dependent SL/SC weighting",
    alternative: "CIE94",
    advantage: "Adapts to lightness and chroma of the specific pair",
  },
  {
    useCase: "A/B testing",
    why: "Human Feedback STRESS=23.26",
    alternative: "CIEDE2000 (62.54)",
    advantage: "63% better on real user feedback data",
  },
  {
    useCase: "Accessibility checking",
    why: "Euclidean \u0394E is perceptually calibrated",
    alternative: "OKLab \u0394E",
    advantage: "OKLab STRESS=47 \u2014 not designed for this",
  },
  {
    useCase: "Research",
    why: "Transparent pipeline, open source",
    alternative: "Proprietary CAMs",
    advantage: "Fully invertible, reproducible",
  },
] as const satisfies readonly UseCase[];

export const genSpaceUseCases = [
  {
    useCase: "Design system palettes",
    why: "Uniform L spacing, 360/360 cusps",
    alternative: "OKLab",
    advantage: "18\u00D7 better Munsell Value uniformity",
  },
  {
    useCase: "Gradient interpolation",
    why: "Lower CV, less hue drift",
    alternative: "OKLab",
    advantage: "31% less max hue drift, better cross-lightness",
  },
  {
    useCase: "Gamut mapping",
    why: "Smooth boundary, no cliffs",
    alternative: "OKLab",
    advantage: "11\u00D7 smoother cusps, 0 monotonicity violations (sRGB/P3)",
  },
  {
    useCase: "Color harmony tools",
    why: "Hue rotations stay truer to targets",
    alternative: "OKLab",
    advantage: "Palette harmony accuracy 9.1\u00b0 vs 11.7\u00b0",
  },
  {
    useCase: "Dark UI themes",
    why: "26% better dark gradient CV",
    alternative: "OKLab",
    advantage: "Dark regions are where OKLab struggles most",
  },
  {
    useCase: "Animation",
    why: "3% better frame-to-frame CV",
    alternative: "OKLab",
    advantage: "Smoother color transitions over time",
  },
  {
    useCase: "Wide gamut (P3, Rec.2020)",
    why: "360/360 P3 cusps",
    alternative: "OKLab (308/360)",
    advantage: "Complete gamut boundary representation",
  },
  {
    useCase: "Blue-sensitive design",
    why: "G/R=1.51 vs OKLab 1.41 — stays a bit more saturated, both blue",
    alternative: "OKLab (G/R=1.41)",
    advantage: "Blue\u2192white midpoint is actually blue",
  },
  {
    useCase: "Yellow-sensitive design",
    why: "58% higher yellow chroma",
    alternative: "OKLab",
    advantage: "More vivid yellows without saturation clipping",
  },
] as const satisfies readonly UseCase[];

export interface NotRecommended {
  readonly situation: string;
  readonly betterChoice: string;
  readonly why: string;
}

export const notRecommended = [
  {
    situation: "Near-achromatic gradient mastering",
    betterChoice: "OKLab",
    why: "24% better near-achromatic CV",
  },
  {
    situation: "CVD-optimized palettes (deutan)",
    betterChoice: "OKLab",
    why: "43% better deutan distinguishability",
  },
  {
    situation: "CSS oklch() is mandatory",
    betterChoice: "OKLab",
    why: "Web standard, native browser support",
  },
  {
    situation: "HDR/PQ content mastering",
    betterChoice: "Jzazbz",
    why: "Designed for perceptual quantization",
  },
  {
    situation: "Minimal bundle size, no enrichment needed",
    betterChoice: "OKLab",
    why: "3 matrix multiplies, ~2KB",
  },
  {
    situation: "Legacy CIE Lab hue angle compatibility",
    betterChoice: "CIE Lab",
    why: "Established industry hue naming conventions",
  },
] as const satisfies readonly NotRecommended[];
