import { useState } from "react";
import { categorySummary, genspaceResults, headlineScore, type MetricCategory } from "../../data/genspace-results";

const categoryDescriptions: Record<MetricCategory, { designer: string; dev: string }> = {
  Gamut: {
    designer: "How well the space maps to real device screens",
    dev: "Cusp validity, boundary smoothness, clipping across sRGB/P3/Rec2020",
  },
  Gradient: {
    designer: "How smooth colors blend between two endpoints",
    dev: "CV of perceptual step size, hue drift, banding metrics",
  },
  Application: {
    designer: "Real-world tasks like palettes, tints, and accessibility",
    dev: "Palette generation, gamut mapping, WCAG contrast, animation",
  },
  Perceptual: {
    designer: "Agreement with how humans actually see color",
    dev: "Munsell, MacAdam, Hung-Berns hue linearity validation",
  },
  Structural: {
    designer: "Mathematical properties that affect reliability",
    dev: "Hue reversals, OOG excursion, chroma amplification, LMS",
  },
  Hue: {
    designer: "Whether hue labels match human expectation",
    dev: "Hue RMS vs Munsell, primary lightness range",
  },
  Achromatic: {
    designer: "Perfect grays without color contamination",
    dev: "Gray ramp chroma residual under sRGB and D65",
  },
  Advanced: {
    designer: "Edge cases and stress tests",
    dev: "1000-trip roundtrip, Jacobian condition, 8-bit precision",
  },
  Special: {
    designer: "Problem areas where OKLab is known to struggle",
    dev: "Yellow chroma, blue-to-white midpoint, red-to-white shift",
  },
  Banding: {
    designer: "Visible stepping artifacts in gradients",
    dev: "Invisible step ratio, duplicate 8-bit bucket count",
  },
  Accessibility: {
    designer: "Usability for colorblind viewers",
    dev: "CVD simulation minimum step deltaE (protan/deutan)",
  },
  Numerical: {
    designer: "Mathematical precision of conversions",
    dev: "Round-trip error across sRGB, P3, Rec2020 (float64)",
  },
  Independent: {
    designer: "Tests using data we never trained on",
    dev: "Hung-Berns, Ebner-Fairchild, Pointer's Gamut",
  },
};

function WinBar({
  wins,
  losses,
  ties,
  total,
}: {
  wins: number;
  losses: number;
  ties: number;
  total: number;
}) {
  const wPct = (wins / total) * 100;
  const lPct = (losses / total) * 100;
  const tPct = (ties / total) * 100;

  return (
    <div className="flex h-2 w-full overflow-hidden rounded-full bg-white/[0.04]">
      {wPct > 0 && (
        <div
          className="bg-[#34d399] transition-all duration-300"
          style={{ width: `${wPct}%` }}
        />
      )}
      {tPct > 0 && (
        <div
          className="bg-[#3f3f46] transition-all duration-300"
          style={{ width: `${tPct}%` }}
        />
      )}
      {lPct > 0 && (
        <div
          className="bg-[#f87171] transition-all duration-300"
          style={{ width: `${lPct}%` }}
        />
      )}
    </div>
  );
}

function MetricRow({ metric }: { metric: (typeof genspaceResults)[number] }) {
  const winColor =
    metric.winner === "genspace"
      ? "text-[#34d399]"
      : metric.winner === "oklab"
        ? "text-[#f87171]"
        : "text-[#52525b]";

  return (
    <div className="flex items-center gap-3 px-3 py-2 rounded-md hover:bg-white/[0.02] transition-colors text-xs">
      <span className="flex-1 text-[#a1a1aa] truncate">{metric.name}</span>
      <span className="font-mono text-[#52525b] w-20 text-right shrink-0">
        {typeof metric.oklab === "number" ? metric.oklab.toLocaleString() : metric.oklab}
      </span>
      <span className={`font-mono w-20 text-right shrink-0 font-medium ${winColor}`}>
        {typeof metric.genspace === "number" ? metric.genspace.toLocaleString() : metric.genspace}
      </span>
      <span className={`w-5 text-center shrink-0 ${winColor}`}>
        {metric.winner === "genspace" ? "W" : metric.winner === "oklab" ? "L" : "-"}
      </span>
    </div>
  );
}

export default function CategoryCards() {
  const [expanded, setExpanded] = useState<MetricCategory | null>(null);

  return (
    <div>
      <div className="mb-6">
        <p className="for-designer text-sm text-[#a1a1aa]">
          Performance breakdown across 13 test categories. Click any card to see individual metrics.
        </p>
        <p className="for-dev text-sm text-[#a1a1aa]">
          {headlineScore.internalMetrics} internal metrics + {headlineScore.independentMetrics} independent validation metrics in {categorySummary.length} categories. All deterministic, float64.
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {categorySummary.map((cat) => {
          const isExpanded = expanded === cat.category;
          const metrics = genspaceResults.filter((m) => m.category === cat.category);
          const desc = categoryDescriptions[cat.category];

          return (
            <div
              key={cat.category}
              className={`rounded-xl border transition-all duration-200 cursor-pointer ${
                isExpanded
                  ? "border-[#f97316]/30 bg-[#f97316]/[0.03] col-span-1 sm:col-span-2 lg:col-span-3"
                  : "border-white/[0.06] bg-[#0c0c0f] hover:border-white/[0.12]"
              }`}
              onClick={() => setExpanded(isExpanded ? null : cat.category)}
            >
              <div className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-sm font-medium text-[#fafafa]">{cat.category}</h3>
                  <div className="flex items-center gap-2 text-xs">
                    <span className="text-[#34d399] font-medium">{cat.genspace}W</span>
                    <span className="text-[#52525b]">{cat.ties}T</span>
                    <span className="text-[#f87171] font-medium">{cat.oklab}L</span>
                  </div>
                </div>

                <WinBar
                  wins={cat.genspace}
                  losses={cat.oklab}
                  ties={cat.ties}
                  total={cat.total}
                />

                <p className="for-designer text-xs text-[#52525b] mt-2">{desc.designer}</p>
                <p className="for-dev text-xs text-[#52525b] mt-2">{desc.dev}</p>

                {/* Expand indicator */}
                <div className="flex items-center gap-1 mt-2">
                  <svg
                    className={`h-3 w-3 text-[#52525b] transition-transform duration-200 ${
                      isExpanded ? "rotate-90" : ""
                    }`}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                  <span className="text-[10px] text-[#3f3f46]">
                    {cat.total} metrics
                  </span>
                </div>
              </div>

              {/* Expanded metric list */}
              {isExpanded && (
                <div className="border-t border-white/[0.06] px-2 py-2">
                  <div className="flex items-center gap-3 px-3 py-1.5 text-[10px] uppercase tracking-wider text-[#3f3f46]">
                    <span className="flex-1">Metric</span>
                    <span className="w-20 text-right">OKLab</span>
                    <span className="w-20 text-right">GenSpace</span>
                    <span className="w-5 text-center" />
                  </div>
                  {metrics.map((m) => (
                    <MetricRow key={m.name} metric={m} />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
