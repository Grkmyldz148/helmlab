import {
  overfitAnalysis,
  overfitKeyFindings,
  overfitStressReporting,
  trainTestSplit,
} from "../../data/overfit";

function GapBadge({ gap }: { gap: number }) {
  const absGap = Math.abs(gap);
  let color: string;
  let label: string;

  if (absGap < 1) {
    color = "text-[#34d399] bg-[#34d399]/10";
    label = "Excellent";
  } else if (absGap < 2) {
    color = "text-[#fbbf24] bg-[#fbbf24]/10";
    label = "Mild";
  } else {
    color = "text-[#f87171] bg-[#f87171]/10";
    label = "High";
  }

  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${color}`}>
      {gap >= 0 ? "+" : ""}{gap.toFixed(2)}
    </span>
  );
}

export default function OverfitPanel() {
  return (
    <div>
      {/* Dual narrative */}
      <div className="mb-6">
        <p className="for-designer text-sm text-[#a1a1aa]">
          Does the model genuinely predict color perception, or did it just memorize the training data?
          We tested this rigorously with held-out data the model never saw during training.
        </p>
        <p className="for-dev text-sm text-[#a1a1aa]">
          80/20 train-test split (seed=42, {trainTestSplit.trainPairs}/{trainTestSplit.testPairs} pairs).
          Multiple DOF configurations tested. Cross-validated estimate: STRESS {overfitStressReporting.crossValidatedEstimate}.
        </p>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/[0.06] bg-white/[0.02]">
              <th className="px-4 py-3 text-left text-xs font-medium text-[#52525b] uppercase tracking-wider">
                Model
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-[#52525b] uppercase tracking-wider">
                <span className="for-designer">Params</span>
                <span className="for-dev">DOF</span>
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-[#52525b] uppercase tracking-wider">
                Train
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-[#52525b] uppercase tracking-wider">
                Test
              </th>
              <th className="px-4 py-3 text-center text-xs font-medium text-[#52525b] uppercase tracking-wider">
                Gap
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-[#52525b] uppercase tracking-wider hidden sm:table-cell">
                MacAdam
              </th>
              <th className="px-4 py-3 text-right text-xs font-medium text-[#52525b] uppercase tracking-wider hidden sm:table-cell">
                Human FB
              </th>
            </tr>
          </thead>
          <tbody>
            {overfitAnalysis.map((row, i) => (
              <tr
                key={row.model}
                className={`border-b border-white/[0.04] ${
                  i === 1 ? "bg-[#f97316]/[0.04]" : "hover:bg-white/[0.02]"
                } transition-colors`}
              >
                <td className="px-4 py-3">
                  <span className={`font-medium ${i === 1 ? "text-[#f97316]" : "text-[#fafafa]"}`}>
                    {row.model}
                  </span>
                  <p className="text-xs text-[#52525b] mt-0.5 hidden lg:block">{row.description}</p>
                </td>
                <td className="px-4 py-3 text-left text-[#a1a1aa] font-mono text-xs">
                  {row.dof}
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#a1a1aa]">
                  {row.trainStress.toFixed(2)}
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#a1a1aa]">
                  {row.testStress.toFixed(2)}
                </td>
                <td className="px-4 py-3 text-center">
                  <GapBadge gap={row.gap} />
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#52525b] hidden sm:table-cell">
                  {row.macadam.toFixed(2)}
                </td>
                <td className="px-4 py-3 text-right font-mono text-[#52525b] hidden sm:table-cell">
                  {row.humanFeedback.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Key findings */}
      <div className="mt-6 space-y-3">
        <h4 className="text-sm font-medium text-[#fafafa]">Key Findings</h4>
        {overfitKeyFindings.map((finding, i) => (
          <div key={i} className="flex gap-3 items-start">
            <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-[#f97316]" />
            <div>
              <span className="text-sm font-medium text-[#fafafa]">{finding.label}. </span>
              <span className="text-sm text-[#a1a1aa]">{finding.detail}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Bottom note */}
      <div className="mt-6 rounded-lg border border-white/[0.06] bg-white/[0.02] px-4 py-3">
        <p className="text-xs text-[#52525b]">
          <span className="text-[#a1a1aa] font-medium">Published STRESS: {overfitStressReporting.publishedFullData}</span>
          {" "}(full-data optimized) |{" "}
          <span className="text-[#a1a1aa] font-medium">Cross-validated: ~{overfitStressReporting.crossValidatedEstimate}</span>
          {" "}| {overfitStressReporting.note}
        </p>
      </div>
    </div>
  );
}
