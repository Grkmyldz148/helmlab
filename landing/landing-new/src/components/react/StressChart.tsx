import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  stressScores,
  combvdSubDatasets,
  combvdSubsetSplit,
  datasetLabels,
  type StressDataset,
} from "../../data/stress-scores";

const HELMLAB_COLOR = "#f97316";
const COMPETITOR_COLOR = "#3f3f46";
const COMPETITOR_HOVER = "#52525b";
const CIEDE2000_COLOR = "#60a5fa";

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    payload: (typeof stressScores)[number];
    value: number;
  }>;
}

function CustomTooltip({ active, payload }: CustomTooltipProps) {
  if (!active || !payload?.[0]) return null;
  const d = payload[0].payload;
  return (
    <div className="rounded-lg border border-white/[0.08] bg-[#0c0c0f] px-3 py-2 text-xs shadow-xl">
      <p className="font-medium text-[#fafafa] mb-1">{d.metric}</p>
      <div className="space-y-0.5 text-[#a1a1aa]">
        <p>COMBVD: {d.combvd}</p>
        <p>MacAdam: {d.macadam}</p>
        <p className="pt-1 border-t border-white/[0.06] text-[#fafafa] font-medium">
          Avg: {d.average}
        </p>
      </div>
    </div>
  );
}

function DatasetTabs({
  active,
  onChange,
}: {
  active: StressDataset;
  onChange: (d: StressDataset) => void;
}) {
  const datasets: StressDataset[] = ["combvd", "macadam"];
  return (
    <div className="flex gap-1 rounded-lg border border-white/[0.06] bg-white/[0.02] p-1">
      {datasets.map((d) => (
        <button
          key={d}
          onClick={() => onChange(d)}
          className={`rounded-md px-3 py-1.5 text-xs transition-colors ${
            active === d
              ? "bg-white/[0.08] text-[#fafafa] font-medium"
              : "text-[#52525b] hover:text-[#a1a1aa]"
          }`}
        >
          {d === "combvd" ? "COMBVD" : "MacAdam"}
        </button>
      ))}
    </div>
  );
}

export default function StressChart() {
  const [mounted, setMounted] = useState(false);
  const [dataset, setDataset] = useState<StressDataset>("combvd");

  useEffect(() => setMounted(true), []);

  const sorted = [...stressScores].sort((a, b) => a[dataset] - b[dataset]);

  const chartData = sorted.map((s) => ({
    ...s,
    value: s[dataset],
  }));

  return (
    <div>
      {/* Dual narrative labels */}
      <div className="mb-6">
        <p className="for-designer text-sm text-[#a1a1aa] mb-4">
          How well each metric predicts human color perception. Lower scores mean better agreement with human judgement.
        </p>
        <p className="for-dev text-sm text-[#a1a1aa] mb-4">
          STRESS index (CIE 217:2016) across 3 psychophysical datasets. Lower is better. Full-data optimized result.
        </p>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-4">
        <DatasetTabs active={dataset} onChange={setDataset} />
        <span className="text-xs text-[#52525b]">{datasetLabels[dataset]}</span>
      </div>

      <div className="h-[380px] w-full">
        {!mounted ? (
          <div className="h-full w-full flex items-center justify-center text-[#52525b] text-sm">
            Loading chart...
          </div>
        ) : (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 4, right: 30, left: 8, bottom: 4 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.04)"
              horizontal={false}
            />
            <XAxis
              type="number"
              tick={{ fill: "#52525b", fontSize: 11 }}
              axisLine={{ stroke: "rgba(255,255,255,0.06)" }}
              tickLine={false}
              domain={[0, "auto"]}
            />
            <YAxis
              dataKey="shortName"
              type="category"
              width={90}
              tick={{ fill: "#a1a1aa", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ fill: "rgba(255,255,255,0.02)" }}
            />
            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={28}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isHelmlab ? HELMLAB_COLOR : COMPETITOR_COLOR}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        )}
      </div>

      {/* Sub-dataset breakdown for COMBVD */}
      {dataset === "combvd" && (
        <div className="mt-6 pt-6 border-t border-white/[0.06]">
          <h4 className="text-sm font-medium text-[#a1a1aa] mb-3">
            COMBVD Sub-dataset Breakdown{" "}
            <span className="text-[#52525b] font-normal">(MetricSpace vs CIEDE2000)</span>
          </h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {combvdSubDatasets.map((sub) => (
              <div
                key={sub.name}
                className="rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2"
              >
                <div className="text-xs text-[#52525b] mb-1">
                  {sub.name}{" "}
                  <span className="text-[#3f3f46]">n={sub.n}</span>
                </div>
                <div className="flex items-baseline gap-2">
                  <span
                    className={`text-sm font-mono font-medium ${
                      sub.winner === "metricspace"
                        ? "text-[#34d399]"
                        : sub.winner === "tie"
                          ? "text-[#a1a1aa]"
                          : "text-[#a1a1aa]"
                    }`}
                  >
                    {sub.metricSpace}
                  </span>
                  <span className="text-[#3f3f46] text-xs">vs</span>
                  <span
                    className={`text-sm font-mono ${
                      sub.winner === "ciede2000"
                        ? "text-[#60a5fa] font-medium"
                        : "text-[#52525b]"
                    }`}
                  >
                    {sub.ciede2000}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* Where the CIEDE2000 gap lives — median-dE split per subset */}
          <h4 className="text-sm font-medium text-[#a1a1aa] mt-6 mb-1">
            Where the CIEDE2000 Gap Lives{" "}
            <span className="text-[#52525b] font-normal">(each subset split at its median ΔE*ab)</span>
          </h4>
          <p className="text-xs text-[#52525b] mb-3 leading-relaxed">
            MetricSpace's deficit on LEEDS / RIT-DuPont is almost entirely in the
            near-threshold half (ΔE*ab ≲ 1.5) — the tolerance regime CIEDE2000's
            S-functions were originally derived from. On the larger-ΔE halves the
            gap shrinks to 0.5–1.6 STRESS, and MetricSpace wins both halves on
            the much larger BFD-P sets (73% of pairs). Witt sits at its noise floor for
            both metrics.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-[#52525b]">
                  <th className="text-left font-normal py-1 pr-2">Subset</th>
                  <th className="text-right font-normal py-1 px-2">median ΔE*ab</th>
                  <th className="text-right font-normal py-1 px-2">Near-threshold half</th>
                  <th className="text-right font-normal py-1 pl-2">Larger-ΔE half</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {combvdSubsetSplit.map((r) => (
                  <tr key={r.name} className="border-t border-white/[0.04]">
                    <td className="py-1.5 pr-2 font-sans text-[#a1a1aa]">{r.name} <span className="text-[#3f3f46]">n={r.n}</span></td>
                    <td className="text-right py-1.5 px-2 text-[#52525b]">{r.medDE.toFixed(2)}</td>
                    <td className="text-right py-1.5 px-2">
                      <span className={r.smallHelm <= r.smallDE2K ? "text-[#34d399]" : "text-[#a1a1aa]"}>{r.smallHelm.toFixed(2)}</span>
                      <span className="text-[#3f3f46]"> vs </span>
                      <span className={r.smallDE2K < r.smallHelm ? "text-[#60a5fa]" : "text-[#52525b]"}>{r.smallDE2K.toFixed(2)}</span>
                    </td>
                    <td className="text-right py-1.5 pl-2">
                      <span className={r.bigHelm <= r.bigDE2K ? "text-[#34d399]" : "text-[#a1a1aa]"}>{r.bigHelm.toFixed(2)}</span>
                      <span className="text-[#3f3f46]"> vs </span>
                      <span className={r.bigDE2K < r.bigHelm ? "text-[#60a5fa]" : "text-[#52525b]"}>{r.bigDE2K.toFixed(2)}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[10px] text-[#3f3f46] mt-2">
            STRESS per half, optimal scale refit per half. Shipped v21 params, Bradford CAT to D65,
            CIEDE2000 from the same harness (half-values are internally consistent; totals table above
            uses the canonical published run — protocols agree to ≤0.05 STRESS).
          </p>
        </div>
      )}
    </div>
  );
}
