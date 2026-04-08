import { useState, useMemo } from "react";
import {
  genspaceResults,
  headlineScore,
  type MetricCategory,
  type Winner,
  type GenSpaceMetric,
} from "../../data/genspace-results";

type SortKey = "name" | "category" | "oklab" | "genspace" | "winner";
type SortDir = "asc" | "desc";

const allCategories: MetricCategory[] = [
  "Gamut",
  "Gradient",
  "Application",
  "Perceptual",
  "Structural",
  "Hue",
  "Achromatic",
  "Advanced",
  "Special",
  "Banding",
  "Accessibility",
  "Numerical",
];

function formatValue(v: number | string): string {
  if (typeof v === "string") return v;
  if (v === 0) return "0";
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString();
  return v.toLocaleString(undefined, { maximumFractionDigits: 3 });
}

function winnerLabel(w: Winner): string {
  return w === "genspace" ? "GenSpace" : w === "oklab" ? "OKLab" : "Tie";
}

function winnerColor(w: Winner): string {
  return w === "genspace"
    ? "text-[#34d399]"
    : w === "oklab"
      ? "text-[#f87171]"
      : "text-[#52525b]";
}

function winnerBg(w: Winner): string {
  return w === "genspace"
    ? "bg-[#34d399]/10 text-[#34d399]"
    : w === "oklab"
      ? "bg-[#f87171]/10 text-[#f87171]"
      : "bg-white/[0.04] text-[#52525b]";
}

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active) {
    return (
      <svg className="inline-block ml-1 h-3 w-3 text-[#3f3f46]" viewBox="0 0 12 12" fill="currentColor">
        <path d="M6 2l3 4H3zM6 10l-3-4h6z" />
      </svg>
    );
  }
  return (
    <svg className="inline-block ml-1 h-3 w-3 text-[#f97316]" viewBox="0 0 12 12" fill="currentColor">
      {dir === "asc" ? <path d="M6 2l3 4H3z" /> : <path d="M6 10l-3-4h6z" />}
    </svg>
  );
}

function getNumericValue(metric: GenSpaceMetric, key: "oklab" | "genspace"): number {
  const v = metric[key];
  if (typeof v === "number") return v;
  // Parse strings like "360/360", "5.79x"
  const num = parseFloat(v.replace(/[^\d.-]/g, ""));
  return isNaN(num) ? 0 : num;
}

export default function BenchmarkExplorer() {
  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState<MetricCategory | "All">("All");
  const [winnerFilter, setWinnerFilter] = useState<Winner | "All">("All");
  const [sortKey, setSortKey] = useState<SortKey>("category");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  const filtered = useMemo(() => {
    let data = [...genspaceResults];

    // Text search
    if (search.trim()) {
      const q = search.toLowerCase();
      data = data.filter(
        (m) => m.name.toLowerCase().includes(q) || m.category.toLowerCase().includes(q)
      );
    }

    // Category filter
    if (categoryFilter !== "All") {
      data = data.filter((m) => m.category === categoryFilter);
    }

    // Winner filter
    if (winnerFilter !== "All") {
      data = data.filter((m) => m.winner === winnerFilter);
    }

    // Sort
    data.sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case "name":
          cmp = a.name.localeCompare(b.name);
          break;
        case "category":
          cmp = a.category.localeCompare(b.category) || a.name.localeCompare(b.name);
          break;
        case "oklab":
          cmp = getNumericValue(a, "oklab") - getNumericValue(b, "oklab");
          break;
        case "genspace":
          cmp = getNumericValue(a, "genspace") - getNumericValue(b, "genspace");
          break;
        case "winner": {
          const order: Record<Winner, number> = { genspace: 0, tie: 1, oklab: 2 };
          cmp = order[a.winner] - order[b.winner];
          break;
        }
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return data;
  }, [search, categoryFilter, winnerFilter, sortKey, sortDir]);

  const winCount = filtered.filter((m) => m.winner === "genspace").length;
  const lossCount = filtered.filter((m) => m.winner === "oklab").length;
  const tieCount = filtered.filter((m) => m.winner === "tie").length;

  return (
    <div>
      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3 mb-4">
        {/* Search */}
        <div className="relative flex-1">
          <svg
            className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-[#52525b]"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <circle cx="11" cy="11" r="8" />
            <path d="M21 21l-4.35-4.35" />
          </svg>
          <input
            type="text"
            placeholder="Search metrics..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-white/[0.06] bg-white/[0.02] pl-9 pr-3 py-2 text-sm text-[#fafafa] placeholder-[#3f3f46] outline-none focus:border-[#f97316]/40 transition-colors"
          />
        </div>

        {/* Category dropdown */}
        <select
          value={categoryFilter}
          onChange={(e) => setCategoryFilter(e.target.value as MetricCategory | "All")}
          className="rounded-lg border border-white/[0.06] bg-[#0c0c0f] px-3 py-2 text-sm text-[#a1a1aa] outline-none focus:border-[#f97316]/40 transition-colors"
        >
          <option value="All">All Categories</option>
          {allCategories.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>

        {/* Winner filter */}
        <select
          value={winnerFilter}
          onChange={(e) => setWinnerFilter(e.target.value as Winner | "All")}
          className="rounded-lg border border-white/[0.06] bg-[#0c0c0f] px-3 py-2 text-sm text-[#a1a1aa] outline-none focus:border-[#f97316]/40 transition-colors"
        >
          <option value="All">All Results</option>
          <option value="genspace">GenSpace Wins</option>
          <option value="oklab">OKLab Wins</option>
          <option value="tie">Ties</option>
        </select>
      </div>

      {/* Table */}
      <div className="overflow-x-auto rounded-xl border border-white/[0.06]">
        <table className="w-full text-sm min-w-[640px]">
          <thead>
            <tr className="border-b border-white/[0.06] bg-white/[0.02]">
              {[
                { key: "name" as SortKey, label: "Metric", align: "text-left" },
                { key: "category" as SortKey, label: "Category", align: "text-left" },
                { key: "oklab" as SortKey, label: "OKLab", align: "text-right" },
                { key: "genspace" as SortKey, label: "GenSpace", align: "text-right" },
                { key: "winner" as SortKey, label: "Winner", align: "text-center" },
              ].map((col) => (
                <th
                  key={col.key}
                  className={`px-4 py-3 text-xs font-medium text-[#52525b] uppercase tracking-wider cursor-pointer hover:text-[#a1a1aa] transition-colors select-none ${col.align}`}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                  <SortIcon active={sortKey === col.key} dir={sortDir} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map((m, i) => (
              <tr
                key={m.name}
                className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors"
              >
                <td className="px-4 py-2.5 text-[#fafafa] font-medium">
                  {m.name}
                  {m.unit && (
                    <span className="ml-1.5 text-[10px] text-[#3f3f46]">{m.unit}</span>
                  )}
                </td>
                <td className="px-4 py-2.5">
                  <span className="inline-block rounded-full bg-white/[0.04] px-2 py-0.5 text-xs text-[#52525b]">
                    {m.category}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right font-mono text-[#a1a1aa] text-xs">
                  {formatValue(m.oklab)}
                </td>
                <td className={`px-4 py-2.5 text-right font-mono text-xs font-medium ${winnerColor(m.winner)}`}>
                  {formatValue(m.genspace)}
                </td>
                <td className="px-4 py-2.5 text-center">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-xs font-medium ${winnerBg(m.winner)}`}
                  >
                    {winnerLabel(m.winner)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary footer */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[#52525b]">
        <span>
          Showing {filtered.length} of {genspaceResults.length} metrics
        </span>
        <div className="flex items-center gap-4">
          <span className="text-[#34d399]">{winCount} GenSpace</span>
          <span>{tieCount} Ties</span>
          <span className="text-[#f87171]">{lossCount} OKLab</span>
        </div>
      </div>
    </div>
  );
}
