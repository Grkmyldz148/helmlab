import { useState, useEffect, useCallback } from 'react';
import {
  interpolateOklab,
  interpolateCieLab,
  gradientCV,
} from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const PRESETS: { label: string; start: string; end: string }[] = [
  { label: 'Blue \u2192 White', start: '#3b82f6', end: '#ffffff' },
  { label: 'Red \u2192 White', start: '#ef4444', end: '#ffffff' },
  { label: 'Orange \u2192 Cyan', start: '#f97316', end: '#06b6d4' },
  { label: 'Green \u2192 Pink', start: '#22c55e', end: '#ec4899' },
  { label: 'Yellow \u2192 Purple', start: '#eab308', end: '#8b5cf6' },
];

const STEPS = 16;

interface GradientRow {
  label: string;
  colors: string[];
  midpoint: string;
  cv: number;
}

export default function GradientTool() {
  const [start, setStart] = useState(PRESETS[0].start);
  const [end, setEnd] = useState(PRESETS[0].end);
  const [startInput, setStartInput] = useState(PRESETS[0].start);
  const [endInput, setEndInput] = useState(PRESETS[0].end);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [rows, setRows] = useState<GradientRow[]>([]);

  // Load helmlab dynamically
  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const computeGradients = useCallback(
    (s: string, e: string, mod: HelmlabModule | null) => {
      const oklabColors = interpolateOklab(s, e, STEPS);
      const cielabColors = interpolateCieLab(s, e, STEPS);

      const oklabRow: GradientRow = {
        label: 'OKLab',
        colors: oklabColors,
        midpoint: oklabColors[Math.floor(STEPS / 2)],
        cv: gradientCV(oklabColors),
      };
      const cielabRow: GradientRow = {
        label: 'CIE Lab',
        colors: cielabColors,
        midpoint: cielabColors[Math.floor(STEPS / 2)],
        cv: gradientCV(cielabColors),
      };

      if (mod) {
        const hl = new mod.Helmlab();
        const helmlabColors = hl.gradient(s, e, STEPS);
        const helmlabRow: GradientRow = {
          label: 'Helmlab',
          colors: helmlabColors,
          midpoint: helmlabColors[Math.floor(STEPS / 2)],
          cv: gradientCV(helmlabColors),
        };
        setRows([helmlabRow, oklabRow, cielabRow]);
      } else {
        setRows([
          { label: 'Helmlab', colors: [], midpoint: '...', cv: 0 },
          oklabRow,
          cielabRow,
        ]);
      }
    },
    []
  );

  useEffect(() => {
    computeGradients(start, end, hlMod);
  }, [start, end, hlMod, computeGradients]);

  function applyPreset(preset: (typeof PRESETS)[number]) {
    setStart(preset.start);
    setEnd(preset.end);
    setStartInput(preset.start);
    setEndInput(preset.end);
  }

  function handleStartBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(startInput)) setStart(startInput.toLowerCase());
    else setStartInput(start);
  }

  function handleEndBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(endInput)) setEnd(endInput.toLowerCase());
    else setEndInput(end);
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <h2 className="text-xl font-semibold mb-1">Gradient Comparison</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Pick two colors and compare 16-step gradients across three spaces.
      </p>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 mb-6">
        {/* Start color */}
        <div className="flex items-center gap-2">
          <input
            type="color"
            value={start}
            onChange={(e) => {
              setStart(e.target.value);
              setStartInput(e.target.value);
            }}
            className="w-10 h-10 rounded-lg border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
          />
          <input
            type="text"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
            onBlur={handleStartBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleStartBlur()}
            className="w-24 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
            spellCheck={false}
          />
        </div>

        <span className="hidden sm:flex items-center text-[#52525b]">&rarr;</span>

        {/* End color */}
        <div className="flex items-center gap-2">
          <input
            type="color"
            value={end}
            onChange={(e) => {
              setEnd(e.target.value);
              setEndInput(e.target.value);
            }}
            className="w-10 h-10 rounded-lg border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
          />
          <input
            type="text"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
            onBlur={handleEndBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleEndBlur()}
            className="w-24 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
            spellCheck={false}
          />
        </div>
      </div>

      {/* Presets */}
      <div className="flex flex-wrap gap-2 mb-8">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            onClick={() => applyPreset(p)}
            className={`rounded-full border px-3 py-1 text-xs transition-colors ${
              start === p.start && end === p.end
                ? 'border-[#f97316]/40 bg-[#f97316]/10 text-[#f97316]'
                : 'border-white/[0.06] text-[#a1a1aa] hover:border-white/[0.12] hover:text-[#fafafa]'
            }`}
          >
            {p.label}
          </button>
        ))}
      </div>

      {/* Gradient strips */}
      <div className="space-y-8">
        {rows.map((row) => (
          <div key={row.label}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-[#fafafa]">
                {row.label}
                {row.label === 'Helmlab' && (
                  <span className="ml-2 text-xs text-[#f97316]">GenSpace + arc-length reparam.</span>
                )}
              </span>
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-[#52525b]">
                  mid: {row.midpoint}
                </span>
                {row.colors.length > 0 && (
                  <span
                    className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
                      row.cv <= Math.min(...rows.filter(r => r.colors.length > 0).map(r => r.cv)) + 0.5
                        ? 'border-[#34d399]/30 text-[#34d399] bg-[#34d399]/5'
                        : row.cv >= Math.max(...rows.filter(r => r.colors.length > 0).map(r => r.cv)) - 0.5
                        ? 'border-[#f87171]/30 text-[#f87171] bg-[#f87171]/5'
                        : 'border-white/[0.06] text-[#a1a1aa]'
                    }`}
                  >
                    {row.cv.toFixed(1)}%
                  </span>
                )}
              </div>
            </div>
            {row.colors.length > 0 ? (
              <div className="flex h-12 rounded-lg overflow-hidden">
                {row.colors.map((c, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{ backgroundColor: c }}
                    title={c}
                  />
                ))}
              </div>
            ) : (
              <div className="flex h-12 rounded-lg overflow-hidden bg-white/[0.03] items-center justify-center">
                <span className="text-xs text-[#52525b] animate-pulse">Loading helmlab.js...</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Explanation */}
      <p className="text-xs text-[#52525b] mt-6">
        Helmlab gradients use CIEDE2000 arc-length reparameterization for perceptually equal step sizes.
        OKLab and CIE Lab use simple linear interpolation. At the blue midpoint Helmlab stays a little more
        saturated than OKLab — both stay blue, the difference is subtle.
      </p>
    </div>
  );
}
