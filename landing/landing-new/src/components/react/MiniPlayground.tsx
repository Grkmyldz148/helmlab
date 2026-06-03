import { useState, useEffect, useCallback } from 'react';
import {
  interpolateOklab,
  interpolateCieLab,
  gradientCV,
} from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const STEPS = 16;
const DEFAULT_START = '#3b82f6';
const DEFAULT_END = '#ffffff';

interface GradientRow {
  label: string;
  colors: string[];
  cv: number;
}

export default function MiniPlayground() {
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [rows, setRows] = useState<GradientRow[]>([]);
  const [start, setStart] = useState(DEFAULT_START);
  const [end, setEnd] = useState(DEFAULT_END);
  const [startInput, setStartInput] = useState(DEFAULT_START);
  const [endInput, setEndInput] = useState(DEFAULT_END);
  const [conf, setConf] = useState<{ de: number; reliability: number; reliable: boolean } | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const computeGradients = useCallback(
    (s: string, e: string, mod: HelmlabModule | null) => {
      const oklabColors = interpolateOklab(s, e, STEPS);
      const cielabColors = interpolateCieLab(s, e, STEPS);

      const oklabRow: GradientRow = { label: 'OKLab', colors: oklabColors, cv: gradientCV(oklabColors) };
      const cielabRow: GradientRow = { label: 'CIE Lab', colors: cielabColors, cv: gradientCV(cielabColors) };

      if (mod) {
        const hl = new mod.Helmlab();
        const helmlabColors = hl.gradient(s, e, STEPS);
        // Beta: how different the two endpoints are, and how reliable that is.
        const c = hl.differenceWithConfidence(s, e);
        setConf({ de: c.de, reliability: c.reliability, reliable: c.reliable });
        setRows([
          { label: 'Helmlab', colors: helmlabColors, cv: gradientCV(helmlabColors) },
          oklabRow,
          cielabRow,
        ]);
      } else {
        setConf(null);
        setRows([
          { label: 'Helmlab', colors: [], cv: 0 },
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

  function handleStartBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(startInput)) setStart(startInput.toLowerCase());
    else setStartInput(start);
  }

  function handleEndBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(endInput)) setEnd(endInput.toLowerCase());
    else setEndInput(end);
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div>
          <h3 className="text-lg font-semibold">See the Difference</h3>
          <p className="text-xs text-[#a1a1aa] mt-0.5">
            Three color spaces, same endpoints. Notice the midpoint.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            type="color"
            value={start}
            onChange={(e) => {
              setStart(e.target.value);
              setStartInput(e.target.value);
            }}
            className="w-8 h-8 rounded-md border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
          />
          <input
            type="text"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
            onBlur={handleStartBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleStartBlur()}
            className="w-20 rounded-md border border-white/[0.06] bg-white/[0.03] px-2 py-1.5 text-xs font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
            spellCheck={false}
          />
          <span className="text-[#52525b] text-xs">&rarr;</span>
          <input
            type="color"
            value={end}
            onChange={(e) => {
              setEnd(e.target.value);
              setEndInput(e.target.value);
            }}
            className="w-8 h-8 rounded-md border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
          />
          <input
            type="text"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
            onBlur={handleEndBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleEndBlur()}
            className="w-20 rounded-md border border-white/[0.06] bg-white/[0.03] px-2 py-1.5 text-xs font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
            spellCheck={false}
          />
        </div>
      </div>

      {/* Gradient strips */}
      <div className="space-y-6">
        {rows.map((row) => (
          <div key={row.label}>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-[#a1a1aa]">
                {row.label}
                {row.label === 'Helmlab' && (
                  <span className="text-[#f97316] ml-1">*</span>
                )}
              </span>
              {row.colors.length > 0 && (
                <span
                  className={`text-xs font-mono px-1.5 py-0.5 rounded-full border ${
                    row.cv <= Math.min(...rows.filter(r => r.colors.length > 0).map(r => r.cv)) + 0.5
                      ? 'border-[#34d399]/30 text-[#34d399]'
                      : row.cv >= Math.max(...rows.filter(r => r.colors.length > 0).map(r => r.cv)) - 0.5
                      ? 'border-[#f87171]/30 text-[#f87171]'
                      : 'border-white/[0.06] text-[#a1a1aa]'
                  }`}
                >
                  {row.cv.toFixed(1)}%
                </span>
              )}
            </div>
            {row.colors.length > 0 ? (
              <div className="flex h-10 rounded-lg overflow-hidden">
                {row.colors.map((c, i) => (
                  <div
                    key={i}
                    className="flex-1"
                    style={{ backgroundColor: c }}
                  />
                ))}
              </div>
            ) : (
              <div className="flex h-10 rounded-lg overflow-hidden bg-white/[0.03] items-center justify-center">
                <span className="text-[10px] text-[#52525b] animate-pulse">Loading...</span>
              </div>
            )}
          </div>
        ))}
      </div>

      {conf && (
        <div className="mt-5 flex items-center justify-between rounded-lg border border-white/[0.06] bg-white/[0.02] px-3 py-2">
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#a1a1aa]">Endpoint difference</span>
            <span className="text-[10px] font-mono uppercase tracking-wide text-[#f97316] border border-[#f97316]/30 rounded px-1 py-0.5">
              beta
            </span>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="text-[#a1a1aa]">ΔE {conf.de.toFixed(3)}</span>
            <span className={conf.reliable ? 'text-[#34d399]' : 'text-[#f87171]'}>
              {conf.reliable ? 'reliable' : 'within human noise'} · {(conf.reliability * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between mt-4">
        <span className="text-[10px] text-[#52525b]">
          * Perceptually uniform via CIEDE2000 arc-length reparameterization
        </span>
        <a
          href="/playground"
          className="text-xs text-[#f97316] hover:text-[#fb923c] transition-colors"
        >
          Full playground &rarr;
        </a>
      </div>
    </div>
  );
}
