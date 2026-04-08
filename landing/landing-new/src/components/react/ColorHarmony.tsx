import { useState, useEffect, useCallback, useMemo } from 'react';
import { hexToRgb, isInSrgb } from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');
type HelmlabInstance = InstanceType<HelmlabModule['Helmlab']>;

const DEFAULT_BASE = '#3b82f6';

interface Harmony {
  id: string;
  name: string;
  description: string;
  offsets: number[]; // degrees
}

const HARMONIES: Harmony[] = [
  {
    id: 'comp',
    name: 'Complementary',
    description: '180° opposite',
    offsets: [0, 180],
  },
  {
    id: 'analog',
    name: 'Analogous',
    description: '±30° neighbors',
    offsets: [-30, 0, 30],
  },
  {
    id: 'triad',
    name: 'Triadic',
    description: '120° apart',
    offsets: [0, 120, 240],
  },
  {
    id: 'tetrad',
    name: 'Tetradic',
    description: '90° apart',
    offsets: [0, 90, 180, 270],
  },
  {
    id: 'split',
    name: 'Split-Complementary',
    description: '180° ± 30°',
    offsets: [0, 150, 210],
  },
];

function contrastOnColor(hex: string): string {
  const [r, g, b] = hexToRgb(hex);
  const brightness = r * 0.299 + g * 0.587 + b * 0.114;
  return brightness > 0.55 ? '#09090b' : '#fafafa';
}

/**
 * Build a harmony hex from a base GenSpace Lab by rotating hue by `deltaDeg`,
 * keeping L constant and clipping chroma into the sRGB gamut via binary search.
 */
function rotateHueInGen(
  hl: HelmlabInstance,
  baseLab: [number, number, number],
  deltaDeg: number
): string {
  const [L, a, b] = baseLab;
  const C = Math.sqrt(a * a + b * b);
  const hBase = Math.atan2(b, a);
  const hNew = hBase + (deltaDeg * Math.PI) / 180;
  const cosH = Math.cos(hNew);
  const sinH = Math.sin(hNew);

  // Try full chroma first
  const tryHex = hl.genToHex([L, C * cosH, C * sinH]);
  const [tr, tg, tb] = hexToRgb(tryHex);
  // genToHex already gamut-maps, but we prefer a chroma-only reduction to preserve hue/L
  // Binary search for max in-gamut chroma at this (L, hNew)
  let lo = 0;
  let hi = C;
  for (let i = 0; i < 22; i++) {
    const mid = (lo + hi) / 2;
    const hex = hl.genToHex([L, mid * cosH, mid * sinH]);
    // genToHex already gamut-maps; if clipping happened, the round-tripped
    // chroma will be less than `mid`. This isolates chroma reduction only.
    const labRt = hl.genFromHex(hex);
    const cRt = Math.sqrt(labRt[1] * labRt[1] + labRt[2] * labRt[2]);
    if (cRt >= mid - 1e-3) lo = mid;
    else hi = mid;
  }
  const cUse = lo;
  // Prefer exact full-chroma hex if it round-trips well (no clipping needed)
  const fullLabRt = hl.genFromHex(tryHex);
  const fullC = Math.sqrt(fullLabRt[1] * fullLabRt[1] + fullLabRt[2] * fullLabRt[2]);
  if (fullC >= C - 1e-3 && isInSrgb(tr, tg, tb)) return tryHex;
  return hl.genToHex([L, cUse * cosH, cUse * sinH]);
}

interface HarmonyResult {
  id: string;
  name: string;
  description: string;
  colors: { hex: string; angle: number }[];
}

export default function ColorHarmony() {
  const [baseHex, setBaseHex] = useState(DEFAULT_BASE);
  const [baseInput, setBaseInput] = useState(DEFAULT_BASE);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [results, setResults] = useState<HarmonyResult[]>([]);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const compute = useCallback((hex: string, mod: HelmlabModule | null) => {
    if (!mod) {
      setResults([]);
      return;
    }
    const hl = new mod.Helmlab();
    const baseLab = hl.genFromHex(hex);
    const baseAngle =
      ((Math.atan2(baseLab[2], baseLab[1]) * 180) / Math.PI + 360) % 360;

    const next: HarmonyResult[] = HARMONIES.map((h) => ({
      id: h.id,
      name: h.name,
      description: h.description,
      colors: h.offsets.map((off) => ({
        hex: rotateHueInGen(hl, baseLab, off),
        angle: (baseAngle + off + 360) % 360,
      })),
    }));
    setResults(next);
  }, []);

  useEffect(() => {
    compute(baseHex, hlMod);
  }, [baseHex, hlMod, compute]);

  function handleBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(baseInput)) setBaseHex(baseInput.toLowerCase());
    else setBaseInput(baseHex);
  }

  function handleCopy(hex: string) {
    const up = hex.toUpperCase();
    navigator.clipboard
      .writeText(up)
      .then(() => {
        setCopied(up);
        setTimeout(() => setCopied((c) => (c === up ? null : c)), 1400);
      })
      .catch(() => {});
  }

  const loading = useMemo(() => hlMod === null, [hlMod]);

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8 relative">
      <h2 className="text-xl font-semibold mb-1">Color Harmony Generator</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Generate classic color harmonies in Helmlab GenSpace. Hue rotations preserve
        lightness and chroma, with in-gamut clipping per target.
      </p>

      {/* Base color input */}
      <div className="flex items-center gap-3 mb-8">
        <input
          type="color"
          value={baseHex}
          onChange={(e) => {
            setBaseHex(e.target.value);
            setBaseInput(e.target.value);
          }}
          className="w-10 h-10 rounded-lg border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
        />
        <input
          type="text"
          value={baseInput}
          onChange={(e) => setBaseInput(e.target.value)}
          onBlur={handleBlur}
          onKeyDown={(e) => e.key === 'Enter' && handleBlur()}
          className="w-28 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
          spellCheck={false}
        />
        <span className="text-sm text-[#a1a1aa]">Base color</span>
      </div>

      {loading ? (
        <div className="h-24 rounded-lg bg-white/[0.03] flex items-center justify-center">
          <span className="text-xs text-[#52525b] animate-pulse">Loading helmlab.js...</span>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {results.map((h) => (
            <div
              key={h.id}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5"
            >
              <div className="flex items-baseline justify-between mb-4">
                <div>
                  <div className="text-sm font-semibold text-[#fafafa]">{h.name}</div>
                  <div className="text-[11px] text-[#52525b]">{h.description}</div>
                </div>
                <div className="text-[10px] text-[#52525b] font-mono">
                  {h.colors.length} colors
                </div>
              </div>

              <div
                className="grid gap-2 mb-3"
                style={{
                  gridTemplateColumns: `repeat(${h.colors.length}, minmax(0, 1fr))`,
                }}
              >
                {h.colors.map((c, i) => {
                  const up = c.hex.toUpperCase();
                  const isCopied = copied === up;
                  return (
                    <button
                      key={`${h.id}-${i}`}
                      type="button"
                      onClick={() => handleCopy(c.hex)}
                      className="group relative aspect-square rounded-lg border border-white/[0.06] overflow-hidden focus:outline-none focus:ring-2 focus:ring-[#f97316]/40 transition-transform hover:scale-[1.03]"
                      style={{ backgroundColor: c.hex }}
                      title={`${up} · ${c.angle.toFixed(0)}°`}
                    >
                      <span
                        className="absolute inset-0 flex items-center justify-center text-[10px] font-mono opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ color: contrastOnColor(c.hex) }}
                      >
                        {isCopied ? 'Copied' : 'Copy'}
                      </span>
                    </button>
                  );
                })}
              </div>

              <div
                className="grid gap-2"
                style={{
                  gridTemplateColumns: `repeat(${h.colors.length}, minmax(0, 1fr))`,
                }}
              >
                {h.colors.map((c, i) => (
                  <div
                    key={`${h.id}-label-${i}`}
                    className="text-center text-[10px] font-mono text-[#a1a1aa] truncate"
                  >
                    {c.hex.toUpperCase()}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-[#52525b] mt-6">
        Click a swatch to copy its hex. Harmonies rotate hue around the base color in
        Helmlab GenSpace LCh — lightness stays constant, chroma is clipped per target
        to keep every color in sRGB.
      </p>

      {copied && (
        <div className="pointer-events-none fixed bottom-6 left-1/2 -translate-x-1/2 rounded-lg bg-[#18181b] border border-white/[0.08] px-4 py-2 text-xs font-mono text-[#fafafa] shadow-xl z-50">
          Copied {copied}
        </div>
      )}
    </div>
  );
}
