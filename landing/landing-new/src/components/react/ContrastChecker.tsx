import { useState, useEffect, useCallback, useMemo } from 'react';
import { hexToRgb, relativeLuminance } from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const DEFAULT_FG = '#3b82f6';
const DEFAULT_BG = '#ffffff';

interface WcagResults {
  ratio: number;
  aaNormal: boolean;
  aaLarge: boolean;
  aaaNormal: boolean;
  aaaLarge: boolean;
}

function wcagContrast(fg: string, bg: string): number {
  const [r1, g1, b1] = hexToRgb(fg);
  const [r2, g2, b2] = hexToRgb(bg);
  const l1 = relativeLuminance(r1, g1, b1);
  const l2 = relativeLuminance(r2, g2, b2);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

function evaluate(fg: string, bg: string): WcagResults {
  const ratio = wcagContrast(fg, bg);
  return {
    ratio,
    aaLarge: ratio >= 3.0,
    aaNormal: ratio >= 4.5,
    aaaLarge: ratio >= 4.5,
    aaaNormal: ratio >= 7.0,
  };
}

export default function ContrastChecker() {
  const [fg, setFg] = useState(DEFAULT_FG);
  const [bg, setBg] = useState(DEFAULT_BG);
  const [fgInput, setFgInput] = useState(DEFAULT_FG);
  const [bgInput, setBgInput] = useState(DEFAULT_BG);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [fixedFg, setFixedFg] = useState<string | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  // Reset the "fixed" preview when the user manually changes inputs
  useEffect(() => {
    setFixedFg(null);
  }, [fg, bg]);

  const results = useMemo(() => evaluate(fg, bg), [fg, bg]);
  const fixedResults = useMemo(
    () => (fixedFg ? evaluate(fixedFg, bg) : null),
    [fixedFg, bg]
  );

  const handleFgBlur = () => {
    if (/^#[0-9a-fA-F]{6}$/.test(fgInput)) setFg(fgInput.toLowerCase());
    else setFgInput(fg);
  };
  const handleBgBlur = () => {
    if (/^#[0-9a-fA-F]{6}$/.test(bgInput)) setBg(bgInput.toLowerCase());
    else setBgInput(bg);
  };

  const autoFix = useCallback(() => {
    // Preferred: use Helmlab's built-in ensureContrast (GenSpace-based L adjust)
    if (hlMod) {
      try {
        const hl = new hlMod.Helmlab();
        const fixed = hl.ensureContrast(fg, bg, 4.5);
        setFixedFg(fixed.toLowerCase());
        return;
      } catch {
        // fall through to HSL fallback
      }
    }
    // Fallback: HSL lightness binary search
    setFixedFg(hslFallbackFix(fg, bg, 4.5));
  }, [fg, bg, hlMod]);

  const applyFixed = () => {
    if (!fixedFg) return;
    setFg(fixedFg);
    setFgInput(fixedFg);
    setFixedFg(null);
  };

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <h2 className="text-xl font-semibold mb-1">Contrast Checker & Fixer</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Check WCAG contrast between two colors. Auto-fix adjusts the foreground
        lightness in Helmlab GenSpace while preserving hue.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Controls ────────────────────────────────── */}
        <div className="flex flex-col gap-5">
          {/* Foreground */}
          <div>
            <label className="text-xs font-medium text-[#a1a1aa] uppercase tracking-wider block mb-2">
              Foreground
            </label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                value={fg}
                onChange={(e) => {
                  setFg(e.target.value);
                  setFgInput(e.target.value);
                }}
                className="w-12 h-12 rounded-xl border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
              />
              <input
                type="text"
                value={fgInput}
                onChange={(e) => setFgInput(e.target.value)}
                onBlur={handleFgBlur}
                onKeyDown={(e) => e.key === 'Enter' && handleFgBlur()}
                className="w-28 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
                spellCheck={false}
              />
            </div>
          </div>

          {/* Background */}
          <div>
            <label className="text-xs font-medium text-[#a1a1aa] uppercase tracking-wider block mb-2">
              Background
            </label>
            <div className="flex items-center gap-3">
              <input
                type="color"
                value={bg}
                onChange={(e) => {
                  setBg(e.target.value);
                  setBgInput(e.target.value);
                }}
                className="w-12 h-12 rounded-xl border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
              />
              <input
                type="text"
                value={bgInput}
                onChange={(e) => setBgInput(e.target.value)}
                onBlur={handleBgBlur}
                onKeyDown={(e) => e.key === 'Enter' && handleBgBlur()}
                className="w-28 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
                spellCheck={false}
              />
            </div>
          </div>

          {/* Ratio + badges */}
          <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-5">
            <div className="flex items-baseline gap-3">
              <span className="text-4xl font-bold font-mono text-[#fafafa] tracking-tight">
                {results.ratio.toFixed(2)}
              </span>
              <span className="text-base text-[#52525b] font-mono">:1</span>
              <span className="ml-auto text-xs text-[#52525b]">WCAG 2.1</span>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              <Badge label="AA Normal" pass={results.aaNormal} threshold="≥ 4.5" />
              <Badge label="AA Large" pass={results.aaLarge} threshold="≥ 3.0" />
              <Badge label="AAA Normal" pass={results.aaaNormal} threshold="≥ 7.0" />
              <Badge label="AAA Large" pass={results.aaaLarge} threshold="≥ 4.5" />
            </div>
          </div>

          {/* Auto-fix */}
          <div className="flex items-center gap-3">
            <button
              onClick={autoFix}
              className="rounded-lg border border-[#f97316]/40 bg-[#f97316]/10 px-4 py-2 text-sm font-medium text-[#f97316] hover:bg-[#f97316]/20 transition-colors"
            >
              Auto-fix foreground (AA)
            </button>
            {fixedFg && (
              <button
                onClick={applyFixed}
                className="rounded-lg border border-white/[0.06] bg-white/[0.03] px-4 py-2 text-sm font-medium text-[#fafafa] hover:bg-white/[0.06] transition-colors"
              >
                Apply
              </button>
            )}
          </div>
          <p className="text-[11px] text-[#52525b] leading-relaxed">
            {hlMod
              ? 'Uses helmlab.ensureContrast — binary-searches L in Helmlab GenSpace while preserving hue and chroma.'
              : 'Helmlab not yet loaded — falling back to HSL lightness search.'}
          </p>
        </div>

        {/* ── Preview ─────────────────────────────────── */}
        <div className="flex flex-col gap-4">
          <div
            className="rounded-xl border border-white/[0.06] p-6 transition-colors"
            style={{ backgroundColor: bg }}
          >
            <div
              className="text-2xl font-semibold mb-2"
              style={{ color: fg }}
            >
              The quick brown fox
            </div>
            <div className="text-base mb-1" style={{ color: fg }}>
              Sample body text at a normal reading size (16 px).
            </div>
            <div className="text-sm mb-4" style={{ color: fg, opacity: 0.9 }}>
              Smaller caption text at 14 px for fine print.
            </div>
            <button
              className="px-4 py-2 rounded-lg font-semibold text-sm"
              style={{ backgroundColor: fg, color: bg }}
            >
              Primary Button
            </button>
            <div className="mt-4 text-[10px] font-mono" style={{ color: fg, opacity: 0.6 }}>
              {fg.toUpperCase()} on {bg.toUpperCase()}
            </div>
          </div>

          {/* Before / After */}
          {fixedFg && fixedResults && (
            <div className="rounded-xl border border-[#34d399]/30 bg-[#34d399]/5 p-4">
              <div className="text-xs font-medium text-[#34d399] uppercase tracking-wider mb-3">
                Auto-fix result
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div
                  className="rounded-lg p-3 border border-white/[0.06]"
                  style={{ backgroundColor: bg }}
                >
                  <div className="text-[10px] uppercase font-mono mb-1" style={{ color: fg, opacity: 0.7 }}>
                    Before
                  </div>
                  <div className="text-sm font-semibold" style={{ color: fg }}>
                    Sample text
                  </div>
                  <div className="text-[10px] font-mono mt-2" style={{ color: fg, opacity: 0.7 }}>
                    {fg.toUpperCase()} · {results.ratio.toFixed(2)}:1
                  </div>
                </div>
                <div
                  className="rounded-lg p-3 border border-white/[0.06]"
                  style={{ backgroundColor: bg }}
                >
                  <div className="text-[10px] uppercase font-mono mb-1" style={{ color: fixedFg, opacity: 0.7 }}>
                    After
                  </div>
                  <div className="text-sm font-semibold" style={{ color: fixedFg }}>
                    Sample text
                  </div>
                  <div className="text-[10px] font-mono mt-2" style={{ color: fixedFg, opacity: 0.7 }}>
                    {fixedFg.toUpperCase()} · {fixedResults.ratio.toFixed(2)}:1
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Badge({
  label,
  pass,
  threshold,
}: {
  label: string;
  pass: boolean;
  threshold: string;
}) {
  return (
    <div
      className={`flex items-center justify-between rounded-lg px-3 py-2 border ${
        pass
          ? 'border-[#34d399]/30 bg-[#34d399]/5'
          : 'border-[#f87171]/30 bg-[#f87171]/5'
      }`}
    >
      <div className="flex flex-col">
        <span
          className={`text-[11px] font-semibold ${
            pass ? 'text-[#34d399]' : 'text-[#f87171]'
          }`}
        >
          {label}
        </span>
        <span className="text-[9px] font-mono text-[#52525b]">{threshold}</span>
      </div>
      <span
        className={`text-[10px] font-bold tracking-wider ${
          pass ? 'text-[#34d399]' : 'text-[#f87171]'
        }`}
      >
        {pass ? 'PASS' : 'FAIL'}
      </span>
    </div>
  );
}

// ── HSL fallback auto-fix ─────────────────────────────────────
function rgbToHsl(r: number, g: number, b: number): [number, number, number] {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  let h = 0;
  let s = 0;
  const d = max - min;
  if (d !== 0) {
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h /= 6;
  }
  return [h, s, l];
}

function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  if (s === 0) return [l, l, l];
  const hue2rgb = (p: number, q: number, t: number) => {
    if (t < 0) t += 1;
    if (t > 1) t -= 1;
    if (t < 1 / 6) return p + (q - p) * 6 * t;
    if (t < 1 / 2) return q;
    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

function rgbToHexStr(r: number, g: number, b: number): string {
  const c = (v: number) =>
    Math.max(0, Math.min(255, Math.round(v * 255)))
      .toString(16)
      .padStart(2, '0');
  return '#' + c(r) + c(g) + c(b);
}

function hslFallbackFix(fg: string, bg: string, minRatio: number): string {
  if (wcagContrast(fg, bg) >= minRatio) return fg;
  const [fr, fg_, fb] = hexToRgb(fg);
  const [h, s] = rgbToHsl(fr, fg_, fb);
  let best = fg;
  let bestGap = Infinity;
  for (const dir of [-1, 1]) {
    for (let step = 1; step <= 100; step++) {
      const l = Math.min(1, Math.max(0, 0.5 + dir * step * 0.005));
      const [r, g, b] = hslToRgb(h, s, l);
      const hex = rgbToHexStr(r, g, b);
      const ratio = wcagContrast(hex, bg);
      if (ratio >= minRatio) {
        const gap = ratio - minRatio;
        if (gap < bestGap) {
          bestGap = gap;
          best = hex;
        }
        break;
      }
    }
  }
  if (bestGap === Infinity) {
    return wcagContrast('#000000', bg) > wcagContrast('#ffffff', bg)
      ? '#000000'
      : '#ffffff';
  }
  return best;
}
