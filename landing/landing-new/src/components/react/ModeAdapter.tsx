import { useState, useEffect, useMemo } from 'react';

type HelmlabModule = typeof import('helmlab');

const DEFAULT_COLORS = ['#3b82f6', '#ef4444', '#22c55e'];
const MAX_COLORS = 5;
const MIN_COLORS = 1;

const LIGHT_BG = '#ffffff';
const DARK_BG = '#09090b';

interface AdaptedColor {
  base: string;
  light: string;
  dark: string;
  lightRatio: number;
  darkRatio: number;
}

export default function ModeAdapter() {
  const [colors, setColors] = useState<string[]>(DEFAULT_COLORS);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [copied, setCopied] = useState<string | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const hl = useMemo(() => (hlMod ? new hlMod.Helmlab() : null), [hlMod]);

  const adapted: AdaptedColor[] = useMemo(() => {
    if (!hl) {
      return colors.map((c) => ({
        base: c,
        light: c,
        dark: c,
        lightRatio: 0,
        darkRatio: 0,
      }));
    }
    return colors.map((c) => {
      const light = hl.ensureContrast(c, LIGHT_BG, 4.5);
      const dark = hl.ensureContrast(c, DARK_BG, 4.5);
      return {
        base: c,
        light,
        dark,
        lightRatio: hl.contrastRatio(light, LIGHT_BG),
        darkRatio: hl.contrastRatio(dark, DARK_BG),
      };
    });
  }, [colors, hl]);

  function updateColor(index: number, value: string) {
    const next = [...colors];
    next[index] = value.toLowerCase();
    setColors(next);
  }

  function addColor() {
    if (colors.length >= MAX_COLORS) return;
    const palette = ['#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6'];
    setColors([...colors, palette[colors.length] ?? '#a855f7']);
  }

  function removeColor(index: number) {
    if (colors.length <= MIN_COLORS) return;
    setColors(colors.filter((_, i) => i !== index));
  }

  function copy(hex: string) {
    navigator.clipboard?.writeText(hex).then(
      () => {
        setCopied(hex);
        setTimeout(() => setCopied((c) => (c === hex ? null : c)), 1200);
      },
      () => {}
    );
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <h2 className="text-xl font-semibold mb-1">Dark/Light Mode Adapter</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Enter up to {MAX_COLORS} base colors. Helmlab adjusts lightness (via GenSpace) until each
        color meets WCAG AA (4.5:1) on both white and dark backgrounds.
      </p>

      {/* Color inputs */}
      <div className="flex flex-wrap items-center gap-3 mb-8">
        {colors.map((c, i) => (
          <div
            key={i}
            className="flex items-center gap-2 rounded-lg border border-white/[0.06] bg-white/[0.03] px-2 py-1.5"
          >
            <input
              type="color"
              value={c}
              onChange={(e) => updateColor(i, e.target.value)}
              className="w-7 h-7 rounded border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
            />
            <span className="text-xs font-mono text-[#fafafa] uppercase">{c}</span>
            {colors.length > MIN_COLORS && (
              <button
                type="button"
                onClick={() => removeColor(i)}
                className="text-[#52525b] hover:text-[#f97316] text-sm leading-none px-1"
                aria-label="Remove color"
              >
                ×
              </button>
            )}
          </div>
        ))}
        {colors.length < MAX_COLORS && (
          <button
            type="button"
            onClick={addColor}
            className="rounded-lg border border-[#f97316]/40 bg-[#f97316]/10 px-3 py-1.5 text-xs font-medium text-[#f97316] hover:bg-[#f97316]/20 transition-colors"
          >
            + Add color
          </button>
        )}
      </div>

      {/* Previews — side by side on desktop, stacked on mobile */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PreviewCard
          title="Light Mode"
          bg={LIGHT_BG}
          textColor="#18181b"
          subTextColor="#52525b"
          borderColor="#e4e4e7"
          items={adapted.map((a) => ({ hex: a.light, ratio: a.lightRatio }))}
          copied={copied}
          onCopy={copy}
          loading={!hl}
        />
        <PreviewCard
          title="Dark Mode"
          bg={DARK_BG}
          textColor="#fafafa"
          subTextColor="#a1a1aa"
          borderColor="#27272a"
          items={adapted.map((a) => ({ hex: a.dark, ratio: a.darkRatio }))}
          copied={copied}
          onCopy={copy}
          loading={!hl}
        />
      </div>

      <p className="text-xs text-[#52525b] mt-6">
        Adaptation uses <span className="font-mono text-[#a1a1aa]">helmlab.ensureContrast()</span>:
        binary-searches GenSpace L along both darken/lighten directions and picks the closest
        candidate that clears the target ratio. Hue and chroma are preserved; only lightness moves.
      </p>
    </div>
  );
}

interface PreviewCardProps {
  title: string;
  bg: string;
  textColor: string;
  subTextColor: string;
  borderColor: string;
  items: { hex: string; ratio: number }[];
  copied: string | null;
  onCopy: (hex: string) => void;
  loading: boolean;
}

function PreviewCard({
  title,
  bg,
  textColor,
  subTextColor,
  borderColor,
  items,
  copied,
  onCopy,
  loading,
}: PreviewCardProps) {
  return (
    <div
      className="rounded-xl border p-5"
      style={{ backgroundColor: bg, borderColor }}
    >
      <div
        className="text-xs uppercase tracking-widest font-semibold mb-4"
        style={{ color: subTextColor }}
      >
        {title}
      </div>

      {loading ? (
        <div className="h-24 rounded-lg flex items-center justify-center" style={{ border: `1px dashed ${borderColor}` }}>
          <span className="text-xs animate-pulse" style={{ color: subTextColor }}>
            Loading helmlab.js...
          </span>
        </div>
      ) : (
        <>
          {/* Circular swatches row */}
          <div className="flex flex-wrap gap-3 mb-5">
            {items.map((it, i) => (
              <button
                key={i}
                type="button"
                onClick={() => onCopy(it.hex)}
                className="flex flex-col items-center gap-1.5 group"
                title={`${it.hex} — ${it.ratio.toFixed(2)}:1 (click to copy)`}
              >
                <div
                  className="w-12 h-12 rounded-full shadow-sm transition-transform group-hover:scale-105"
                  style={{
                    backgroundColor: it.hex,
                    border: `1px solid ${borderColor}`,
                  }}
                />
                <span
                  className="text-[10px] font-mono uppercase"
                  style={{ color: textColor }}
                >
                  {copied === it.hex ? 'copied!' : it.hex}
                </span>
                <span
                  className="text-[9px] font-mono"
                  style={{ color: subTextColor }}
                >
                  {it.ratio.toFixed(2)}:1
                </span>
              </button>
            ))}
          </div>

          {/* Sample card preview */}
          <div
            className="rounded-lg p-4"
            style={{ border: `1px solid ${borderColor}` }}
          >
            <div className="text-sm font-semibold mb-1" style={{ color: textColor }}>
              Card title
            </div>
            <div className="text-xs mb-3" style={{ color: subTextColor }}>
              Sample card text. Test the readability of the adapted colors on this background.
            </div>
            <div className="flex flex-wrap gap-2">
              {items.slice(0, 3).map((it, i) => (
                <span
                  key={i}
                  className="px-2.5 py-1 rounded-md text-[10px] font-mono font-semibold uppercase"
                  style={{
                    backgroundColor: it.hex,
                    color: bg,
                  }}
                >
                  {it.hex}
                </span>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
