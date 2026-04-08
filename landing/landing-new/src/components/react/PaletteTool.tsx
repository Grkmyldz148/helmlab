import { useState, useEffect, useCallback } from 'react';
import { oklabPalette } from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const SCALE_LABELS = ['50', '100', '200', '300', '400', '500', '600', '700', '800', '900', '950'];
const STEPS = 11;

const DEFAULT_BASE = '#3b82f6';

interface PaletteData {
  label: string;
  colors: string[];
}

function contrastOnColor(hex: string): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  // Simple perceived brightness
  const brightness = (r * 299 + g * 587 + b * 114) / 1000;
  return brightness > 128 ? '#09090b' : '#fafafa';
}

export default function PaletteTool() {
  const [baseHex, setBaseHex] = useState(DEFAULT_BASE);
  const [baseInput, setBaseInput] = useState(DEFAULT_BASE);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [palettes, setPalettes] = useState<PaletteData[]>([]);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const computePalettes = useCallback(
    (hex: string, mod: HelmlabModule | null) => {
      const oklabColors = oklabPalette(hex, STEPS);
      const oklabData: PaletteData = { label: 'OKLab', colors: oklabColors };

      if (mod) {
        const hl = new mod.Helmlab();
        const helmlabColors = hl.palette(hex, STEPS);
        setPalettes([
          { label: 'Helmlab', colors: helmlabColors },
          oklabData,
        ]);
      } else {
        setPalettes([
          { label: 'Helmlab', colors: [] },
          oklabData,
        ]);
      }
    },
    []
  );

  useEffect(() => {
    computePalettes(baseHex, hlMod);
  }, [baseHex, hlMod, computePalettes]);

  function handleBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(baseInput)) setBaseHex(baseInput.toLowerCase());
    else setBaseInput(baseHex);
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <h2 className="text-xl font-semibold mb-1">Palette Generator</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Generate a Tailwind-style 50-950 scale from any base color.
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

      {/* Palette strips */}
      <div className="space-y-8">
        {palettes.map((pal) => (
          <div key={pal.label}>
            <div className="flex items-center gap-2 mb-3">
              <span className="text-sm font-medium text-[#fafafa]">{pal.label}</span>
              {pal.label === 'Helmlab' && (
                <span className="text-xs text-[#f97316]">GenSpace</span>
              )}
            </div>
            {pal.colors.length > 0 ? (
              <div className="grid grid-cols-11 gap-3">
                {pal.colors.map((c, i) => (
                  <div key={i} className="flex flex-col items-center gap-1">
                    <div
                      className="w-full aspect-square rounded-lg border border-white/[0.04]"
                      style={{ backgroundColor: c }}
                      title={c}
                    />
                    <span className="text-[10px] text-[#52525b]">{SCALE_LABELS[i]}</span>
                    <span
                      className="text-[9px] font-mono hidden sm:block"
                      style={{ color: contrastOnColor('#0c0c0f') === '#fafafa' ? '#52525b' : '#a1a1aa' }}
                    >
                      {c}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="h-16 rounded-lg bg-white/[0.03] flex items-center justify-center">
                <span className="text-xs text-[#52525b] animate-pulse">Loading helmlab.js...</span>
              </div>
            )}
          </div>
        ))}
      </div>

      <p className="text-xs text-[#52525b] mt-6">
        Both palettes vary lightness while keeping the base hue and chroma. Helmlab uses GenSpace
        (depressed cubic transfer) for smoother cusp behavior at extreme lightness values.
      </p>
    </div>
  );
}
