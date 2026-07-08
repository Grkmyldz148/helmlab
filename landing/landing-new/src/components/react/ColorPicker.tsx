import { useState, useEffect, useCallback } from 'react';
import {
  hexToRgb,
  srgbToOklab,
  srgbToCieLab,
  srgbToXyz,
  isInSrgb,
} from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const DEFAULT_HEX = '#3b82f6';

interface SpaceCoords {
  label: string;
  L: string;
  a: string;
  b: string;
}

interface GamutInfo {
  srgb: boolean;
  p3: boolean;
}

export default function ColorPicker() {
  const [hex, setHex] = useState(DEFAULT_HEX);
  const [hexInput, setHexInput] = useState(DEFAULT_HEX);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);
  const [coords, setCoords] = useState<SpaceCoords[]>([]);
  const [gamut, setGamut] = useState<GamutInfo>({ srgb: true, p3: true });
  const [roundTrip, setRoundTrip] = useState<{ hex: string; error: string } | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const analyze = useCallback(
    (h: string, mod: HelmlabModule | null) => {
      const [r, g, b] = hexToRgb(h);

      // OKLab
      const [oL, oa, ob] = srgbToOklab(r, g, b);
      // CIE Lab
      const [cL, ca, cb] = srgbToCieLab(r, g, b);

      const spaces: SpaceCoords[] = [];

      if (mod) {
        const hl = new mod.Helmlab();
        // GenSpace coordinates
        const genLab = hl.gen.fromHex(h);
        spaces.push({
          label: 'Helmlab GenSpace',
          L: genLab[0].toFixed(4),
          a: genLab[1].toFixed(4),
          b: genLab[2].toFixed(4),
        });

        // Round-trip test
        const backHex = hl.gen.toHex(genLab);
        const [r2, g2, b2] = hexToRgb(backHex);
        const dR = Math.abs(r - r2);
        const dG = Math.abs(g - g2);
        const dB = Math.abs(b - b2);
        const maxErr = Math.max(dR, dG, dB);
        setRoundTrip({
          hex: backHex,
          error: maxErr < 0.0001 ? '<0.0001' : maxErr.toFixed(6),
        });

        // Gamut info via helmlab
        const metricLab = hl.metric.fromHex(h);
        setGamut({
          srgb: hl.metric.inGamut(metricLab),
          p3: hl.metric.inGamut(metricLab, 'display-p3'),
        });
      } else {
        spaces.push({
          label: 'Helmlab GenSpace',
          L: '...',
          a: '...',
          b: '...',
        });
        setRoundTrip(null);
        setGamut({ srgb: isInSrgb(r, g, b), p3: true });
      }

      spaces.push(
        {
          label: 'OKLab',
          L: oL.toFixed(4),
          a: oa.toFixed(4),
          b: ob.toFixed(4),
        },
        {
          label: 'CIE Lab',
          L: cL.toFixed(2),
          a: ca.toFixed(2),
          b: cb.toFixed(2),
        }
      );

      // Also include XYZ
      const [x, y, z] = srgbToXyz(r, g, b);
      spaces.push({
        label: 'CIE XYZ',
        L: x.toFixed(5),
        a: y.toFixed(5),
        b: z.toFixed(5),
      });

      setCoords(spaces);
    },
    []
  );

  useEffect(() => {
    analyze(hex, hlMod);
  }, [hex, hlMod, analyze]);

  function handleBlur() {
    if (/^#[0-9a-fA-F]{6}$/.test(hexInput)) setHex(hexInput.toLowerCase());
    else setHexInput(hex);
  }

  const xyzLabels = ['X', 'Y', 'Z'];

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <h2 className="text-xl font-semibold mb-1">Color Inspector</h2>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Pick a color and see its coordinates in every space.
      </p>

      {/* Color input */}
      <div className="flex items-center gap-3 mb-8">
        <input
          type="color"
          value={hex}
          onChange={(e) => {
            setHex(e.target.value);
            setHexInput(e.target.value);
          }}
          className="w-14 h-14 rounded-xl border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
        />
        <div>
          <input
            type="text"
            value={hexInput}
            onChange={(e) => setHexInput(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={(e) => e.key === 'Enter' && handleBlur()}
            className="w-28 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
            spellCheck={false}
          />
          {/* Gamut badges */}
          <div className="flex gap-2 mt-2">
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full border ${
                gamut.srgb
                  ? 'border-[#34d399]/30 text-[#34d399] bg-[#34d399]/5'
                  : 'border-[#f87171]/30 text-[#f87171] bg-[#f87171]/5'
              }`}
            >
              {gamut.srgb ? 'In sRGB' : 'Out of sRGB'}
            </span>
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full border ${
                gamut.p3
                  ? 'border-[#34d399]/30 text-[#34d399] bg-[#34d399]/5'
                  : 'border-[#a1a1aa]/30 text-[#a1a1aa] bg-white/[0.02]'
              }`}
            >
              {gamut.p3 ? 'In P3' : 'Out of P3'}
            </span>
          </div>
        </div>

        {/* Color swatch preview */}
        <div
          className="w-14 h-14 rounded-xl border border-white/[0.06] shrink-0 ml-auto hidden sm:block"
          style={{ backgroundColor: hex }}
        />
      </div>

      {/* Coordinate grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-6">
        {coords.map((s) => {
          const isXYZ = s.label === 'CIE XYZ';
          const labels = isXYZ ? xyzLabels : ['L', 'a', 'b'];
          return (
            <div
              key={s.label}
              className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4"
            >
              <span className="text-xs font-medium text-[#a1a1aa] uppercase tracking-wider">
                {s.label}
              </span>
              <div className="grid grid-cols-3 gap-2 mt-3">
                {[s.L, s.a, s.b].map((val, i) => (
                  <div key={i} className="text-center">
                    <span className="text-[10px] text-[#52525b] block mb-0.5">
                      {labels[i]}
                    </span>
                    <span className="text-sm font-mono text-[#fafafa]">{val}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* Round-trip test */}
      {roundTrip && (
        <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-4">
          <span className="text-xs font-medium text-[#a1a1aa] uppercase tracking-wider">
            GenSpace Round-Trip
          </span>
          <div className="flex items-center gap-4 mt-3">
            <div className="flex items-center gap-2">
              <div
                className="w-8 h-8 rounded-md border border-white/[0.06]"
                style={{ backgroundColor: hex }}
              />
              <span className="text-xs text-[#52525b]">&rarr;</span>
              <span className="text-xs font-mono text-[#a1a1aa]">GenSpace</span>
              <span className="text-xs text-[#52525b]">&rarr;</span>
              <div
                className="w-8 h-8 rounded-md border border-white/[0.06]"
                style={{ backgroundColor: roundTrip.hex }}
              />
            </div>
            <div className="ml-auto text-right">
              <span className="text-xs text-[#52525b] block">Max channel error</span>
              <span className="text-sm font-mono text-[#fafafa]">{roundTrip.error}</span>
            </div>
          </div>
          <div className="flex items-center gap-2 mt-2">
            <span className="text-xs font-mono text-[#52525b]">{hex}</span>
            <span className="text-xs text-[#52525b]">&rarr;</span>
            <span className="text-xs font-mono text-[#52525b]">{roundTrip.hex}</span>
          </div>
        </div>
      )}

      <p className="text-xs text-[#52525b] mt-6">
        Any sRGB hex color is in gamut by definition. The P3 check applies to MetricSpace Lab coordinates
        and tests whether the inverse maps into the wider Display P3 gamut.
      </p>
    </div>
  );
}
