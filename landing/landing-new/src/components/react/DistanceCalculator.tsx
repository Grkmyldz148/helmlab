import { useState, useEffect, useMemo } from 'react';
import {
  hexToRgb,
  srgbToOklab,
  srgbToCieLab,
  relativeLuminance,
} from '../../lib/color-utils';

type HelmlabModule = typeof import('helmlab');

const DEFAULT_C1 = '#3b82f6';
const DEFAULT_C2 = '#4c8af7';

// Scale factor to bring Helmlab Lab (L in [0,1.5]) onto CIE-Lab-like scale
// (L in [0,100]) so classical JND thresholds (1, 2.3, 5, 10) apply.
const HELMLAB_SCALE = 100;

interface Preset {
  label: string;
  c1: string;
  c2: string;
}

const PRESETS: Preset[] = [
  { label: 'Identical',         c1: '#3b82f6', c2: '#3b82f6' },
  { label: 'Just noticeable',   c1: '#3b82f6', c2: '#4c8af7' },
  { label: 'Clearly different', c1: '#3b82f6', c2: '#5b9af8' },
  { label: 'Very different',    c1: '#3b82f6', c2: '#ef4444' },
];

interface JndBand {
  label: string;
  desc: string;
  border: string;
  text: string;
  bg: string;
}

function classifyJnd(dE: number): JndBand {
  if (dE < 1.0) {
    return {
      label: 'Imperceptible',
      desc: 'Below the just-noticeable-difference threshold.',
      border: 'border-[#34d399]/30',
      text: 'text-[#34d399]',
      bg: 'bg-[#34d399]/5',
    };
  }
  if (dE < 2.3) {
    return {
      label: 'Just noticeable',
      desc: 'At the edge of human perception — trained eyes may spot it.',
      border: 'border-[#a3e635]/30',
      text: 'text-[#a3e635]',
      bg: 'bg-[#a3e635]/5',
    };
  }
  if (dE < 5) {
    return {
      label: 'Clearly different',
      desc: 'Easily distinguishable, but the colors still feel related.',
      border: 'border-[#fbbf24]/30',
      text: 'text-[#fbbf24]',
      bg: 'bg-[#fbbf24]/5',
    };
  }
  if (dE < 10) {
    return {
      label: 'Strongly different',
      desc: 'A substantial perceptual gap — different hues or lightnesses.',
      border: 'border-[#fb923c]/30',
      text: 'text-[#fb923c]',
      bg: 'bg-[#fb923c]/5',
    };
  }
  return {
    label: 'Very different',
    desc: 'A dramatic, unmistakable color difference.',
    border: 'border-[#f87171]/30',
    text: 'text-[#f87171]',
    bg: 'bg-[#f87171]/5',
  };
}

// ── CIEDE2000 (standard formula) ─────────────────────────────────
function ciede2000(
  L1: number, a1: number, b1: number,
  L2: number, a2: number, b2: number,
): number {
  const { sqrt, atan2, cos, sin, exp, PI, pow, abs } = Math;
  const C1 = sqrt(a1 * a1 + b1 * b1);
  const C2 = sqrt(a2 * a2 + b2 * b2);
  const Cab = (C1 + C2) / 2;
  const Cab7 = pow(Cab, 7);
  const p257 = pow(25, 7);
  const G = 0.5 * (1 - sqrt(Cab7 / (Cab7 + p257)));
  const ap1 = (1 + G) * a1;
  const ap2 = (1 + G) * a2;
  const Cp1 = sqrt(ap1 * ap1 + b1 * b1);
  const Cp2 = sqrt(ap2 * ap2 + b2 * b2);

  let hp1 = atan2(b1, ap1);
  let hp2 = atan2(b2, ap2);
  if (hp1 < 0) hp1 += 2 * PI;
  if (hp2 < 0) hp2 += 2 * PI;

  const dLp = L2 - L1;
  const dCp = Cp2 - Cp1;
  let dhp = 0;
  if (Cp1 * Cp2 !== 0) {
    dhp = hp2 - hp1;
    if (dhp > PI) dhp -= 2 * PI;
    if (dhp < -PI) dhp += 2 * PI;
  }
  const dHp = 2 * sqrt(Cp1 * Cp2) * sin(dhp / 2);

  const Lp = (L1 + L2) / 2;
  const Cp = (Cp1 + Cp2) / 2;

  let hp = 0;
  if (Cp1 * Cp2 === 0) {
    hp = hp1 + hp2;
  } else {
    hp = (hp1 + hp2) / 2;
    if (abs(hp1 - hp2) > PI) hp += hp < PI ? PI : -PI;
  }

  const T =
    1 -
    0.17 * cos(hp - PI / 6) +
    0.24 * cos(2 * hp) +
    0.32 * cos(3 * hp + PI / 30) -
    0.20 * cos(4 * hp - (63 * PI) / 180);

  const Lp50 = Lp - 50;
  const SL = 1 + (0.015 * Lp50 * Lp50) / sqrt(20 + Lp50 * Lp50);
  const SC = 1 + 0.045 * Cp;
  const SH = 1 + 0.015 * Cp * T;

  const Cp7 = pow(Cp, 7);
  const RC = 2 * sqrt(Cp7 / (Cp7 + p257));
  const hpDeg = (hp * 180) / PI;
  const dth = 30 * exp(-pow((hpDeg - 275) / 25, 2));
  const RT = -sin((2 * dth * PI) / 180) * RC;

  return sqrt(
    pow(dLp / SL, 2) +
      pow(dCp / SC, 2) +
      pow(dHp / SH, 2) +
      RT * (dCp / SC) * (dHp / SH),
  );
}

function euclid3(
  a: [number, number, number],
  b: [number, number, number],
): number {
  const dL = a[0] - b[0];
  const da = a[1] - b[1];
  const db = a[2] - b[2];
  return Math.sqrt(dL * dL + da * da + db * db);
}

// OKLab ΔE is naturally tiny (L in [0,1]) — scale so JND ≈ 1 is comparable
// to other ΔE scales. Factor 100 is the community-accepted convention.
const OKLAB_SCALE = 100;

function isValidHex(s: string): boolean {
  return /^#[0-9a-fA-F]{6}$/.test(s);
}

function contrastTextColor(hex: string): string {
  if (!isValidHex(hex)) return 'rgba(255,255,255,0.9)';
  const [r, g, b] = hexToRgb(hex);
  return relativeLuminance(r, g, b) > 0.2
    ? 'rgba(0,0,0,0.78)'
    : 'rgba(255,255,255,0.92)';
}

interface DeltaRow {
  key: string;
  label: string;
  value: number;
  primary: boolean;
}

export default function DistanceCalculator() {
  const [hex1, setHex1] = useState(DEFAULT_C1);
  const [hex2, setHex2] = useState(DEFAULT_C2);
  const [input1, setInput1] = useState(DEFAULT_C1);
  const [input2, setInput2] = useState(DEFAULT_C2);
  const [hlMod, setHlMod] = useState<HelmlabModule | null>(null);

  useEffect(() => {
    import('helmlab').then(setHlMod).catch(() => {});
  }, []);

  const deltas = useMemo<DeltaRow[]>(() => {
    if (!isValidHex(hex1) || !isValidHex(hex2)) return [];

    const [r1, g1, b1] = hexToRgb(hex1);
    const [r2, g2, b2] = hexToRgb(hex2);

    const ok1 = srgbToOklab(r1, g1, b1);
    const ok2 = srgbToOklab(r2, g2, b2);
    const okDE = euclid3(ok1, ok2) * OKLAB_SCALE;

    const cie1 = srgbToCieLab(r1, g1, b1);
    const cie2 = srgbToCieLab(r2, g2, b2);
    const cieDE = euclid3(cie1, cie2);
    const cie2k = ciede2000(cie1[0], cie1[1], cie1[2], cie2[0], cie2[1], cie2[2]);

    const rows: DeltaRow[] = [];

    if (hlMod) {
      const hl = new hlMod.Helmlab();
      const helmDE = hl.deltaE(hex1, hex2) * HELMLAB_SCALE;
      rows.push({
        key: 'helmlab',
        label: 'Helmlab ΔE',
        value: helmDE,
        primary: true,
      });
    } else {
      rows.push({ key: 'helmlab', label: 'Helmlab ΔE', value: NaN, primary: true });
    }

    rows.push(
      { key: 'oklab',   label: 'OKLab ΔE',      value: okDE,  primary: false },
      { key: 'cie76',   label: 'CIE Lab ΔE76',  value: cieDE, primary: false },
      { key: 'cie2000', label: 'CIEDE2000',     value: cie2k, primary: false },
    );

    return rows;
  }, [hex1, hex2, hlMod]);

  const helmDE = deltas.find((d) => d.key === 'helmlab')?.value ?? NaN;
  const jnd = !Number.isNaN(helmDE) ? classifyJnd(helmDE) : null;

  function handleBlur(which: 1 | 2) {
    if (which === 1) {
      if (isValidHex(input1)) setHex1(input1.toLowerCase());
      else setInput1(hex1);
    } else {
      if (isValidHex(input2)) setHex2(input2.toLowerCase());
      else setInput2(hex2);
    }
  }

  function applyPreset(p: Preset) {
    setHex1(p.c1);
    setHex2(p.c2);
    setInput1(p.c1);
    setInput2(p.c2);
  }

  return (
    <div className="rounded-2xl border border-white/[0.06] bg-[#0c0c0f] p-6 sm:p-8">
      <div className="flex items-start justify-between gap-4 mb-1">
        <h2 className="text-xl font-semibold">Perceptual Distance</h2>
        <span className="text-[10px] px-2 py-0.5 rounded-full border border-[#f97316]/30 text-[#f97316] bg-[#f97316]/5 uppercase tracking-wider font-medium shrink-0">
          MetricSpace
        </span>
      </div>
      <p className="text-sm text-[#a1a1aa] mb-6">
        Measure how different two colors look. Helmlab ΔE is trained on{' '}
        <span className="text-[#fafafa]">COMBVD + MacAdam + He</span> — the
        other metrics are shown for comparison.
      </p>

      {/* Color swatches */}
      <div className="grid grid-cols-1 sm:grid-cols-[1fr_auto_1fr] gap-4 items-stretch mb-6">
        {/* Color 1 */}
        <div className="flex flex-col gap-3">
          <div
            className="rounded-xl border border-white/[0.06] h-28 sm:h-36 flex items-center justify-center font-mono text-sm font-semibold tracking-wider"
            style={{
              backgroundColor: isValidHex(hex1) ? hex1 : '#000',
              color: contrastTextColor(hex1),
            }}
          >
            {hex1.toUpperCase()}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={isValidHex(hex1) ? hex1 : '#000000'}
              onChange={(e) => {
                setHex1(e.target.value);
                setInput1(e.target.value);
              }}
              className="w-10 h-10 rounded-lg border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
            />
            <input
              type="text"
              value={input1}
              onChange={(e) => setInput1(e.target.value)}
              onBlur={() => handleBlur(1)}
              onKeyDown={(e) => e.key === 'Enter' && handleBlur(1)}
              className="flex-1 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
              spellCheck={false}
            />
          </div>
        </div>

        {/* VS */}
        <div className="hidden sm:flex items-center justify-center">
          <div className="w-10 h-10 rounded-full border border-white/[0.06] bg-white/[0.02] flex items-center justify-center text-xs font-medium text-[#52525b] uppercase tracking-wider">
            vs
          </div>
        </div>

        {/* Color 2 */}
        <div className="flex flex-col gap-3">
          <div
            className="rounded-xl border border-white/[0.06] h-28 sm:h-36 flex items-center justify-center font-mono text-sm font-semibold tracking-wider"
            style={{
              backgroundColor: isValidHex(hex2) ? hex2 : '#000',
              color: contrastTextColor(hex2),
            }}
          >
            {hex2.toUpperCase()}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="color"
              value={isValidHex(hex2) ? hex2 : '#000000'}
              onChange={(e) => {
                setHex2(e.target.value);
                setInput2(e.target.value);
              }}
              className="w-10 h-10 rounded-lg border border-white/[0.06] bg-transparent cursor-pointer shrink-0"
            />
            <input
              type="text"
              value={input2}
              onChange={(e) => setInput2(e.target.value)}
              onBlur={() => handleBlur(2)}
              onKeyDown={(e) => e.key === 'Enter' && handleBlur(2)}
              className="flex-1 rounded-lg border border-white/[0.06] bg-white/[0.03] px-3 py-2 text-sm font-mono text-[#fafafa] focus:outline-none focus:border-[#f97316]/40"
              spellCheck={false}
            />
          </div>
        </div>
      </div>

      {/* ΔE values row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {deltas.map((d) => {
          const loading = Number.isNaN(d.value);
          const display = loading ? '…' : d.value.toFixed(2);
          return (
            <div
              key={d.key}
              className={`rounded-xl border p-4 ${
                d.primary
                  ? 'border-[#f97316]/30 bg-[#f97316]/5'
                  : 'border-white/[0.06] bg-white/[0.02]'
              }`}
            >
              <div
                className={`text-[10px] uppercase tracking-wider font-medium mb-2 ${
                  d.primary ? 'text-[#f97316]' : 'text-[#a1a1aa]'
                }`}
              >
                {d.label}
              </div>
              <div
                className={`text-2xl font-mono font-semibold ${
                  d.primary ? 'text-[#fafafa]' : 'text-[#d4d4d8]'
                }`}
              >
                {display}
              </div>
            </div>
          );
        })}
      </div>

      {/* JND interpretation */}
      {jnd && (
        <div
          className={`rounded-xl border p-4 mb-6 ${jnd.border} ${jnd.bg}`}
        >
          <div className="flex items-center gap-3">
            <span
              className={`text-xs uppercase tracking-wider font-semibold ${jnd.text}`}
            >
              {jnd.label}
            </span>
            <span className="text-xs text-[#52525b] font-mono">
              Helmlab ΔE = {helmDE.toFixed(2)}
            </span>
          </div>
          <p className="text-sm text-[#a1a1aa] mt-1.5">{jnd.desc}</p>
        </div>
      )}

      {/* Presets */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-[#52525b] font-medium mb-2">
          Try a preset
        </div>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p) => {
            const active = hex1 === p.c1 && hex2 === p.c2;
            return (
              <button
                key={p.label}
                onClick={() => applyPreset(p)}
                className={`text-xs px-3 py-2 rounded-lg border transition-colors ${
                  active
                    ? 'border-[#f97316]/40 bg-[#f97316]/10 text-[#f97316]'
                    : 'border-white/[0.06] bg-white/[0.02] text-[#a1a1aa] hover:border-white/[0.12] hover:text-[#fafafa]'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-sm border border-white/[0.08]"
                    style={{ backgroundColor: p.c1 }}
                  />
                  <span
                    className="w-3 h-3 rounded-sm border border-white/[0.08]"
                    style={{ backgroundColor: p.c2 }}
                  />
                  {p.label}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <p className="text-xs text-[#52525b] mt-6 leading-relaxed">
        Helmlab ΔE and OKLab ΔE are Euclidean distances in their respective
        Lab spaces (scaled by 100 to match the CIE Lab range). CIEDE2000 is
        the industry reference — Helmlab achieves lower STRESS than CIEDE2000
        on COMBVD, MacAdam, and He datasets.
      </p>
    </div>
  );
}
