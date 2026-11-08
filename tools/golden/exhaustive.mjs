// Exhaustive A/B check between two helmlab JS builds: every 8-bit sRGB colour,
// 64^3 Display P3 and Rec.2020 grids, and 3600 cusp hues per gamut.
// Usage: node exhaustive.mjs <A dist/index.js> <B dist/index.js>
// Exit code 1 if any output differs between A and B. The round-trip and cusp
// validity counts describe B alone and do not change the exit code.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [, , pa, pb] = process.argv;
if (!pa || !pb) {
  console.error('Usage: node exhaustive.mjs <A dist/index.js> <B dist/index.js>');
  process.exit(2);
}
const version = (p) => {
  try { return JSON.parse(fs.readFileSync(path.join(path.dirname(path.resolve(p)), '..', 'package.json'), 'utf8')).version; }
  catch { return '?'; }
};
console.log(`A helmlab ${version(pa)}, B helmlab ${version(pb)}, node ${process.version}`);
const A = new (await import(pathToFileURL(pa).href)).Helmlab();
const B = new (await import(pathToFileURL(pb).href)).Helmlab();
const same = (x, y) => x.length === y.length && x.every((v, i) => Object.is(v, y[i]));
const hx = (v) => v.toString(16).padStart(2, '0');
let differences = 0;

// 1) all 16,777,216 sRGB colours: gen and metric forward, hex round-trip
let n = 0, genDiff = 0, metDiff = 0, rtDiff = 0, genRtLossB = 0, metRtLossB = 0;
const t0 = Date.now();
for (let r = 0; r < 256; r++) {
  for (let g = 0; g < 256; g++) {
    for (let b = 0; b < 256; b++) {
      const h = '#' + hx(r) + hx(g) + hx(b);
      const ga = A.gen.fromHex(h), gb = B.gen.fromHex(h);
      const ma = A.metric.fromHex(h), mb = B.metric.fromHex(h);
      if (!same(ga, gb)) genDiff++;
      if (!same(ma, mb)) metDiff++;
      const grB = B.gen.toHex(gb), mrB = B.metric.toHex(mb);
      if (grB !== h) genRtLossB++;
      if (mrB !== h) metRtLossB++;
      if (grB !== A.gen.toHex(ga) || mrB !== A.metric.toHex(ma)) rtDiff++;
      n++;
    }
  }
}
differences += genDiff + metDiff + rtDiff;
console.log(`sRGB, all ${n} colours: gen forward differs ${genDiff}, metric forward differs ${metDiff}, hex round-trip differs ${rtDiff} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
console.log(`  B hex round-trip losses: gen ${genRtLossB}, metric ${metRtLossB}`);

// 2) wide-gamut grids: gen and metric forward, gen toCss
for (const sp of ['display-p3', 'rec2020']) {
  let m = 0, gd = 0, md = 0, cssDiff = 0;
  for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) for (let k = 0; k < 64; k++) {
    const c = `color(${sp} ${i / 63} ${j / 63} ${k / 63})`;
    const ga = A.gen.fromHex(c), gb = B.gen.fromHex(c);
    const ma = A.metric.fromHex(c), mb = B.metric.fromHex(c);
    if (!same(ga, gb)) gd++;
    if (A.gen.toCss(ga, sp) !== B.gen.toCss(gb, sp)) cssDiff++;
    if (!same(ma, mb)) md++;
    m++;
  }
  differences += gd + cssDiff + md;
  console.log(`${sp}, ${m} grid points: gen forward differs ${gd}, gen toCss differs ${cssDiff}, metric forward differs ${md}`);
}

// 3) cusps: 3600 hues per gamut; identity between versions, plus validity of B's cusp
for (const gm of ['srgb', 'display-p3', 'rec2020']) {
  let d = 0, inBad = 0, outBad = 0, notMax = 0;
  for (let k = 0; k < 3600; k++) {
    const hDeg = k / 10;
    const ca = A.gen.cusp(hDeg, gm), cb = B.gen.cusp(hDeg, gm);
    if (!same(ca, cb)) d++;
    const [L, C] = cb, hr = hDeg * Math.PI / 180;
    if (!B.gen.inGamut(B.gen.lab(L, C * 0.999 * Math.cos(hr), C * 0.999 * Math.sin(hr)), gm)) inBad++;
    if (B.gen.inGamut(B.gen.lab(L, C * 1.01 * Math.cos(hr), C * 1.01 * Math.sin(hr)), gm)) outBad++;
    for (const dL of [-0.02, 0.02]) if (B.gen.maxChroma(L + dL, hDeg, gm) > C * 1.0005) notMax++;
  }
  differences += d;
  console.log(`cusp ${gm}, 3600 hues: differs ${d}; B checks: cusp outside gamut ${inBad}, still inside at +1% C ${outBad}, neighbouring L has more chroma ${notMax}`);
}

console.log(differences === 0 ? 'RESULT: no differences' : `RESULT: ${differences} differences`);
process.exitCode = differences === 0 ? 0 : 1;
