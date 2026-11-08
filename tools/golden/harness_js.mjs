// Golden-output harness for helmlab JS.
// Usage: node harness_js.mjs <dist/index.js> <out.json> [--inputs <inputs.json>]
// Calls every public API method over a deterministic input set and records outputs (or errors).
// --inputs also writes the input set; harness_py.py reads it, so both languages get the same inputs.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const [, , distPath, outPath, flag, inputsPath] = process.argv;
if (!distPath || !outPath || (flag !== undefined && (flag !== '--inputs' || !inputsPath))) {
  console.error('Usage: node harness_js.mjs <dist/index.js> <out.json> [--inputs <inputs.json>]');
  process.exit(2);
}
const { Helmlab } = await import(pathToFileURL(distPath).href);
const hl = new Helmlab();
const G = hl.gen, M = hl.metric, T = hl.tokens;

let seed = 12345;
const rnd = () => { seed = (seed * 16807) % 2147483647; return (seed - 1) / 2147483646; };
const hex2 = (v) => v.toString(16).padStart(2, '0');
const hexOf = (r, g, b) => '#' + hex2(r) + hex2(g) + hex2(b);
const fmt = (v) => Number(v.toFixed(6)).toString();
const out = {};
const rec = (key, fn) => {
  try {
    const v = fn();
    out[key] = v === undefined ? null : JSON.parse(JSON.stringify(v, (k, x) =>
      typeof x === 'number' && !Number.isFinite(x) ? `__${x}` : x));
  } catch (e) { out[key] = { __error: String(e && e.message || e).slice(0, 200) }; }
};

// ---------- input sets ----------
const LV = [0, 1, 2, 3, 5, 8, 16, 32, 64, 96, 127, 128, 160, 192, 224, 250, 253, 254, 255];
const hexes = [];
for (const r of LV) for (const g of LV) for (const b of LV) hexes.push(hexOf(r, g, b));
for (let v = 0; v < 256; v++) hexes.push(hexOf(v, v, v));
for (let i = 0; i < 2000; i++) hexes.push(hexOf(...[0, 0, 0].map(() => Math.floor(rnd() * 256))));
const wide = [];
const WL = [0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1];
for (const sp of ['display-p3', 'rec2020']) {
  for (const r of WL) for (const g of WL) for (const b of WL) wide.push(`color(${sp} ${r} ${g} ${b})`);
  for (let i = 0; i < 500; i++) wide.push(`color(${sp} ${fmt(rnd())} ${fmt(rnd())} ${fmt(rnd())})`);
}
const xyzs = [];
for (let i = 0; i < 300; i++) { const Y = Math.pow(10, -6 + 7 * i / 299); xyzs.push([0.95047 * Y, Y, 1.08883 * Y]); }
for (let i = 0; i < 2000; i++) { const Y = rnd() * 10 - 0.2; xyzs.push([Y * (0.3 + 1.4 * rnd()), Y, Y * (0.2 + 1.8 * rnd())]); }
const genLabs = [], metLabs = [];
for (let i = 0; i < 3000; i++) genLabs.push([-0.05 + 1.35 * rnd(), -0.45 + 0.9 * rnd(), -0.45 + 0.9 * rnd()]);
for (let i = 0; i < 3000; i++) metLabs.push([-0.05 + 2.05 * rnd(), -0.5 + rnd(), -0.5 + rnd()]);
const GAMUTS = ['srgb', 'display-p3', 'rec2020'];
const special = ['#0000ff', '#ffffff', '#000000', '#ff0000', '#00ff00', '#ffff00', '#00ffff', '#ff00ff', '#808080', '#3b82f6', '#010101', '#fefefe'];
const pairs = [];
for (let i = 0; i < special.length; i++) for (let j = 0; j < special.length; j++) if (i !== j) pairs.push([special[i], special[j]]);
for (let i = 0; i < 150; i++) pairs.push([hexes[Math.floor(rnd() * hexes.length)], hexes[Math.floor(rnd() * hexes.length)]]);
pairs.push(['#808080', '#808080'], ['#808080', '#818181'], ['#3b82f6', '#3b82f7']);
const wpairs = [];
for (let i = 0; i < 300; i++) wpairs.push([wide[Math.floor(rnd() * wide.length)], wide[Math.floor(rnd() * wide.length)]]);
const bases = special.concat(Array.from({ length: 60 }, () => hexes[Math.floor(rnd() * hexes.length)]));
console.warn = () => {};
if (inputsPath) fs.writeFileSync(inputsPath, JSON.stringify({ hexes, wide, xyzs, genLabs, metLabs, pairs, wpairs, bases, special }));

// ---------- conversions ----------
hexes.forEach((h, i) => {
  rec(`hex.gen.fromHex:${h}`, () => G.fromHex(h));
  rec(`hex.gen.rt:${h}`, () => G.toHex(G.fromHex(h)));
  rec(`hex.gen.toLch:${h}`, () => G.toLch(G.fromHex(h)));
  rec(`hex.metric.fromHex:${h}`, () => M.fromHex(h));
  rec(`hex.metric.rt:${h}`, () => M.toHex(M.fromHex(h)));
  rec(`hex.metric.info:${h}`, () => M.info(h));
  if (i % 3 === 0) for (const f of ['hex', 'rgb', 'hsl', 'oklch', 'p3', 'rec2020']) rec(`hex.tokens.css.${f}:${h}`, () => T.css(h, f));
  if (i % 3 === 0) { rec(`hex.tokens.android:${h}`, () => T.android(h)); rec(`hex.tokens.swift:${h}`, () => T.swift(h));
    if (T.iosP3) rec(`hex.tokens.iosP3:${h}`, () => T.iosP3(h)); }
});
wide.forEach((c) => {
  rec(`wide.gen.fromHex:${c}`, () => G.fromHex(c));
  rec(`wide.metric.fromHex:${c}`, () => M.fromHex(c));
  rec(`wide.metric.info:${c}`, () => M.info(c));
  for (const g of GAMUTS) {
    rec(`wide.gen.toCss.${g}:${c}`, () => G.toCss(G.fromHex(c), g));
    rec(`wide.metric.toCss.${g}:${c}`, () => M.toCss(M.fromHex(c), g));
    rec(`wide.gen.inGamut.${g}:${c}`, () => G.inGamut(c, g));
    rec(`wide.metric.inGamut.${g}:${c}`, () => M.inGamut(c, g));
  }
  rec(`wide.gen.toHex:${c}`, () => G.toHex(G.fromHex(c)));
  rec(`wide.metric.toHex:${c}`, () => M.toHex(M.fromHex(c)));
});
xyzs.forEach((x, i) => {
  rec(`xyz.metric.fromXyz:${i}`, () => M.fromXyz(x));
  rec(`xyz.metric.rt:${i}`, () => M.toXyz(M.fromXyz(x)));
});
genLabs.forEach((l, i) => {
  const L = G.lab(...l);
  rec(`glab.toSrgb:${i}`, () => G.toSrgb(L));
  rec(`glab.toHex:${i}`, () => G.toHex(L));
  rec(`glab.toLch:${i}`, () => G.toLch(L));
  rec(`glab.lchrt:${i}`, () => G.fromLch(G.toLch(L)));
  for (const g of GAMUTS) {
    rec(`glab.toCss.${g}:${i}`, () => G.toCss(L, g));
    rec(`glab.inGamut.${g}:${i}`, () => G.inGamut(L, g));
    if (i % 2 === 0) for (const m of ['chroma', 'adaptive']) rec(`glab.gamutMap.${g}.${m}:${i}`, () => G.gamutMap(L, g, m));
  }
});
metLabs.forEach((l, i) => {
  const L = M.lab(...l);
  rec(`mlab.toSrgb:${i}`, () => M.toSrgb(L));
  rec(`mlab.toHex:${i}`, () => M.toHex(L));
  rec(`mlab.toXyz:${i}`, () => M.toXyz(L));
  rec(`mlab.toLch:${i}`, () => M.toLch(L));
  rec(`mlab.lchrt:${i}`, () => M.fromLch(M.toLch(L)));
  for (const g of GAMUTS) {
    rec(`mlab.toCss.${g}:${i}`, () => M.toCss(L, g));
    rec(`mlab.inGamut.${g}:${i}`, () => M.inGamut(L, g));
  }
});

// ---------- cusp geometry ----------
for (const g of GAMUTS) {
  for (let k = 0; k < 720; k++) rec(`cusp.${g}:${k / 2}`, () => G.cusp(k / 2, g));
  for (let Li = 1; Li <= 19; Li++) for (let h = 0; h < 360; h += 5) rec(`maxChroma.${g}:${Li}:${h}`, () => G.maxChroma(Li * 0.05, h, g));
}
bases.forEach((b) => { for (const g of GAMUTS) rec(`vivid.${g}:${b}`, () => G.vivid(b, { gamut: g })); });

// ---------- generation ----------
pairs.forEach(([a, b], i) => {
  for (const g of GAMUTS) for (const n of [0, 1, 2, 5, 16]) {
    if (g !== 'srgb' && i % 3 !== 0) continue;
    rec(`gradient.${g}.${n}:${a}>${b}`, () => G.gradient(a, b, n, { gamut: g }));
  }
  for (const t of [0, 0.25, 0.5, 0.9, 1, -0.5, 1.5]) rec(`mix.${t}:${a}>${b}`, () => G.mix(a, b, t));
});
wpairs.slice(0, 60).forEach(([a, b]) => { for (const g of GAMUTS) rec(`wgradient.${g}:${a}>${b}`, () => G.gradient(a, b, 9, { gamut: g })); });
bases.forEach((b) => {
  for (const g of GAMUTS) {
    rec(`palette.${g}:${b}`, () => G.palette(b, 10, { gamut: g }));
    rec(`scale.${g}:${b}`, () => G.scale(b, undefined, { gamut: g }));
  }
  rec(`scale.custom:${b}`, () => G.scale(b, [100, 500, 900]));
  for (const k of ['complementary', 'analogous', 'triadic', 'tetradic', 'split_complementary']) rec(`harm.${k}:${b}`, () => G.harmonies(b, k));
  for (const d of [30, 90, 180, -45, 360, 720.5]) rec(`rot.${d}:${b}`, () => G.rotateHue(b, d));
});
for (const g of GAMUTS) for (const n of [1, 6, 12]) for (const [Lx, Cx] of [[0.6, 0.15], [0.3, 0.05], [0.9, 0.3]])
  rec(`hueRing.${g}.${n}.${Lx}.${Cx}`, () => G.hueRing(n, { lightness: Lx, chroma: Cx, gamut: g }));

// ---------- contrast & modes ----------
pairs.forEach(([a, b]) => {
  rec(`cr:${a}>${b}`, () => G.contrastRatio(a, b));
  rec(`meets.AA:${a}>${b}`, () => G.meetsContrast(a, b, 'AA'));
  rec(`meets.AAA:${a}>${b}`, () => G.meetsContrast(a, b, 'AAA'));
  for (const r of [3, 4.5, 7]) rec(`ensure.${r}:${a}>${b}`, () => G.ensureContrast(a, b, r));
  rec(`ensure.strict:${a}>${b}`, () => G.ensureContrast(a, b, 7, { strict: true }));
  rec(`mode.l2d:${a}`, () => G.adaptToMode(a, 'light', 'dark'));
  rec(`mode.d2l:${a}`, () => G.adaptToMode(a, 'dark', 'light'));
  rec(`pair:${a}>${b}`, () => G.adaptPair(a, b, 'light', 'dark'));
});

// ---------- measurement ----------
const mpairs = pairs.concat(wpairs);
mpairs.forEach(([a, b]) => {
  rec(`diff:${a}>${b}`, () => M.difference(a, b));
  rec(`eucl:${a}>${b}`, () => M.euclidean(a, b));
  rec(`de00:${a}>${b}`, () => M.ciede2000(a, b));
  rec(`jnd:${a}>${b}`, () => M.jnd(a, b));
  rec(`dist:${a}>${b}`, () => M.distance(M.fromHex(a), M.fromHex(b)));
  rec(`conf:${a}>${b}`, () => M.confidence(a, b));
});
const pal = special.slice(0, 10);
bases.concat(wide.slice(0, 40)).forEach((t) => {
  for (const m of ['ciede2000', 'difference', 'euclidean']) rec(`nearest.${m}:${t}`, () => M.nearest(t, pal, m));
});

// ---------- tokens on scales ----------
bases.slice(0, 30).forEach((b, i) => {
  const sc = G.scale(b);
  rec(`tok.vars:${b}`, () => T.cssVariables(sc));
  rec(`tok.tw:${b}`, () => T.tailwind(sc, 'c' + i));
  rec(`tok.multi:${b}`, () => T.multiFormat(sc, 'c' + i));
  rec(`tok.json:${b}`, () => T.json({ ['c' + i]: sc }));
});

const nKeys = Object.keys(out).length;
const nErr = Object.values(out).filter((v) => v && typeof v === 'object' && '__error' in v).length;
let version = null;
try {
  version = JSON.parse(fs.readFileSync(path.join(path.dirname(path.resolve(distPath)), '..', 'package.json'), 'utf8')).version;
} catch { /* not an npm package layout */ }
// ECMA-262 leaves the accuracy of Math.cbrt, Math.exp and Math.pow to the engine, so record it.
out.__meta = { harness: 'js', helmlab: version, node: process.version, v8: process.versions.v8, platform: process.platform, arch: process.arch };
fs.writeFileSync(outPath, JSON.stringify(out));
console.log(`keys=${nKeys} errors=${nErr} helmlab=${version} node=${process.version}`);
