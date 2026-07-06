// ColorQA — auto-verifiable color-engineering tasks for measuring the
// color-space-routing skill (skill-on vs skill-off pass rate).
// Usage: node colorqa.mjs list            → print task prompts (give to the model)
//        node colorqa.mjs verify ans.json → grade {taskId: answer} JSON
// Code answers are JS expression bodies evaluated with `hl` (Helmlab) in scope.
import { Helmlab } from 'helmlab';
const hl = new Helmlab();
const hex2rgb = h => [1,3,5].map(i => parseInt(h.slice(i,i+2),16)/255);
const srgb2lin = c => c <= 0.04045 ? c/12.92 : ((c+0.055)/1.055)**2.4;
const okhue = h => { const [L,a,b] = hl.genToLch(hl.genFromHex(h)); return b; }; // placeholder unused

export const TASKS = [
  { id:'T1-blue-gradient', type:'code',
    prompt:'Write a JS expression producing a 16-step gradient (array of hex) from #0000ff to #ffffff that stays perceptually blue through the midpoint (no purple detour). `hl` (helmlab) is available.',
    verify: (out) => { if (!Array.isArray(out) || out.length!==16) return false;
      const mid = hl.genToLch(hl.genFromHex(out[8])); // hue° of midpoint
      return mid[2] > 230 && mid[2] < 290; } },
  { id:'T2-delta-e', type:'code',
    prompt:'Write a JS expression: a function f(hex1,hex2) returning a perceptual color difference suitable for "did the brand color change?" checks. `hl` is available.',
    verify: (f) => { const pairs=[['#808080','#828282'],['#ff0000','#ff2000'],['#ff0000','#00ff00'],['#0000ff','#0080ff']];
      const ref=[0.5,1.5,3.5,2.5]; // rank order only
      const d = pairs.map(p=>f(p[0],p[1]));
      const rank = a=>a.map((v,i)=>[v,i]).sort((x,y)=>x[0]-y[0]).map(x=>x[1]).join();
      return rank(d)===rank(ref); } },
  { id:'T3-gray-axis', type:'code',
    prompt:'Write a JS function f(hex) converting a color to a perceptual Lab for palette generation. Grays must map to chroma ≈ 0. `hl` is available.',
    verify: (f) => ['#333333','#808080','#cccccc'].every(g=>{ const lab=f(g);
      return Math.hypot(lab[1],lab[2]) < 1e-3; }) },
  { id:'T4-physical-mix', type:'code',
    prompt:'Write a JS function f(hexA,hexB) returning the hex of a 50/50 PHYSICAL light mix of two colors (as if two lights overlap).',
    verify: (f) => { const out = f('#000000','#ffffff').toLowerCase();
      const v = parseInt(out.slice(1,3),16)/255; // linear 0.5 → sRGB ≈ 0.7354
      return Math.abs(srgb2lin(v) - 0.5) < 0.02; } },
  { id:'T5-hue-wrap', type:'code',
    prompt:'Write a JS function f(h1,h2,t) interpolating two hue angles in degrees along the shorter arc (result in [0,360)).',
    verify: (f) => Math.abs(f(350,10,0.5))<1e-6 || Math.abs(f(350,10,0.5)-360)<1e-6 },
  { id:'T6-lch-distance', type:'code',
    prompt:'Two colors are given in LCh form: [0.5,0.1,10] and [0.5,0.1,350]. Write a JS expression computing a CORRECT perceptual-space distance between them.',
    verify: (d) => { if (typeof d==='function') d = d([0.5,0.1,10],[0.5,0.1,350]);
      return typeof d==='number' && d>0.01 && d<0.08; } }, // rectangular-form chord ≈ 2*C*sin(10°)≈0.035; naive LCh gives 340
  { id:'T7-cvd', type:'choice',
    prompt:'You must generate a categorical palette safe for deuteranopia. Which color space do you generate candidates in? One word.',
    verify: (s) => /oklab/i.test(String(s)) },
  { id:'T8-picker', type:'choice',
    prompt:'You are building a color picker UI needing bounded, HSL-shaped but perceptually sane coordinates. Which space? One word.',
    verify: (s) => /okhs[lv]/i.test(String(s)) },
  { id:'T9-hdr', type:'choice',
    prompt:'You must compute perceptual differences on HDR/PQ content peaking at 4000 cd/m². Which space family? One word.',
    verify: (s) => /jzazbz|ictcp/i.test(String(s)) },
  { id:'T10-noticeable', type:'choice',
    prompt:'A regression test reports ΔE00 = 1.0 between two low-chroma UI grays. Do you fail the build? Answer yes/no with one-line reason.',
    verify: (s) => /no/i.test(String(s).slice(0,20)) },
  { id:'T11-css-nearby', type:'choice',
    prompt:'CSS-only gradient between two vivid nearby hues (#0088ff to #00ff88) that must stay vivid throughout. Which interpolation color space do you write after "in"? One token.',
    verify: (s) => /oklch/i.test(String(s)) && !/oklab\b/i.test(String(s).replace(/oklch/ig,'')) },
  { id:'T12-css-distant', type:'choice',
    prompt:'CSS-only gradient between two DISTANT vivid hues (#ff0000 to #0000ff) avoiding weird intermediate hues. Which interpolation space after "in"? One token.',
    verify: (s) => /oklab/i.test(String(s)) && !/oklch/i.test(String(s)) },
  { id:'T13-wcag', type:'code',
    prompt:'Write a JS function f(hexFg,hexBg) returning the WCAG 2.1 contrast ratio.',
    verify: (f) => Math.abs(f('#ffffff','#000000')-21)<0.1 && Math.abs(f('#ffffff','#3b82f6')-3.68)<0.08 },
  { id:'T14-darken', type:'code',
    prompt:'Write a JS function f(hex) returning the hex of the color perceptually darkened (noticeably lower lightness) WITHOUT changing its hue. `hl` (helmlab) is available.',
    verify: (f) => { const out=f('#3b82f6'); const a=hl.genToLch(hl.genFromHex('#3b82f6')), b=hl.genToLch(hl.genFromHex(out));
      let dh=Math.abs(a[2]-b[2]); dh=Math.min(dh,360-dh);
      return b[0] < a[0]-0.05 && dh < 10; } },
  { id:'T15-lightness-sort', type:'code',
    prompt:'Write a JS function f(hexArray) sorting colors by PERCEIVED lightness, darkest first. `hl` is available.',
    verify: (f) => { const out=f(['#ffff00','#0000ff','#808080']);
      return out[0]==='#0000ff' && out[2]==='#ffff00'; } },
  { id:'T16-circular-mean', type:'code',
    prompt:'Write a JS expression or function computing the circular mean of the hue angles 350 and 10 (degrees).',
    verify: (d) => { if (typeof d==='function') d=d(350,10); d=((d%360)+360)%360;
      return d<1e-6 || Math.abs(d-360)<1e-6; } },
  { id:'T17-equal-gray', type:'code',
    prompt:'Write a JS expression or function returning the hex of the NEUTRAL GRAY that has the same perceived lightness as #3b82f6. `hl` is available.',
    verify: (g) => { if (typeof g==='function') g=g('#3b82f6'); const lch=hl.genToLch(hl.genFromHex(g));
      return Math.abs(lch[0]-0.5586)<0.04 && lch[1]<0.03; } },
  { id:'T18-angular-dist', type:'code',
    prompt:'Write a JS expression or function computing the shortest angular distance in degrees between hue 20 and hue 340.',
    verify: (d) => { if (typeof d==='function') d=d(20,340); return Math.abs(d-40)<1e-6; } },
  { id:'T19-cat', type:'choice',
    prompt:'You must compare colors photographed under illuminant A against references measured under D65. What must you apply before computing any ΔE? One or two words.',
    verify: (s) => /adapt|bradford|cat\b|chromatic/i.test(String(s)) },
  { id:'T20-hsv-contrast', type:'choice',
    prompt:'A teammate suggests using HSV Value (V) to check text contrast for accessibility. Is that acceptable? Answer yes/no with one-line reason.',
    verify: (s) => /^no/i.test(String(s).trim()) },
];

let GTASKS = TASKS;
if (process.env.TASKS) { GTASKS = (await import(process.env.TASKS)).TASKS; }
const mode = process.argv[2];
if (mode === 'list') for (const t of GTASKS) console.log(`\n[${t.id}] (${t.type})\n${t.prompt}`);
if (mode === 'verify') {
  const answers = JSON.parse(await import('node:fs').then(fs=>fs.readFileSync(process.argv[3],'utf8')));
  let pass=0;
  for (const t of GTASKS) {
    let ok=false, val=answers[t.id];
    if (t.type==='code') {
      const attempts = [
        c => new Function('hl', `return (${c})`)(hl),
        c => new Function('hl', `${c}; return typeof f!=='undefined' ? f : undefined`)(hl),
      ];
      val = undefined;
      for (const a of attempts) { try { const v = a(answers[t.id]); if (v!==undefined) { val=v; break; } } catch(e){} }
    }
    try { ok = !!t.verify(val); } catch(e){ ok=false; }
    if (t.type==='choice') { try { ok = !!t.verify(answers[t.id]); } catch(e){ ok=false; } }
    console.log(`${ok?'PASS':'FAIL'}  ${t.id}`); if(ok) pass++;
  }
  console.log(`\n${pass}/${GTASKS.length} passed`);
}
