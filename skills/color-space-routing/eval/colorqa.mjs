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
    verify: (d) => typeof d==='number' && d>0.01 && d<0.08 }, // rectangular-form chord ≈ 2*C*sin(10°)≈0.035; naive LCh gives 340
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
];

const mode = process.argv[2];
if (mode === 'list') for (const t of TASKS) console.log(`\n[${t.id}] (${t.type})\n${t.prompt}`);
if (mode === 'verify') {
  const answers = JSON.parse(await import('node:fs').then(fs=>fs.readFileSync(process.argv[3],'utf8')));
  let pass=0;
  for (const t of TASKS) {
    let ok=false, val=answers[t.id];
    try { if (t.type==='code') val = eval(val); ok = !!t.verify(val); } catch(e){ ok=false; }
    if (t.type==='choice') { try { ok = !!t.verify(answers[t.id]); } catch(e){ ok=false; } }
    console.log(`${ok?'PASS':'FAIL'}  ${t.id}`); if(ok) pass++;
  }
  console.log(`\n${pass}/${TASKS.length} passed`);
}
