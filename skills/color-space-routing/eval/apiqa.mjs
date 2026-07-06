// API-QA: does the `helmlab` skill teach correct package usage?
import { Helmlab } from 'helmlab';
const hl = new Helmlab();
export const TASKS = [
  { id:'A1-export-space', type:'code',
    prompt:'Using helmlab (`hl` available), write a JS expression producing the CSS oklch() string for #3b82f6 via the TokenExporter.',
    verify: (s)=> /oklch\(6[12]/.test(String(s)) },
  { id:'A2-palette-dir', type:'choice',
    prompt:'helmlab: does hl.palette("#3b82f6",10) return colors light-to-dark or dark-to-light? Two words.',
    verify: (s)=> /light[\s-]*to[\s-]*dark|light.*dark/i.test(String(s)) && !/dark[\s-]*to[\s-]*light/i.test(String(s)) },
  { id:'A3-exact-500', type:'choice',
    prompt:'helmlab: which method returns a Tailwind-style scale where level 500 is EXACTLY the input color? One method name.',
    verify: (s)=> /semantic/i.test(String(s)) },
  { id:'A4-darken-scale', type:'code',
    prompt:'Using helmlab (`hl`), write a JS function f(hex) that darkens a color noticeably while keeping its hue, via GenSpace LCh.',
    verify: (f)=>{ const out=f('#3b82f6'); const a=hl.genToLch(hl.genFromHex('#3b82f6')), b=hl.genToLch(hl.genFromHex(out));
      let dh=Math.abs(a[2]-b[2]); dh=Math.min(dh,360-dh); return b[0]<a[0]-0.05 && b[0]>0.05 && dh<10; } },
  { id:'A5-trained-metric', type:'code',
    prompt:'Using helmlab (`hl`), write a JS expression: the TRAINED perceptual difference between #ff0000 and #00ff00 (hex inputs).',
    verify: (d)=> typeof d==='number' && Math.abs(d-0.148)<0.01 },
  { id:'A6-noticeable-field', type:'choice',
    prompt:'helmlab differenceWithConfidence(): which output field gives the probability an observer would notice the difference? One word.',
    verify: (s)=> /pnoticeable|p_noticeable/i.test(String(s)) },
  { id:'A7-py-distance', type:'choice',
    prompt:'helmlab Python: MetricSpace.distance(x1, x2) expects which input coordinates? One word.',
    verify: (s)=> /xyz/i.test(String(s)) },
  { id:'A8-gradient', type:'code',
    prompt:'Using helmlab (`hl`), write a JS expression for a perceptually uniform 12-step gradient from #ef4444 to #3b82f6.',
    verify: (g)=> Array.isArray(g) && g.length===12 && g[0]==='#ef4444' },
];
