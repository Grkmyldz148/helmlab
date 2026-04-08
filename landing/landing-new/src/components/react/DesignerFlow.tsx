// DesignerFlow — Story-driven visual experience for designers
// React component (instead of Astro) to avoid esbuild template parsing issues

import MiniPlayground from "./MiniPlayground";

// REAL computed gradient values (not hardcoded approximations)
const gradients = {
  // Blue→White: OKLab midpoint #8ba8ff (blueish-white), Helmlab midpoint #649cff (stronger blue)
  blueOk: "linear-gradient(to right, #0000ff, #3c5cff, #6586ff, #8ba8ff, #b1c6ff, #d8e3ff, #ffffff)",
  blueHl: "linear-gradient(to right, #0000ff, #104fff, #3375ff, #649cff, #a3c4ff, #d9e5ff, #ffffff)",
  // Red→White: very similar — both stay warm
  redOk: "linear-gradient(to right, #ff0000, #ff5545, #ff7e6d, #ffa192, #ffc1b6, #ffe0da, #ffffff)",
  redHl: "linear-gradient(to right, #ff0000, #ff5846, #ff8675, #ffada0, #ffcfc7, #ffeae6, #ffffff)",
  // Yellow→White: OKLab slightly more saturated early, Helmlab more linear fade
  yelOk: "linear-gradient(to right, #facc15, #fbd65c, #fcdf84, #fde8a5, #fef0c4, #fff8e2, #ffffff)",
  yelHl: "linear-gradient(to right, #facc15, #fbd766, #fbe294, #fcebb8, #fdf3d6, #fefaed, #ffffff)",
  // Teal→Black: computed properly
  tealOk: "linear-gradient(to right, #14b8a6, #0d8a7e, #075e57, #033633, #011514, #000000)",
  tealHl: "linear-gradient(to right, #14b8a6, #109e90, #0c8479, #086a62, #04504b, #000000)",
};

const palOk = ["#eff6ff","#dbeafe","#bfdbfe","#93bbfd","#6296f5","#3b82f6","#2563eb","#1d4ed8","#1e3a8a","#172554","#0c1427"];
const palHl = ["#eef4ff","#d5e3ff","#b5cfff","#8db5fe","#6398f8","#3b82f6","#2968d4","#1d50aa","#153c82","#0e2a5c","#081a3a"];

const faqs = [
  { q: "Do I need to know color science?", a: "No. Import Helmlab, give it a hex color, get better results back. The math is hidden." },
  { q: "Does it work with Tailwind / Figma / Tokens Studio?", a: "Helmlab outputs hex values and CSS variables. It works with anything that accepts standard color formats. Tailwind config export is built in." },
  { q: "Why can't I just use oklch() in CSS?", a: "You can! oklch() is great for CSS-only gradients. But if you're generating colors in JavaScript (design tokens, palette tools, theme generators), Helmlab gives visibly better results because it uses a richer perceptual model." },
  { q: "Is it free?", a: "Yes. MIT licensed. Use it in any personal or commercial project." },
  { q: "How much does it add to my bundle?", a: "11.6 KB gzipped, zero dependencies. About the same as a small icon set." },
  { q: "What if I need to match brand colors exactly?", a: "Helmlab is exactly invertible. The same hex always produces the same Lab values, and vice versa. Zero rounding error." },
];

function GradientStrip({ label, bg, ok }: { label: string; bg: string; ok?: boolean }) {
  return (
    <div>
      <div className="h-8 rounded-lg" style={{ background: bg }} />
      <div className={`text-xs mt-1 ${ok ? "text-zinc-500" : "text-emerald-400"}`}>{label}</div>
    </div>
  );
}

export default function DesignerFlow() {
  return (
    <>
      {/* SECTION 1: The Blue Problem */}
      <section className="py-20 md:py-28">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-widest text-blue-400 font-medium">The Problem</span>
            <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">You've seen this before</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              You pick a vivid blue. You make a gradient to white. And the midpoint loses its punch — it fades out instead of staying blue. The difference is subtle, but once you see it, you can't unsee it.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div className="rounded-2xl border border-white/5 bg-zinc-950 p-6">
              <div className="text-xs text-red-400 uppercase tracking-widest font-medium mb-3">OKLab (the current standard)</div>
              <div className="h-16 rounded-xl mb-3" style={{ background: gradients.blueOk }} />
              <p className="text-sm text-zinc-500">The midpoint (#8ba8ff) loses blue intensity — it fades toward a pale, washed-out periwinkle.</p>
            </div>
            <div className="rounded-2xl border border-emerald-400/20 bg-zinc-950 p-6">
              <div className="text-xs text-emerald-400 uppercase tracking-widest font-medium mb-3">Helmlab</div>
              <div className="h-16 rounded-xl mb-3" style={{ background: gradients.blueHl }} />
              <p className="text-sm text-zinc-400">The midpoint (#649cff) holds stronger blue saturation through the transition.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="rounded-xl border border-white/5 bg-zinc-950 p-4">
              <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Red to White</div>
              <div className="space-y-2">
                <GradientStrip label="OKLab — similar warmth" bg={gradients.redOk} ok />
                <GradientStrip label="Helmlab — very similar" bg={gradients.redHl} />
              </div>
            </div>
            <div className="rounded-xl border border-white/5 bg-zinc-950 p-4">
              <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Yellow to White</div>
              <div className="space-y-2">
                <GradientStrip label="OKLab — slightly more saturated early" bg={gradients.yelOk} ok />
                <GradientStrip label="Helmlab — more linear fade" bg={gradients.yelHl} />
              </div>
            </div>
            <div className="rounded-xl border border-white/5 bg-zinc-950 p-4">
              <div className="text-xs text-zinc-500 uppercase tracking-wider mb-2">Teal to Black</div>
              <div className="space-y-2">
                <GradientStrip label="OKLab — slightly compressed darks" bg={gradients.tealOk} ok />
                <GradientStrip label="Helmlab — smoother dark transition" bg={gradients.tealHl} />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 2: The Palette Problem */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-widest text-violet-400 font-medium">Palettes</span>
            <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">The dark hole problem</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              You generate a 50-950 scale from your brand color. But 700-800 looks like a black hole — the steps aren't even. That's because OKLab's lightness channel isn't actually uniform.
            </p>
          </div>

          <div className="space-y-8">
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-red-400 font-medium">OKLab palette</span>
                <span className="text-xs text-zinc-600">from #3b82f6</span>
              </div>
              <div className="flex rounded-xl overflow-hidden h-14">
                {palOk.map((c, i) => <div key={i} className="flex-1" style={{ background: c }} />)}
              </div>
              <p className="text-xs text-zinc-500 mt-2">Notice the jump between 700 and 800. The dark end feels compressed.</p>
            </div>
            <div>
              <div className="flex items-center gap-2 mb-3">
                <span className="text-xs text-emerald-400 font-medium">Helmlab palette</span>
                <span className="text-xs text-zinc-600">from #3b82f6</span>
              </div>
              <div className="flex rounded-xl overflow-hidden h-14">
                {palHl.map((c, i) => <div key={i} className="flex-1" style={{ background: c }} />)}
              </div>
              <p className="text-xs text-zinc-400 mt-2">Every step looks evenly spaced. The dark end is smooth. 18x better lightness uniformity.</p>
            </div>
          </div>
        </div>
      </section>

      {/* SECTION 3: Try it yourself */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-12">
            <span className="text-xs uppercase tracking-widest text-orange-500 font-medium">Try It</span>
            <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">See it with your own colors</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              Pick two colors and compare the gradient. Then try the full playground for palettes and more.
            </p>
          </div>
          <MiniPlayground />
          <div className="text-center mt-8">
            <a href="/playground" className="inline-flex items-center gap-2 rounded-xl bg-orange-500 px-6 py-3 text-sm font-medium text-white hover:bg-orange-600 transition-colors">
              Full Playground
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" /></svg>
            </a>
          </div>
        </div>
      </section>

      {/* SECTION 4: Use cases */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-widest text-emerald-400 font-medium">Use Cases</span>
            <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">Built for the work you actually do</h2>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { title: "Design Systems", desc: "Generate a full 50-950 Tailwind-style scale from any brand color. Every step is perceptually even. Export to CSS variables, Tailwind config, or design tokens.", code: 'hl.semanticScale("#3b82f6")' },
              { title: "Dark Mode", desc: "Dark gradients that don't turn muddy. 26% smoother in the dark range where OKLab struggles most. Your dark theme looks intentional, not accidental.", code: 'hl.adaptToMode("#3b82f6", "light", "dark")' },
              { title: "Data Visualization", desc: "Categorical palettes with maximum separation. Your chart colors stay distinct even for colorblind users.", code: "hl.palette_hues(lightness=0.6, steps=8)" },
              { title: "Wide Gamut (P3)", desc: "Modern displays show colors outside sRGB. Helmlab maps them correctly — no ugly hue shifts when clipping P3 colors to sRGB.", code: 'hl.gamutMap("#ff0080", "srgb")' },
            ].map((uc) => (
              <div key={uc.title} className="rounded-2xl border border-white/5 bg-zinc-950 p-8">
                <div className="text-2xl mb-3">{uc.title}</div>
                <p className="text-sm text-zinc-400 mb-4">{uc.desc}</p>
                <code className="text-xs text-emerald-400 font-mono">{uc.code}</code>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* SECTION 5: Honest limits */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-5xl mx-auto px-6">
          <div className="text-center mb-16">
            <span className="text-xs uppercase tracking-widest text-red-400 font-medium">Honest</span>
            <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">When to use something else</h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">We could hide this. We won't. You deserve to know before choosing.</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {[
              { t: "CSS-only gradients", d: "If you need gradients in pure CSS without JavaScript, use oklch() — it's built into browsers." },
              { t: "Very subtle gray gradients", d: "Gradients between nearly identical grays are 24% smoother in OKLab. It's our biggest weakness." },
              { t: "Green-blind accessibility", d: "OKLab gives 43% better step distinction for deuteranopia palettes. Use it for CVD-critical designs." },
              { t: "HDR displays", d: "For >1000 cd/m2 content, use Jzazbz. Helmlab is optimized for standard dynamic range." },
              { t: "Smallest possible bundle", d: "OKLab is ~2KB. Helmlab is 11.6KB gzipped. If every kilobyte counts, OKLab is lighter." },
              { t: "Legacy CIE Lab workflows", d: "Decades of industry hue naming is based on CIE Lab. If you need interop with those systems, stick with Lab." },
            ].map((item) => (
              <div key={item.t} className="rounded-2xl border border-white/5 bg-zinc-950 p-6">
                <div className="text-sm font-semibold mb-2">{item.t}</div>
                <p className="text-xs text-zinc-400">{item.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* SECTION 6: Getting started */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-3xl mx-auto px-6 text-center">
          <span className="text-xs uppercase tracking-widest text-orange-500 font-medium">Get Started</span>
          <h2 className="text-3xl sm:text-4xl font-bold mt-3 mb-4">Ask your developer for three lines</h2>
          <p className="text-zinc-400 max-w-xl mx-auto mb-8">One install, one import, one function call. Then hand the hex values to your design tokens.</p>
          <div className="rounded-2xl border border-white/5 bg-zinc-950 p-6 text-left mb-8">
            <pre className="text-sm font-mono leading-relaxed overflow-x-auto">
              <code>
                <span className="text-zinc-500">{"// That's it. Really."}</span>{"\n"}
                <span className="text-purple-400">import</span>{" { Helmlab } "}<span className="text-purple-400">from</span> <span className="text-orange-400">'helmlab'</span>;{"\n"}
                <span className="text-purple-400">const</span> scale = <span className="text-purple-400">new</span> <span className="text-blue-400">Helmlab</span>().<span className="text-emerald-400">semanticScale</span>(<span className="text-orange-400">'#3b82f6'</span>);
              </code>
            </pre>
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <a href="/playground" className="inline-flex items-center gap-2 rounded-xl bg-orange-500 px-6 py-3 text-sm font-medium text-white hover:bg-orange-600 transition-colors">Try the Playground</a>
            <a href="/docs" className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/5 px-6 py-3 text-sm font-medium text-zinc-400 hover:text-white transition-colors">Read the Docs</a>
          </div>
        </div>
      </section>

      {/* SECTION 7: Designer FAQ */}
      <section className="py-20 md:py-28 border-t border-white/5">
        <div className="max-w-3xl mx-auto px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl sm:text-4xl font-bold mb-4">Common questions</h2>
          </div>
          <div className="space-y-3">
            {faqs.map((faq) => (
              <details key={faq.q} className="faq-item group rounded-2xl border border-white/5 bg-zinc-950 overflow-hidden">
                <summary className="flex items-center justify-between cursor-pointer select-none px-6 py-5 text-sm font-medium text-zinc-50 hover:bg-white/5 transition-colors list-none">
                  <span>{faq.q}</span>
                  <svg className="shrink-0 ml-4 w-4 h-4 text-zinc-500 transition-transform duration-300 group-open:rotate-45" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" /></svg>
                </summary>
                <div className="faq-content">
                  <div className="faq-inner">
                    <p className="px-6 pb-5 text-sm text-zinc-400 leading-relaxed">{faq.a}</p>
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
