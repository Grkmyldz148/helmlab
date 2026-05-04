// @ts-check
import { defineConfig } from 'astro/config';
import { fileURLToPath } from 'node:url';

import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import postcss from 'postcss';
import postcssHelmlab from 'postcss-helmlab';

// Pre-transform .css files: rewrite helmlab() / helmlch() / helmgen() / helmgenlch()
// into a sRGB-fallback + display-p3 + rec2020 cascade BEFORE Tailwind v4 sees the file.
// This is what lets the landing dogfood Helmlab as its own color basis.
const helmlabPreTransform = {
  name: 'helmlab-pre-transform',
  enforce: 'pre',
  async transform(src, id) {
    if (!id.endsWith('.css')) return null;
    if (!src.includes('helm')) return null;
    const result = await postcss([postcssHelmlab({ outputMode: 'all' })])
      .process(src, { from: id });
    return { code: result.css, map: result.map };
  },
};

// https://astro.build/config
export default defineConfig({
  site: 'https://helmlab.space',
  integrations: [react(), sitemap()],

  vite: {
    plugins: [helmlabPreTransform, tailwindcss()],
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
      },
    },
  }
});
