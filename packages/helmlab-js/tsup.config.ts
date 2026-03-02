import { defineConfig } from 'tsup';

export default defineConfig([
  // ESM + CJS for bundlers (npm)
  {
    entry: ['src/index.ts'],
    format: ['esm', 'cjs'],
    dts: true,
    sourcemap: true,
    clean: true,
    minify: true,
    target: 'es2020',
    noExternal: [/.*/],
  },
  // IIFE for <script> tag — exposes window.helmlab
  {
    entry: ['src/index.ts'],
    format: ['iife'],
    globalName: 'helmlab',
    outDir: 'dist',
    sourcemap: true,
    minify: true,
    target: 'es2020',
    noExternal: [/.*/],
    outExtension: () => ({ js: '.global.js' }),
  },
]);
