// Regenerate the OG social card PNG from the built /og-card page.
// The page (src/pages/og-card.astro) derives every number from claims.ts, so the
// PNG always matches the site. Run AFTER `npm run build`:  npm run og
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const builtPage = resolve(root, 'dist/og-card/index.html');
const out = resolve(root, 'public/og-card.png');
const tmp = '/tmp/og-card-2x.png';

if (!existsSync(builtPage)) {
  console.error('dist/og-card/index.html not found — run `npm run build` first.');
  process.exit(1);
}

const chrome = [
  process.env.CHROME_BIN,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/Applications/Chromium.app/Contents/MacOS/Chromium',
  '/usr/bin/google-chrome',
  '/usr/bin/chromium',
  '/usr/bin/chromium-browser',
].filter(Boolean).find(existsSync);

if (!chrome) {
  console.error('No Chrome/Chromium found. Set CHROME_BIN to a browser binary.');
  process.exit(1);
}

execFileSync(chrome, [
  '--headless=new', '--disable-gpu', '--hide-scrollbars',
  '--force-device-scale-factor=2', '--window-size=1200,630',
  `--screenshot=${tmp}`, `file://${builtPage}`,
], { stdio: 'ignore' });

await sharp(tmp).resize(1200, 630).png().toFile(out);
console.log(`wrote ${out} (1200×630) from claims-driven /og-card`);
