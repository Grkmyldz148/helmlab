// Regenerate the OG social card PNG from the built /og-card page.
// The page (src/pages/og-card.astro) derives every number from claims.ts, so the
// PNG always matches the site. Run AFTER `npm run build`:  npm run og
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
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

function shoot(width, height, dest) {
  execFileSync(chrome, [
    '--headless=new', '--disable-gpu', '--hide-scrollbars',
    '--force-device-scale-factor=2', `--window-size=${width},${height}`,
    `--screenshot=${tmp}`, `file://${builtPage}`,
  ], { stdio: 'ignore' });
  return sharp(tmp).resize(width, height).png().toFile(dest);
}

// 1) OG/social preview card (1.91:1)
await shoot(1200, 630, out);
console.log(`wrote ${out} (1200×630) from claims-driven /og-card`);

// 2) Tweet attachment (X in-feed optimal 16:9), version-stamped filename so
//    every release gets a fresh, cache-proof visual ready to attach.
const versionTs = readFileSync(resolve(root, 'src/data/version.ts'), 'utf8');
const version = versionTs.match(/["']([0-9]+\.[0-9]+\.[0-9]+)["']/)[1];
const tw = resolve(root, `public/twitter-card-v${version}.png`);
await shoot(1200, 675, tw);
console.log(`wrote ${tw} (1200×675, X in-feed 16:9)`);
