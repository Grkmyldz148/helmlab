import { defineConfig } from 'vitest/config';
import pkg from './package.json';

export default defineConfig({
  define: {
    __HELMLAB_VERSION__: JSON.stringify(pkg.version),
  },
  test: {
    globals: true,
  },
});
