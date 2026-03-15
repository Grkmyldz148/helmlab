# postcss-helmlab

[PostCSS] plugin for [Helmlab] color spaces. Transforms `helmlab()`, `helmlch()`, `helmgen()`, `helmgenlch()` CSS functions into `rgb()`/`rgba()` fallbacks at build time.

[PostCSS]: https://postcss.org/
[Helmlab]: https://github.com/Grkmyldz148/helmlab

## Why?

Helmlab is a data-driven color space family trained on 64,000+ human color perception observations. It provides two spaces:

- **MetricSpace** — 20% lower STRESS than CIEDE2000 on perceptual distance measurement
- **GenSpace** — wins 28/43 perceptual benchmarks vs OKLab (6/43) for color generation

This plugin lets you use Helmlab color spaces in CSS today, without waiting for browser support.

## Installation

```bash
npm install postcss-helmlab
```

## Usage

Add `postcss-helmlab` to your PostCSS config:

```js
// postcss.config.js
module.exports = {
  plugins: [
    require('postcss-helmlab'),
  ],
};
```

Or with PostCSS API:

```js
const postcss = require('postcss');
const helmlab = require('postcss-helmlab');

const result = await postcss([helmlab]).process(css);
```

## Syntax

### Color functions

```css
/* Input */
.card {
  color: helmlab(0.78 0.52 -0.20);
  background: helmgen(0.60 0.22 0.03);
  border-color: helmlch(0.78 0.56 338.7deg);
  outline-color: helmgenlch(0.60 0.15 30deg);
}

/* Output */
.card {
  color: rgb(255, 76, 119);
  background: rgb(196, 107, 68);
  border-color: rgb(255, 67, 131);
  outline-color: rgb(175, 120, 91);
}
```

### Alpha

```css
/* Input */
.overlay { background: helmlab(0.78 0.52 -0.20 / 0.5); }

/* Output */
.overlay { background: rgba(255, 76, 119, 0.5); }
```

### Gradients

Interpolate gradients through Helmlab spaces with perceptually uniform steps:

```css
/* Input */
.gradient {
  background: linear-gradient(in helmgen, #e63946, #457b9d);
}

/* Output — 10 perceptually spaced stops */
.gradient {
  background: linear-gradient(#e63946 0.0%, #c7555e 11.1%, ..., #457b9d 100.0%);
}
```

### Color mixing

```css
/* Input */
.mix { color: color-mix(in helmgen, #e63946 50%, #457b9d); }

/* Output */
.mix { color: rgb(147, 93, 111); }
```

## Supported spaces

| Function | Space | Best for |
|----------|-------|----------|
| `helmlab(L a b)` | MetricSpace | Perceptual distance, accessibility |
| `helmlch(L C h)` | MetricSpace (cylindrical) | Hue-based selection |
| `helmgen(L a b)` | GenSpace | Gradients, palettes |
| `helmgenlch(L C h)` | GenSpace (cylindrical) | Hue-based generation |

## Links

- [Helmlab documentation](https://grkmyldz148.github.io/helmlab/)
- [GitHub](https://github.com/Grkmyldz148/helmlab)
- [arXiv paper](https://arxiv.org/abs/2602.23010)
- [npm: helmlab](https://www.npmjs.com/package/helmlab)

## License

MIT
