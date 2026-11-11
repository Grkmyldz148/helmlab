# postcss-helmlab

[PostCSS] plugin for [Helmlab] color spaces. It turns the CSS functions `helmlab()`, `helmlch()`, `helmgen()` and `helmgenlch()`, and gradients and `color-mix()` in `helmgen`, into `rgb()`/`rgba()` values, plus `color(display-p3 …)` and `color(rec2020 …)` values wrapped in `@supports` queries.

```js
// postcss.config.mjs
import helmlab from 'postcss-helmlab';
export default { plugins: [helmlab({ outputMode: 'all' })] };
```

| Mode      | Output                                                            |
|-----------|-------------------------------------------------------------------|
| `srgb`    | `rgb()` / `rgba()` only                                           |
| `p3`      | `color(display-p3 …)` only                                        |
| `rec2020` | `color(rec2020 …)` only                                           |
| `both`    | sRGB inline **+** `@supports (color: color(display-p3 …))` block  |
| `all`     | sRGB inline **+** P3 `@supports` **+** Rec2020 `@supports` blocks |

`both` is the default.

[PostCSS]: https://postcss.org/
[Helmlab]: https://github.com/Grkmyldz148/helmlab

## About Helmlab

Helmlab defines color spaces for generating colors and for measuring color differences. The [Helmlab README](https://github.com/Grkmyldz148/helmlab#readme) describes the current definition, Helmlab 2026, and its known limitations. This README gives no performance figures.

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

Or with the PostCSS API:

```js
const postcss = require('postcss');
const helmlab = require('postcss-helmlab');

const result = await postcss([helmlab]).process(css);
```

## Syntax

The examples show the output of `outputMode: 'srgb'`. The default mode, `both`, adds a Display P3 version of each rule inside `@supports`.

### Color functions

```css
/* Input */
.card {
  color: helmlab(0.7221 -0.2468 -0.2378);
  border-color: helmlch(0.7221 0.3427 223.9deg);
  background: helmgen(0.60 0.22 0.03);
  outline-color: helmgenlch(0.60 0.15 30deg);
}

/* Output */
.card {
  color: rgb(59, 130, 246);
  border-color: rgb(59, 130, 246);
  background: rgb(229, 93, 124);
  outline-color: rgb(205, 119, 109);
}
```

### Alpha

```css
/* Input */
.overlay { background: helmlab(0.7221 -0.2468 -0.2378 / 0.5); }

/* Output */
.overlay { background: rgba(59, 130, 246, 0.5); }
```

### Gradients

A gradient `in helmgen` becomes 10 stops from `gen.gradient()` of the helmlab package: a straight path in GenSpace with steps of about equal CIEDE2000 length.

```css
/* Input */
.gradient {
  background: linear-gradient(in helmgen, #e63946, #457b9d);
}

/* Output */
.gradient {
  background: linear-gradient(#e63946 0.0%, #cd525a 11.1%, ..., #457b9d 100.0%);
}
```

### Color mixing

```css
/* Input */
.mix { color: color-mix(in helmgen, #e63946 50%, #457b9d); }

/* Output */
.mix { color: rgb(162, 105, 119); }
```

## Supported spaces

| Function | Coordinates |
|----------|-------------|
| `helmlab(L a b)` | MetricSpace as returned by `hl.metric` in the helmlab package: the neutral-corrected form (`helmlab-metric-neutral`) |
| `helmlch(L C h)` | The same MetricSpace coordinates in LCh |
| `helmgen(L a b)` | GenSpace (`helmgen`), the space for gradients and palettes |
| `helmgenlch(L C h)` | GenSpace in LCh (`helmgenlch`) |

`helmlab()` takes neutral-corrected coordinates. They differ from those of the base form `helmlab-metric`, which Color.js and ColorAide implement; the same triple names different colors in the two forms (see the Helmlab README).

## Links

- [Helmlab README](https://github.com/Grkmyldz148/helmlab#readme)
- [API reference](https://github.com/Grkmyldz148/helmlab/blob/main/API.md)
- [Paper (arXiv:2602.23010)](https://arxiv.org/abs/2602.23010)
- [npm: helmlab](https://www.npmjs.com/package/helmlab)

## License

MIT
