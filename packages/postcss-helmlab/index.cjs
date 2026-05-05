var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
  // If the importer is in node compatibility mode or this is not an ESM
  // file that has been converted to a CommonJS file using a Babel-
  // compatible transform (i.e. "__esModule" has not been set), then set
  // "default" to the CommonJS "module.exports" for node compatibility.
  isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
  mod
));
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// index.js
var index_exports = {};
__export(index_exports, {
  default: () => index_default
});
module.exports = __toCommonJS(index_exports);
var import_postcss = __toESM(require("postcss"), 1);
var import_helmlab = require("helmlab");
var h = new import_helmlab.Helmlab();
var { cos, sin, PI } = Math;
function fmtP3(n) {
  return n.toFixed(4);
}
function srgbHexToP3String(hex, alpha) {
  const [r, g, b] = (0, import_helmlab.hexToSrgb)(hex);
  const xyz = (0, import_helmlab.srgbToXyz)([r, g, b]);
  const p3 = (0, import_helmlab.clampRgb)((0, import_helmlab.xyzToDisplayP3)(xyz));
  if (alpha !== null && alpha !== void 0 && alpha !== 1) {
    return `color(display-p3 ${fmtP3(p3[0])} ${fmtP3(p3[1])} ${fmtP3(p3[2])} / ${alpha})`;
  }
  return `color(display-p3 ${fmtP3(p3[0])} ${fmtP3(p3[1])} ${fmtP3(p3[2])})`;
}
function srgbHexToRec2020String(hex, alpha) {
  const [r, g, b] = (0, import_helmlab.hexToSrgb)(hex);
  const xyz = (0, import_helmlab.srgbToXyz)([r, g, b]);
  const rec = (0, import_helmlab.clampRgb)((0, import_helmlab.xyzToRec2020)(xyz));
  if (alpha !== null && alpha !== void 0 && alpha !== 1) {
    return `color(rec2020 ${fmtP3(rec[0])} ${fmtP3(rec[1])} ${fmtP3(rec[2])} / ${alpha})`;
  }
  return `color(rec2020 ${fmtP3(rec[0])} ${fmtP3(rec[1])} ${fmtP3(rec[2])})`;
}
function labToSrgbString(L, a, b, alpha, space) {
  const hex = space === "helmlab" || space === "helmlch" ? h.toHex([L, a, b]) : h.genToHex([L, a, b]);
  const [r, g, bb] = (0, import_helmlab.hexToSrgb)(hex);
  const ri = Math.round(r * 255);
  const gi = Math.round(g * 255);
  const bi = Math.round(bb * 255);
  if (alpha !== null && alpha !== 1) {
    return `rgba(${ri}, ${gi}, ${bi}, ${alpha})`;
  }
  return `rgb(${ri}, ${gi}, ${bi})`;
}
function labToP3String(L, a, b, alpha, space) {
  if (space === "helmlab" || space === "helmlch") {
    const base = h.toHexP3([L, a, b]);
    if (alpha !== null && alpha !== 1) {
      return base.replace(/\)$/, ` / ${alpha})`);
    }
    return base;
  }
  const hex = h.genToHex([L, a, b]);
  return srgbHexToP3String(hex, alpha);
}
function labToRec2020String(L, a, b, alpha, space) {
  if (space === "helmlab" || space === "helmlch") {
    const base = h.toHexRec2020([L, a, b]);
    if (alpha !== null && alpha !== 1) {
      return base.replace(/\)$/, ` / ${alpha})`);
    }
    return base;
  }
  const hex = h.genToHex([L, a, b]);
  return srgbHexToRec2020String(hex, alpha);
}
function lchToLab(L, C, hDeg) {
  const hRad = hDeg * PI / 180;
  return [L, C * cos(hRad), C * sin(hRad)];
}
function findClosingParen(str, pos) {
  let depth = 0;
  for (let i = pos; i < str.length; i++) {
    if (str[i] === "(") {
      depth++;
    }
    if (str[i] === ")") {
      depth--;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}
var COLOR_FN_RE = /\b(helmlab|helmlch|helmgen|helmgenlch)\s*\(/g;
function parseColorFn(value) {
  const results = [];
  let match;
  COLOR_FN_RE.lastIndex = 0;
  while ((match = COLOR_FN_RE.exec(value)) !== null) {
    const space = match[1];
    const openIdx = match.index + match[0].length - 1;
    const closeIdx = findClosingParen(value, openIdx);
    if (closeIdx === -1) {
      continue;
    }
    const inner = value.slice(openIdx + 1, closeIdx).trim();
    const fullMatch = value.slice(match.index, closeIdx + 1);
    const parts = inner.split("/");
    const coordStr = parts[0].trim();
    const alphaStr = parts[1]?.trim();
    const coords = coordStr.replace(/,/g, " ").split(/\s+/).filter(Boolean).map((s) => {
      if (s.endsWith("deg")) {
        return parseFloat(s);
      }
      if (s.endsWith("%")) {
        return parseFloat(s) / 100;
      }
      return parseFloat(s);
    });
    if (coords.length !== 3 || coords.some(isNaN)) {
      continue;
    }
    let alpha = null;
    if (alphaStr) {
      alpha = parseFloat(alphaStr);
      if (alphaStr.endsWith("%")) {
        alpha = alpha / 100;
      }
      if (isNaN(alpha)) {
        alpha = null;
      }
    }
    let [L, second, third] = coords;
    let a, b;
    if (space === "helmlch" || space === "helmgenlch") {
      [L, a, b] = lchToLab(L, second, third);
    } else {
      a = second;
      b = third;
    }
    results.push({ L, a, b, alpha, space, fullMatch });
  }
  return results;
}
var NAMED_COLORS = {
  red: "#ff0000",
  green: "#008000",
  blue: "#0000ff",
  white: "#ffffff",
  black: "#000000",
  yellow: "#ffff00",
  cyan: "#00ffff",
  magenta: "#ff00ff",
  orange: "#ffa500",
  pink: "#ffc0cb",
  purple: "#800080",
  gray: "#808080",
  grey: "#808080",
  lime: "#00ff00",
  navy: "#000080",
  teal: "#008080",
  maroon: "#800000",
  olive: "#808000",
  aqua: "#00ffff",
  fuchsia: "#ff00ff",
  silver: "#c0c0c0",
  coral: "#ff7f50",
  salmon: "#fa8072",
  tomato: "#ff6347",
  gold: "#ffd700",
  indigo: "#4b0082",
  violet: "#ee82ee",
  khaki: "#f0e68c",
  plum: "#dda0dd",
  orchid: "#da70d6",
  tan: "#d2b48c",
  wheat: "#f5deb3",
  snow: "#fffafa",
  ivory: "#fffff0",
  linen: "#faf0e6",
  beige: "#f5f5dc",
  azure: "#f0ffff",
  lavender: "#e6e6fa",
  mintcream: "#f5fffa",
  honeydew: "#f0fff0",
  seashell: "#fff5ee",
  oldlace: "#fdf5e6",
  cornsilk: "#fff8dc",
  bisque: "#ffe4c4",
  blanchedalmond: "#ffebcd",
  papayawhip: "#ffefd5",
  peachpuff: "#ffdab9",
  moccasin: "#ffe4b5",
  mistyrose: "#ffe4e1",
  aliceblue: "#f0f8ff",
  ghostwhite: "#f8f8ff",
  whitesmoke: "#f5f5f5",
  floralwhite: "#fffaf0",
  antiquewhite: "#faebd7",
  lemonchiffon: "#fffacd",
  lightyellow: "#ffffe0",
  lightcyan: "#e0ffff",
  slategray: "#708090",
  darkgray: "#a9a9a9",
  lightgray: "#d3d3d3",
  dimgray: "#696969",
  gainsboro: "#dcdcdc",
  steelblue: "#4682b4",
  dodgerblue: "#1e90ff",
  royalblue: "#4169e1",
  cornflowerblue: "#6495ed",
  deepskyblue: "#00bfff",
  skyblue: "#87ceeb",
  lightskyblue: "#87cefa",
  lightblue: "#add8e6",
  powderblue: "#b0e0e6",
  cadetblue: "#5f9ea0",
  darkslategray: "#2f4f4f",
  firebrick: "#b22222",
  darkred: "#8b0000",
  crimson: "#dc143c",
  indianred: "#cd5c5c",
  lightcoral: "#f08080",
  rosybrown: "#bc8f8f",
  darkorange: "#ff8c00",
  orangered: "#ff4500",
  darkgreen: "#006400",
  forestgreen: "#228b22",
  seagreen: "#2e8b57",
  limegreen: "#32cd32",
  springgreen: "#00ff7f",
  mediumseagreen: "#3cb371",
  darkslateblue: "#483d8b",
  mediumslateblue: "#7b68ee",
  slateblue: "#6a5acd",
  rebeccapurple: "#663399",
  mediumpurple: "#9370db",
  blueviolet: "#8a2be2",
  darkorchid: "#9932cc",
  darkviolet: "#9400d3",
  darkmagenta: "#8b008b",
  deeppink: "#ff1493",
  hotpink: "#ff69b4",
  mediumvioletred: "#c71585",
  palevioletred: "#db7093",
  chocolate: "#d2691e",
  saddlebrown: "#8b4513",
  sienna: "#a0522d",
  peru: "#cd853f",
  sandybrown: "#f4a460",
  burlywood: "#deb887",
  goldenrod: "#daa520",
  darkgoldenrod: "#b8860b",
  darkcyan: "#008b8b",
  lightseagreen: "#20b2aa",
  mediumturquoise: "#48d1cc",
  turquoise: "#40e0d0",
  darkturquoise: "#00ced1",
  aquamarine: "#7fffd4",
  mediumaquamarine: "#66cdaa",
  paleturquoise: "#afeeee",
  chartreuse: "#7fff00",
  lawngreen: "#7cfc00",
  greenyellow: "#adff2f",
  yellowgreen: "#9acd32",
  olivedrab: "#6b8e23",
  darkolivegreen: "#556b2f",
  mediumspringgreen: "#00fa9a",
  palegreen: "#98fb98",
  lightgreen: "#90ee90",
  darkseagreen: "#8fbc8f"
};
function parseHexColor(str) {
  str = str.trim();
  if (/^#[0-9a-fA-F]{3,8}$/.test(str)) {
    return str;
  }
  if (NAMED_COLORS[str.toLowerCase()]) {
    return NAMED_COLORS[str.toLowerCase()];
  }
  return null;
}
var GRADIENT_RE = /linear-gradient\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;
function handleGradient(value, target) {
  let result = value;
  let match;
  GRADIENT_RE.lastIndex = 0;
  while ((match = GRADIENT_RE.exec(value)) !== null) {
    const space = match[1];
    const gradStart = match.index;
    const argsStart = match.index + match[0].length;
    const closeIdx = findClosingParen(value, value.indexOf("(", gradStart));
    if (closeIdx === -1) {
      continue;
    }
    const argsStr = value.slice(argsStart, closeIdx).trim();
    const fullGradient = value.slice(gradStart, closeIdx + 1);
    const stopParts = argsStr.split(",").map((s) => s.trim()).filter(Boolean);
    const hexStops = [];
    for (const part of stopParts) {
      const tokens = part.split(/\s+/);
      const colorPart = tokens[0];
      const hex = parseHexColor(colorPart);
      if (!hex) {
        hexStops.length = 0;
        break;
      }
      hexStops.push(hex);
    }
    if (hexStops.length >= 2) {
      const steps = 10;
      const startHex = hexStops[0];
      const endHex = hexStops[hexStops.length - 1];
      const gradientHexes = h.gradient(startHex, endHex, steps);
      const formatStop = (hex, i) => {
        const pct = (i / (steps - 1) * 100).toFixed(1);
        if (target === "p3") {
          if (space === "helmlab" || space === "helmlch") {
            return `${h.toHexP3(h.fromHex(hex))} ${pct}%`;
          }
          return `${srgbHexToP3String(hex, 1)} ${pct}%`;
        }
        if (target === "rec2020") {
          if (space === "helmlab" || space === "helmlch") {
            return `${h.toHexRec2020(h.fromHex(hex))} ${pct}%`;
          }
          return `${srgbHexToRec2020String(hex, 1)} ${pct}%`;
        }
        return `${hex} ${pct}%`;
      };
      const cssStops = gradientHexes.map(formatStop).join(", ");
      result = result.replace(fullGradient, `linear-gradient(${cssStops})`);
    }
  }
  return result;
}
var COLOR_MIX_RE = /color-mix\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;
function handleColorMix(value, target) {
  let result = value;
  let match;
  COLOR_MIX_RE.lastIndex = 0;
  while ((match = COLOR_MIX_RE.exec(value)) !== null) {
    const space = match[1];
    const mixStart = match.index;
    const openIdx = value.indexOf("(", mixStart);
    const closeIdx = findClosingParen(value, openIdx);
    if (closeIdx === -1) {
      continue;
    }
    const argsStr = value.slice(match.index + match[0].length, closeIdx).trim();
    const fullMix = value.slice(mixStart, closeIdx + 1);
    const parts = argsStr.split(",").map((s) => s.trim());
    if (parts.length !== 2) {
      continue;
    }
    const parseStop = (s) => {
      const tokens = s.trim().split(/\s+/);
      const color = tokens[0];
      let pct = 50;
      if (tokens.length > 1 && tokens[1].endsWith("%")) {
        pct = parseFloat(tokens[1]);
      }
      return { color: parseHexColor(color), pct };
    };
    const stop1 = parseStop(parts[0]);
    const stop2 = parseStop(parts[1]);
    if (!stop1.color || !stop2.color) {
      continue;
    }
    const useGen = space === "helmgen" || space === "helmgenlch";
    const lab1 = useGen ? h.genFromHex(stop1.color) : h.fromHex(stop1.color);
    const lab2 = useGen ? h.genFromHex(stop2.color) : h.fromHex(stop2.color);
    const t = stop1.pct / 100;
    const mixed = [
      lab1[0] * t + lab2[0] * (1 - t),
      lab1[1] * t + lab2[1] * (1 - t),
      lab1[2] * t + lab2[2] * (1 - t)
    ];
    let replacement;
    if (target === "p3") {
      replacement = labToP3String(mixed[0], mixed[1], mixed[2], 1, space);
    } else if (target === "rec2020") {
      replacement = labToRec2020String(mixed[0], mixed[1], mixed[2], 1, space);
    } else {
      replacement = labToSrgbString(mixed[0], mixed[1], mixed[2], 1, space);
    }
    result = result.replace(fullMix, replacement);
  }
  return result;
}
function transformValue(value, target) {
  let newValue = value;
  if (GRADIENT_RE.test(newValue)) {
    newValue = handleGradient(newValue, target);
  }
  if (COLOR_MIX_RE.test(newValue)) {
    newValue = handleColorMix(newValue, target);
  }
  const fns = parseColorFn(newValue);
  for (const fn of fns) {
    let replacement;
    if (target === "p3") {
      replacement = labToP3String(fn.L, fn.a, fn.b, fn.alpha, fn.space);
    } else if (target === "rec2020") {
      replacement = labToRec2020String(fn.L, fn.a, fn.b, fn.alpha, fn.space);
    } else {
      replacement = labToSrgbString(fn.L, fn.a, fn.b, fn.alpha, fn.space);
    }
    newValue = newValue.replace(fn.fullMatch, replacement);
  }
  return newValue;
}
var VALID_MODES = /* @__PURE__ */ new Set(["srgb", "p3", "rec2020", "both", "all"]);
var SUPPORTS_QUERY = {
  p3: "(color: color(display-p3 0 0 0))",
  rec2020: "(color: color(rec2020 0 0 0))"
};
function appendSupportsOverride(decl, postcss2, target, wideValue, anchor) {
  const parentRule = decl.parent;
  if (!parentRule || parentRule.type !== "rule") {
    return null;
  }
  const innerRule = postcss2.rule({ selector: parentRule.selector });
  innerRule.append(decl.clone({ value: wideValue }));
  const supports = postcss2.atRule({ name: "supports", params: SUPPORTS_QUERY[target] });
  supports.append(innerRule);
  parentRule.parent.insertAfter(anchor, supports);
  return supports;
}
var plugin = (opts = {}) => {
  const outputMode = VALID_MODES.has(opts.outputMode) ? opts.outputMode : "both";
  return {
    postcssPlugin: "postcss-helmlab",
    Declaration(decl) {
      const value = decl.value;
      if (!value.includes("helm")) {
        return;
      }
      if (outputMode === "srgb" || outputMode === "p3" || outputMode === "rec2020") {
        const out = transformValue(value, outputMode);
        if (out !== value) {
          decl.value = out;
        }
        return;
      }
      const wideTargets = outputMode === "all" ? ["p3", "rec2020"] : ["p3"];
      const srgbValue = transformValue(value, "srgb");
      if (srgbValue === value) {
        return;
      }
      decl.value = srgbValue;
      let anchor = decl.parent;
      for (const target of wideTargets) {
        const wideValue = transformValue(value, target);
        if (wideValue === srgbValue) continue;
        const inserted = appendSupportsOverride(decl, import_postcss.default, target, wideValue, anchor);
        if (inserted) anchor = inserted;
      }
    }
  };
};
plugin.postcss = true;
var index_default = plugin;
module.exports = module.exports.default; module.exports.default = module.exports; module.exports.postcss = true;
