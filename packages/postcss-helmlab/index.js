/**
 * PostCSS plugin for Helmlab color spaces.
 *
 * Transforms helmlab(), helmlch(), helmgen(), helmgenlch() CSS functions
 * into rgb()/rgba(), color(display-p3 …), and/or color(rec2020 …) output.
 * Also handles linear-gradient(in helmlab, …) and color-mix(in helmlab, …).
 *
 * Usage:
 *   postcss([ require('postcss-helmlab') ])                            // 'both' (sRGB + P3)
 *   postcss([ require('postcss-helmlab')({ outputMode: 'rec2020' }) ])
 *
 * Options:
 *   outputMode: 'srgb' | 'p3' | 'rec2020' | 'both' | 'all'   (default: 'both')
 *     'srgb'    → rgb()/rgba() only.
 *     'p3'      → color(display-p3 …) only.
 *     'rec2020' → color(rec2020 …) only.
 *     'both'    → sRGB fallback first, then P3 override (progressive enhancement).
 *     'all'     → sRGB fallback, then P3, then Rec2020 (widest available wins).
 *
 * Input:
 *   .card { color: helmlab(0.78 0.52 -0.20); }
 *
 * Output (all mode):
 *   .card {
 *     color: rgb(255, 76, 119);
 *     color: color(display-p3 0.97 0.32 0.47);
 *     color: color(rec2020 0.89 0.36 0.45);
 *   }
 */

import postcss from "postcss";
import {
	Helmlab,
	hexToSrgb, srgbToXyz, xyzToDisplayP3, xyzToRec2020, clampRgb,
} from "helmlab";

const h = new Helmlab();

const { cos, sin, PI } = Math;

// ── Color conversion helpers ───────────────────────────────────────

function fmtP3 (n) {
	return n.toFixed(4);
}

function srgbHexToP3String (hex, alpha) {
	const [r, g, b] = hexToSrgb(hex);
	const xyz = srgbToXyz([r, g, b]);
	const p3 = clampRgb(xyzToDisplayP3(xyz));
	if (alpha !== null && alpha !== undefined && alpha !== 1) {
		return `color(display-p3 ${fmtP3(p3[0])} ${fmtP3(p3[1])} ${fmtP3(p3[2])} / ${alpha})`;
	}
	return `color(display-p3 ${fmtP3(p3[0])} ${fmtP3(p3[1])} ${fmtP3(p3[2])})`;
}

function srgbHexToRec2020String (hex, alpha) {
	const [r, g, b] = hexToSrgb(hex);
	const xyz = srgbToXyz([r, g, b]);
	const rec = clampRgb(xyzToRec2020(xyz));
	if (alpha !== null && alpha !== undefined && alpha !== 1) {
		return `color(rec2020 ${fmtP3(rec[0])} ${fmtP3(rec[1])} ${fmtP3(rec[2])} / ${alpha})`;
	}
	return `color(rec2020 ${fmtP3(rec[0])} ${fmtP3(rec[1])} ${fmtP3(rec[2])})`;
}

function labToSrgbString (L, a, b, alpha, space) {
	const hex = (space === "helmlab" || space === "helmlch")
		? h.metric.toHex([L, a, b])
		: h.gen.toHex([L, a, b]);
	const [r, g, bb] = hexToSrgb(hex);
	const ri = Math.round(r * 255);
	const gi = Math.round(g * 255);
	const bi = Math.round(bb * 255);
	if (alpha !== null && alpha !== 1) {
		return `rgba(${ri}, ${gi}, ${bi}, ${alpha})`;
	}
	return `rgb(${ri}, ${gi}, ${bi})`;
}

function labToP3String (L, a, b, alpha, space) {
	// 1.0: BOTH spaces have a true gamut-mapped P3 path → wider colors
	const ns = (space === "helmlab" || space === "helmlch") ? h.metric : h.gen;
	const base = ns.toCss([L, a, b], "display-p3");
	if (alpha !== null && alpha !== 1) {
		return base.replace(/\)$/, ` / ${alpha})`);
	}
	return base;
}

function labToRec2020String (L, a, b, alpha, space) {
	// 1.0: BOTH spaces have a true gamut-mapped Rec2020 path → widest colors
	const ns = (space === "helmlab" || space === "helmlch") ? h.metric : h.gen;
	const base = ns.toCss([L, a, b], "rec2020");
	if (alpha !== null && alpha !== 1) {
		return base.replace(/\)$/, ` / ${alpha})`);
	}
	return base;
}

function lchToLab (L, C, hDeg) {
	const hRad = hDeg * PI / 180;
	return [L, C * cos(hRad), C * sin(hRad)];
}

// ── Parsing helpers ────────────────────────────────────────────────

function findClosingParen (str, pos) {
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

const COLOR_FN_RE = /\b(helmlab|helmlch|helmgen|helmgenlch)\s*\(/g;

function parseColorFn (value) {
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

		const coords = coordStr
			.replace(/,/g, " ")
			.split(/\s+/)
			.filter(Boolean)
			.map((s) => {
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
		}
		else {
			a = second;
			b = third;
		}

		results.push({ L, a, b, alpha, space, fullMatch });
	}

	return results;
}

// ── CSS named colors ───────────────────────────────────────────────

const NAMED_COLORS = {
	red: "#ff0000", green: "#008000", blue: "#0000ff",
	white: "#ffffff", black: "#000000", yellow: "#ffff00",
	cyan: "#00ffff", magenta: "#ff00ff", orange: "#ffa500",
	pink: "#ffc0cb", purple: "#800080", gray: "#808080",
	grey: "#808080", lime: "#00ff00", navy: "#000080",
	teal: "#008080", maroon: "#800000", olive: "#808000",
	aqua: "#00ffff", fuchsia: "#ff00ff", silver: "#c0c0c0",
	coral: "#ff7f50", salmon: "#fa8072", tomato: "#ff6347",
	gold: "#ffd700", indigo: "#4b0082", violet: "#ee82ee",
	khaki: "#f0e68c", plum: "#dda0dd", orchid: "#da70d6",
	tan: "#d2b48c", wheat: "#f5deb3", snow: "#fffafa",
	ivory: "#fffff0", linen: "#faf0e6", beige: "#f5f5dc",
	azure: "#f0ffff", lavender: "#e6e6fa", mintcream: "#f5fffa",
	honeydew: "#f0fff0", seashell: "#fff5ee", oldlace: "#fdf5e6",
	cornsilk: "#fff8dc", bisque: "#ffe4c4", blanchedalmond: "#ffebcd",
	papayawhip: "#ffefd5", peachpuff: "#ffdab9", moccasin: "#ffe4b5",
	mistyrose: "#ffe4e1", aliceblue: "#f0f8ff", ghostwhite: "#f8f8ff",
	whitesmoke: "#f5f5f5", floralwhite: "#fffaf0", antiquewhite: "#faebd7",
	lemonchiffon: "#fffacd", lightyellow: "#ffffe0", lightcyan: "#e0ffff",
	slategray: "#708090", darkgray: "#a9a9a9", lightgray: "#d3d3d3",
	dimgray: "#696969", gainsboro: "#dcdcdc",
	steelblue: "#4682b4", dodgerblue: "#1e90ff", royalblue: "#4169e1",
	cornflowerblue: "#6495ed", deepskyblue: "#00bfff", skyblue: "#87ceeb",
	lightskyblue: "#87cefa", lightblue: "#add8e6", powderblue: "#b0e0e6",
	cadetblue: "#5f9ea0", darkslategray: "#2f4f4f",
	firebrick: "#b22222", darkred: "#8b0000", crimson: "#dc143c",
	indianred: "#cd5c5c", lightcoral: "#f08080", rosybrown: "#bc8f8f",
	darkorange: "#ff8c00", orangered: "#ff4500",
	darkgreen: "#006400", forestgreen: "#228b22", seagreen: "#2e8b57",
	limegreen: "#32cd32", springgreen: "#00ff7f", mediumseagreen: "#3cb371",
	darkslateblue: "#483d8b", mediumslateblue: "#7b68ee", slateblue: "#6a5acd",
	rebeccapurple: "#663399", mediumpurple: "#9370db", blueviolet: "#8a2be2",
	darkorchid: "#9932cc", darkviolet: "#9400d3", darkmagenta: "#8b008b",
	deeppink: "#ff1493", hotpink: "#ff69b4", mediumvioletred: "#c71585",
	palevioletred: "#db7093",
	chocolate: "#d2691e", saddlebrown: "#8b4513", sienna: "#a0522d",
	peru: "#cd853f", sandybrown: "#f4a460", burlywood: "#deb887",
	goldenrod: "#daa520", darkgoldenrod: "#b8860b",
	darkcyan: "#008b8b", lightseagreen: "#20b2aa", mediumturquoise: "#48d1cc",
	turquoise: "#40e0d0", darkturquoise: "#00ced1", aquamarine: "#7fffd4",
	mediumaquamarine: "#66cdaa", paleturquoise: "#afeeee",
	chartreuse: "#7fff00", lawngreen: "#7cfc00", greenyellow: "#adff2f",
	yellowgreen: "#9acd32", olivedrab: "#6b8e23", darkolivegreen: "#556b2f",
	mediumspringgreen: "#00fa9a", palegreen: "#98fb98", lightgreen: "#90ee90",
	darkseagreen: "#8fbc8f",
};

function parseHexColor (str) {
	str = str.trim();
	if (/^#[0-9a-fA-F]{3,8}$/.test(str)) {
		return str;
	}
	if (NAMED_COLORS[str.toLowerCase()]) {
		return NAMED_COLORS[str.toLowerCase()];
	}
	return null;
}

// ── Gradient handling ──────────────────────────────────────────────

const GRADIENT_RE = /linear-gradient\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;

function handleGradient (value, target) {
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

			// h.gradient gives sRGB hex stops with CIEDE2000 arc-length spacing
			const gradientHexes = h.gen.gradient(startHex, endHex, steps);

			const formatStop = (hex, i) => {
				const pct = ((i / (steps - 1)) * 100).toFixed(1);
				if (target === "p3") {
					if (space === "helmlab" || space === "helmlch") {
						return `${h.metric.toCss(h.metric.fromHex(hex), 'display-p3')} ${pct}%`;
					}
					return `${srgbHexToP3String(hex, 1)} ${pct}%`;
				}
				if (target === "rec2020") {
					if (space === "helmlab" || space === "helmlch") {
						return `${h.metric.toCss(h.metric.fromHex(hex), 'rec2020')} ${pct}%`;
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

// ── color-mix handling ─────────────────────────────────────────────

const COLOR_MIX_RE = /color-mix\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;

function handleColorMix (value, target) {
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
		const lab1 = useGen ? h.gen.fromHex(stop1.color) : h.metric.fromHex(stop1.color);
		const lab2 = useGen ? h.gen.fromHex(stop2.color) : h.metric.fromHex(stop2.color);

		const t = stop1.pct / 100;
		const mixed = [
			lab1[0] * t + lab2[0] * (1 - t),
			lab1[1] * t + lab2[1] * (1 - t),
			lab1[2] * t + lab2[2] * (1 - t),
		];

		let replacement;
		if (target === "p3") {
			replacement = labToP3String(mixed[0], mixed[1], mixed[2], 1, space);
		}
		else if (target === "rec2020") {
			replacement = labToRec2020String(mixed[0], mixed[1], mixed[2], 1, space);
		}
		else {
			replacement = labToSrgbString(mixed[0], mixed[1], mixed[2], 1, space);
		}

		result = result.replace(fullMix, replacement);
	}

	return result;
}

// ── Value transform ────────────────────────────────────────────────

function transformValue (value, target) {
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
		}
		else if (target === "rec2020") {
			replacement = labToRec2020String(fn.L, fn.a, fn.b, fn.alpha, fn.space);
		}
		else {
			replacement = labToSrgbString(fn.L, fn.a, fn.b, fn.alpha, fn.space);
		}
		newValue = newValue.replace(fn.fullMatch, replacement);
	}

	return newValue;
}

// ── PostCSS plugin ─────────────────────────────────────────────────

const VALID_MODES = new Set(["srgb", "p3", "rec2020", "both", "all"]);

const SUPPORTS_QUERY = {
	p3: "(color: color(display-p3 0 0 0))",
	rec2020: "(color: color(rec2020 0 0 0))",
};

/**
 * Build a sibling rule that mirrors the original rule's selector but contains
 * a single declaration with the wide-gamut value, wrapped in an @supports query.
 * Older browsers (and CSS minifiers that drop shadowed declarations) fall back
 * to the sRGB declaration in the original rule. @supports blocks are not
 * deduped across, so the cascade survives Lightning CSS / cssnano.
 *
 * Inserts the @supports rule AFTER `anchor` so chained calls preserve order.
 * Returns the inserted @supports node (or null if not applicable).
 */
function appendSupportsOverride (decl, postcss, target, wideValue, anchor) {
	const parentRule = decl.parent;
	if (!parentRule || parentRule.type !== "rule") {
		return null;
	}

	const innerRule = postcss.rule({ selector: parentRule.selector });
	innerRule.append(decl.clone({ value: wideValue }));

	const supports = postcss.atRule({ name: "supports", params: SUPPORTS_QUERY[target] });
	supports.append(innerRule);

	parentRule.parent.insertAfter(anchor, supports);
	return supports;
}

const plugin = (opts = {}) => {
	const outputMode = VALID_MODES.has(opts.outputMode) ? opts.outputMode : "both";

	return {
		postcssPlugin: "postcss-helmlab",

		Declaration (decl) {
			const value = decl.value;
			if (!value.includes("helm")) {
				return;
			}

			// Single-target modes: rewrite the declaration in place
			if (outputMode === "srgb" || outputMode === "p3" || outputMode === "rec2020") {
				const out = transformValue(value, outputMode);
				if (out !== value) {
					decl.value = out;
				}
				return;
			}

			// Progressive enhancement modes:
			//   both → sRGB declaration in place + @supports(P3) override
			//   all  → sRGB in place + @supports(P3) + @supports(Rec2020)
			//
			// We emit @supports siblings (not extra declarations in the same rule)
			// because CSS minifiers (Lightning CSS, cssnano) drop earlier same-property
			// declarations as "shadowed". @supports queries are not deduped across.
			const wideTargets = outputMode === "all" ? ["p3", "rec2020"] : ["p3"];
			const srgbValue = transformValue(value, "srgb");
			if (srgbValue === value) {
				return; // No helmlab funcs actually transformed
			}
			decl.value = srgbValue;

			let anchor = decl.parent;
			for (const target of wideTargets) {
				const wideValue = transformValue(value, target);
				if (wideValue === srgbValue) continue;
				const inserted = appendSupportsOverride(decl, postcss, target, wideValue, anchor);
				if (inserted) anchor = inserted;
			}
		},
	};
};

plugin.postcss = true;
export default plugin;
