/**
 * PostCSS plugin for Helmlab color spaces.
 *
 * Transforms helmlab(), helmlch(), helmgen(), helmgenlch() CSS functions
 * into rgb()/rgba() fallbacks. Also handles linear-gradient(in helmlab, ...)
 * and color-mix(in helmlab, ...).
 *
 * Usage:
 *   postcss([ require('postcss-helmlab') ])
 *
 * Input:
 *   .card { color: helmlab(0.78 0.52 -0.20); }
 *   .bg   { background: linear-gradient(in helmgen, #e63946, #457b9d); }
 *
 * Output:
 *   .card { color: rgb(255, 76, 119); }
 *   .bg   { background: linear-gradient(#e63946, #c0555e, ..., #457b9d); }
 */

import { Helmlab, hexToSrgb, srgbToHex, srgbToXyz, xyzToSrgb, clampRgb } from "helmlab";

const h = new Helmlab();

const { sqrt, cos, sin, atan2, PI } = Math;

// ── Color conversion helpers ───────────────────────────────────────

function labToRgbString (L, a, b, alpha, space) {
	let hex;
	if (space === "helmlab" || space === "helmlch") {
		hex = h.toHex([L, a, b]);
	}
	else {
		hex = h.genToHex([L, a, b]);
	}
	const [r, g, bb] = hexToSrgb(hex);
	const ri = Math.round(r * 255);
	const gi = Math.round(g * 255);
	const bi = Math.round(bb * 255);
	if (alpha !== null && alpha !== 1) {
		return `rgba(${ri}, ${gi}, ${bi}, ${alpha})`;
	}
	return `rgb(${ri}, ${gi}, ${bi})`;
}

function lchToLab (L, C, hDeg) {
	const hRad = hDeg * PI / 180;
	return [L, C * cos(hRad), C * sin(hRad)];
}

// ── Parsing helpers ────────────────────────────────────────────────

/**
 * Find matching closing paren for a function call starting at pos.
 * pos should point to the opening paren.
 */
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

/**
 * Parse a helmlab/helmlch/helmgen/helmgenlch function call.
 * Returns { L, a, b, alpha, space } or null.
 */
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

		// Split by "/" for alpha: "L a b / alpha" or "L C hdeg / alpha"
		const parts = inner.split("/");
		const coordStr = parts[0].trim();
		const alphaStr = parts[1]?.trim();

		// Parse coordinates — space or comma separated
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

		// For LCH spaces, convert to Lab
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

// ── Gradient handling ──────────────────────────────────────────────

/**
 * Parse "linear-gradient(in helmlab, color1, color2)" or similar.
 * Also handles "in helmgen", "in helmlch", "in helmgenlch".
 */
const GRADIENT_RE = /linear-gradient\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;

function parseHexColor (str) {
	str = str.trim();
	// Try direct hex
	if (/^#[0-9a-fA-F]{3,8}$/.test(str)) {
		return str;
	}
	// Named CSS colors — use a small map of common ones
	const named = {
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
	if (named[str.toLowerCase()]) {
		return named[str.toLowerCase()];
	}
	return null;
}

function handleGradient (value) {
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

		// Parse color stops: "color1, color2" or "color1 10%, color2 90%"
		// Simple approach: split by comma and extract colors
		const stopParts = argsStr.split(",").map((s) => s.trim()).filter(Boolean);

		const hexStops = [];
		for (const part of stopParts) {
			const tokens = part.split(/\s+/);
			const colorPart = tokens[0];
			const hex = parseHexColor(colorPart);
			if (!hex) {
				// Can't parse this color, skip the whole gradient
				hexStops.length = 0;
				break;
			}
			hexStops.push(hex);
		}

		if (hexStops.length >= 2) {
			// Generate gradient steps using Helmlab
			const steps = 10;
			const startHex = hexStops[0];
			const endHex = hexStops[hexStops.length - 1];

			let gradientHexes;
			if (space === "helmgen" || space === "helmgenlch") {
				gradientHexes = h.gradient(startHex, endHex, steps);
			}
			else {
				// For MetricSpace, use GenSpace gradient with arc-length reparam
				gradientHexes = h.gradient(startHex, endHex, steps);
			}

			// Build CSS gradient with percentage stops
			const cssStops = gradientHexes.map((hex, i) => {
				const pct = ((i / (steps - 1)) * 100).toFixed(1);
				return `${hex} ${pct}%`;
			}).join(", ");

			result = result.replace(fullGradient, `linear-gradient(${cssStops})`);
		}
	}

	return result;
}

// ── color-mix handling ─────────────────────────────────────────────

const COLOR_MIX_RE = /color-mix\s*\(\s*in\s+(helmlab|helmlch|helmgen|helmgenlch)\s*,/g;

function handleColorMix (value) {
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

		// Parse "color1 p1%, color2 p2%"
		const parts = argsStr.split(",").map((s) => s.trim());
		if (parts.length !== 2) {
			continue;
		}

		const parseStop = (s) => {
			const tokens = s.trim().split(/\s+/);
			let color = tokens[0];
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

		// Interpolate in the specified space
		const useGen = space === "helmgen" || space === "helmgenlch";
		const lab1 = useGen ? h.genFromHex(stop1.color) : h.fromHex(stop1.color);
		const lab2 = useGen ? h.genFromHex(stop2.color) : h.fromHex(stop2.color);

		// color-mix uses p1 as the fraction of color1
		const t = stop1.pct / 100;
		const mixed = [
			lab1[0] * t + lab2[0] * (1 - t),
			lab1[1] * t + lab2[1] * (1 - t),
			lab1[2] * t + lab2[2] * (1 - t),
		];

		const hex = useGen ? h.genToHex(mixed) : h.toHex(mixed);
		const [r, g, b] = hexToSrgb(hex);
		const ri = Math.round(r * 255);
		const gi = Math.round(g * 255);
		const bi = Math.round(b * 255);

		result = result.replace(fullMix, `rgb(${ri}, ${gi}, ${bi})`);
	}

	return result;
}

// ── PostCSS plugin ─────────────────────────────────────────────────

const plugin = (opts = {}) => {
	return {
		postcssPlugin: "postcss-helmlab",

		Declaration (decl) {
			const value = decl.value;

			// Skip if no helmlab functions present
			if (!value.includes("helm")) {
				return;
			}

			let newValue = value;

			// 1. Handle linear-gradient(in helmlab, ...)
			if (GRADIENT_RE.test(newValue)) {
				newValue = handleGradient(newValue);
			}

			// 2. Handle color-mix(in helmlab, ...)
			if (COLOR_MIX_RE.test(newValue)) {
				newValue = handleColorMix(newValue);
			}

			// 3. Handle helmlab(), helmlch(), helmgen(), helmgenlch()
			const fns = parseColorFn(newValue);
			for (const fn of fns) {
				const rgb = labToRgbString(fn.L, fn.a, fn.b, fn.alpha, fn.space);
				newValue = newValue.replace(fn.fullMatch, rgb);
			}

			if (newValue !== value) {
				decl.value = newValue;
			}
		},
	};
};

plugin.postcss = true;
export default plugin;
