import postcss from "postcss";
import plugin from "./index.js";

let pass = 0, fail = 0;

async function test (name, input, expected) {
	const result = await postcss([plugin]).process(input, { from: undefined });
	const output = result.css.trim();
	if (output === expected.trim()) {
		pass++;
	}
	else {
		fail++;
		console.log(`FAIL: ${name}`);
		console.log(`  expected: ${expected.trim()}`);
		console.log(`  got:      ${output}`);
	}
}

function rgb (str) {
	const m = str.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
	return m ? [+m[1], +m[2], +m[3]] : null;
}

async function run () {
	// 1. Basic helmlab() function
	await test(
		"helmlab() basic",
		"a { color: helmlab(0.78 0.52 -0.20) }",
		null, // We'll check it produces valid rgb()
	).catch(() => {});

	// Let's do assertion-based tests instead
	let result;

	// helmlab()
	result = await postcss([plugin]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	let hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb) { pass++; } else { fail++; console.log("FAIL: helmlab() → rgb()"); console.log("  got:", result.css); }

	// helmlch()
	result = await postcss([plugin]).process(
		"a { color: helmlch(0.78 0.56 338.7deg) }", { from: undefined },
	);
	hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb) { pass++; } else { fail++; console.log("FAIL: helmlch() → rgb()"); console.log("  got:", result.css); }

	// helmgen()
	result = await postcss([plugin]).process(
		"a { color: helmgen(0.60 0.22 0.03) }", { from: undefined },
	);
	hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb) { pass++; } else { fail++; console.log("FAIL: helmgen() → rgb()"); console.log("  got:", result.css); }

	// helmgenlch()
	result = await postcss([plugin]).process(
		"a { color: helmgenlch(0.60 0.15 30deg) }", { from: undefined },
	);
	hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb) { pass++; } else { fail++; console.log("FAIL: helmgenlch() → rgb()"); console.log("  got:", result.css); }

	// Alpha support
	result = await postcss([plugin]).process(
		"a { color: helmlab(0.78 0.52 -0.20 / 0.5) }", { from: undefined },
	);
	let hasRgba = /rgba\(\d+, \d+, \d+, 0\.5\)/.test(result.css);
	if (hasRgba) { pass++; } else { fail++; console.log("FAIL: helmlab() with alpha"); console.log("  got:", result.css); }

	// No-op on regular CSS
	result = await postcss([plugin]).process(
		"a { color: red; background: #fff }", { from: undefined },
	);
	if (result.css.trim() === "a { color: red; background: #fff }") {
		pass++;
	}
	else {
		fail++; console.log("FAIL: no-op on regular CSS");
	}

	// Multiple values in one declaration
	result = await postcss([plugin]).process(
		"a { border: 1px solid helmlab(0.5 0.1 0.1) }", { from: undefined },
	);
	hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb && result.css.includes("1px solid")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: mixed value"); console.log("  got:", result.css);
	}

	// linear-gradient(in helmgen, ...)
	result = await postcss([plugin]).process(
		"a { background: linear-gradient(in helmgen, #e63946, #457b9d) }", { from: undefined },
	);
	if (result.css.includes("linear-gradient(#") && !result.css.includes("in helmgen")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: linear-gradient"); console.log("  got:", result.css);
	}

	// color-mix(in helmgen, ...)
	result = await postcss([plugin]).process(
		"a { color: color-mix(in helmgen, #e63946 50%, #457b9d) }", { from: undefined },
	);
	hasRgb = /rgb\(\d+, \d+, \d+\)/.test(result.css);
	if (hasRgb && !result.css.includes("color-mix")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: color-mix"); console.log("  got:", result.css);
	}

	// Round-trip consistency: helmlab values should produce valid RGB
	result = await postcss([plugin]).process(
		"a { color: helmlab(1.14 0 0) }", { from: undefined },
	);
	const whiteRgb = rgb(result.css);
	if (whiteRgb && whiteRgb[0] >= 250 && whiteRgb[1] >= 250 && whiteRgb[2] >= 250) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: white round-trip"); console.log("  got:", result.css);
	}

	// Black
	result = await postcss([plugin]).process(
		"a { color: helmlab(0 0 0) }", { from: undefined },
	);
	const blackRgb = rgb(result.css);
	if (blackRgb && blackRgb[0] <= 5 && blackRgb[1] <= 5 && blackRgb[2] <= 5) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: black round-trip"); console.log("  got:", result.css);
	}

	// ── outputMode: 'both' (default) keeps sRGB inline + P3 inside @supports ──
	result = await postcss([plugin]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	const bothSrgb = /a\s*\{\s*color:\s*rgb\(\d+, \d+, \d+\)\s*\}/.test(result.css);
	const bothP3Supports = /@supports\s*\(color:\s*color\(display-p3 0 0 0\)\)\s*\{\s*a\s*\{\s*color:\s*color\(display-p3/.test(result.css);
	if (bothSrgb && bothP3Supports) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: both mode emits sRGB inline + @supports P3"); console.log("  got:", result.css);
	}

	// ── outputMode: 'srgb' only ──
	result = await postcss([plugin({ outputMode: "srgb" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	if (/rgb\(\d+, \d+, \d+\)/.test(result.css) && !result.css.includes("display-p3")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: srgb-only mode"); console.log("  got:", result.css);
	}

	// ── outputMode: 'p3' only ──
	result = await postcss([plugin({ outputMode: "p3" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	if (/color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)/.test(result.css) && !/\brgb\(/.test(result.css)) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: p3-only mode"); console.log("  got:", result.css);
	}

	// ── P3 with alpha ──
	result = await postcss([plugin({ outputMode: "p3" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20 / 0.5) }", { from: undefined },
	);
	if (/color\(display-p3 [\d.]+ [\d.]+ [\d.]+ \/ 0\.5\)/.test(result.css)) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: P3 with alpha"); console.log("  got:", result.css);
	}

	// ── P3 helmgen() (sRGB-bound, just re-encoded) ──
	result = await postcss([plugin({ outputMode: "p3" })]).process(
		"a { color: helmgen(0.60 0.22 0.03) }", { from: undefined },
	);
	if (/color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)/.test(result.css)) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: P3 helmgen"); console.log("  got:", result.css);
	}

	// ── P3 gradient ──
	result = await postcss([plugin({ outputMode: "p3" })]).process(
		"a { background: linear-gradient(in helmgen, #e63946, #457b9d) }", { from: undefined },
	);
	const p3GradientStops = (result.css.match(/color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)/g) || []).length;
	if (p3GradientStops >= 2 && !result.css.includes("in helmgen")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: P3 gradient"); console.log("  got:", result.css);
	}

	// ── P3 color-mix ──
	result = await postcss([plugin({ outputMode: "p3" })]).process(
		"a { color: color-mix(in helmgen, #e63946 50%, #457b9d) }", { from: undefined },
	);
	if (/color\(display-p3 [\d.]+ [\d.]+ [\d.]+\)/.test(result.css) && !result.css.includes("color-mix")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: P3 color-mix"); console.log("  got:", result.css);
	}

	// ── 'both' mode produces 1 inline + 1 @supports (P3) ──
	result = await postcss([plugin]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	{
		const supportsCount = (result.css.match(/@supports/g) || []).length;
		// Count `color:` only as a property declaration (preceded by `{` or `;` or whitespace
		// after the selector's `{`), not as part of `@supports (color: ...)` parameters.
		const colorDeclCount = (result.css.match(/[{\s]color:\s/g) || []).length;
		if (supportsCount === 1 && colorDeclCount === 2) {
			pass++;
		}
		else {
			fail++; console.log("FAIL: both mode → 1 inline + 1 @supports"); console.log("  got:", result.css);
		}
	}

	// ── 'both' mode is a no-op when no helmlab fns present ──
	result = await postcss([plugin]).process(
		"a { color: red }", { from: undefined },
	);
	if (result.css.trim() === "a { color: red }") {
		pass++;
	}
	else {
		fail++; console.log("FAIL: both mode no-op on plain CSS"); console.log("  got:", result.css);
	}

	// ── outputMode: 'rec2020' only ──
	result = await postcss([plugin({ outputMode: "rec2020" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	if (/color\(rec2020 [\d.]+ [\d.]+ [\d.]+\)/.test(result.css)
		&& !/\brgb\(/.test(result.css)
		&& !result.css.includes("display-p3")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: rec2020-only mode"); console.log("  got:", result.css);
	}

	// ── Rec2020 with alpha ──
	result = await postcss([plugin({ outputMode: "rec2020" })]).process(
		"a { color: helmlch(0.5 0.2 200deg / 0.8) }", { from: undefined },
	);
	if (/color\(rec2020 [\d.]+ [\d.]+ [\d.]+ \/ 0\.8\)/.test(result.css)) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: Rec2020 with alpha"); console.log("  got:", result.css);
	}

	// ── outputMode: 'all' inlines sRGB and emits @supports for P3 + Rec2020 ──
	result = await postcss([plugin({ outputMode: "all" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	{
		const idxSrgb = result.css.search(/\brgb\(/);
		const idxP3At = result.css.search(/@supports\s*\(color:\s*color\(display-p3/);
		const idxRecAt = result.css.search(/@supports\s*\(color:\s*color\(rec2020/);
		const ok = idxSrgb >= 0 && idxP3At > idxSrgb && idxRecAt > idxP3At;
		if (ok) {
			pass++;
		}
		else {
			fail++; console.log("FAIL: 'all' mode cascade order"); console.log("  got:", result.css);
		}
	}

	// ── 'all' mode emits 2 @supports blocks ──
	result = await postcss([plugin({ outputMode: "all" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	{
		const supportsCount = (result.css.match(/@supports/g) || []).length;
		if (supportsCount === 2) {
			pass++;
		}
		else {
			fail++; console.log("FAIL: 'all' emits 2 @supports blocks"); console.log("  got:", result.css);
		}
	}

	// ── Rec2020 gradient ──
	result = await postcss([plugin({ outputMode: "rec2020" })]).process(
		"a { background: linear-gradient(in helmlab, #e63946, #457b9d) }", { from: undefined },
	);
	const recGradientStops = (result.css.match(/color\(rec2020 [\d.]+ [\d.]+ [\d.]+\)/g) || []).length;
	if (recGradientStops >= 2 && !result.css.includes("in helmlab")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: Rec2020 gradient"); console.log("  got:", result.css);
	}

	// ── Rec2020 color-mix ──
	result = await postcss([plugin({ outputMode: "rec2020" })]).process(
		"a { color: color-mix(in helmlab, #e63946 50%, #457b9d) }", { from: undefined },
	);
	if (/color\(rec2020 [\d.]+ [\d.]+ [\d.]+\)/.test(result.css) && !result.css.includes("color-mix")) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: Rec2020 color-mix"); console.log("  got:", result.css);
	}

	// ── Invalid outputMode falls back to 'both' ──
	result = await postcss([plugin({ outputMode: "bogus" })]).process(
		"a { color: helmlab(0.78 0.52 -0.20) }", { from: undefined },
	);
	if (/\brgb\(/.test(result.css) && /display-p3/.test(result.css) && !/rec2020/.test(result.css)) {
		pass++;
	}
	else {
		fail++; console.log("FAIL: invalid outputMode → both fallback"); console.log("  got:", result.css);
	}

	console.log(`\n${pass} passed, ${fail} failed out of ${pass + fail} tests`);
	if (fail > 0) {
		process.exit(1);
	}
}

run();
