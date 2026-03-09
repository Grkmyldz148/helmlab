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

	console.log(`\n${pass} passed, ${fail} failed out of ${pass + fail} tests`);
	if (fail > 0) {
		process.exit(1);
	}
}

run();
