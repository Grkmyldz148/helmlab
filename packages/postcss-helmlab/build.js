import { build } from "esbuild";

await build({
	entryPoints: ["index.js"],
	bundle: true,
	platform: "node",
	format: "cjs",
	outfile: "index.cjs",
	external: ["postcss", "helmlab"],
	footer: {
		js: "module.exports = module.exports.default; module.exports.default = module.exports; module.exports.postcss = true;",
	},
});
