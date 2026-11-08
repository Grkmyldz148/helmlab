# Golden outputs

Helmlab 2026 is defined by the outputs of helmlab 1.0.0. These scripts check that a release gives the same outputs as the release before it.

| File | What it does |
|---|---|
| `harness_js.mjs` | Calls every public method of the JavaScript API on a fixed, seeded input set and writes each output, or its error message, to a JSON file. |
| `harness_py.py` | Makes the same calls with the same inputs in the Python package. |
| `compare.py` | Compares two output files key by key. The default tolerance is 0. Exit code 1 if anything differs. |
| `exhaustive.mjs` | Compares two JavaScript builds on every 8-bit sRGB colour, 64³ Display P3 and Rec.2020 grids, and 3600 cusp hues per gamut. Exit code 1 if anything differs. |

## Running a comparison

Download the two versions into an empty directory outside the repository:

```sh
W=/tmp/golden
for v in 1.0.0 1.0.1; do
  mkdir -p $W/npm-$v $W/py-$v
  (cd $W/npm-$v && npm pack helmlab@$v && tar -xzf helmlab-$v.tgz)
  python3 -m pip download --no-deps --only-binary=:all: helmlab==$v -d $W/py-$v
  (cd $W/py-$v && unzip -q helmlab-$v-*.whl)
done
```

JavaScript:

```sh
node tools/golden/harness_js.mjs $W/npm-1.0.0/package/dist/index.js $W/js-1.0.0.json --inputs $W/inputs.json
node tools/golden/harness_js.mjs $W/npm-1.0.1/package/dist/index.js $W/js-1.0.1.json
python3 tools/golden/compare.py $W/js-1.0.0.json $W/js-1.0.1.json
node tools/golden/exhaustive.mjs $W/npm-1.0.0/package/dist/index.js $W/npm-1.0.1/package/dist/index.js
```

Python (needs numpy; the inputs come from the JavaScript run):

```sh
python3 -I tools/golden/harness_py.py $W/py-1.0.0 $W/inputs.json $W/py-1.0.0.json
python3 -I tools/golden/harness_py.py $W/py-1.0.1 $W/inputs.json $W/py-1.0.1.json
python3 tools/golden/compare.py $W/py-1.0.0.json $W/py-1.0.1.json
```

## Notes

- Compare outputs made with the same Node.js version. ECMA-262 leaves the accuracy of `Math.cbrt`, `Math.exp` and `Math.pow` to the engine, so another engine can change the last bits. Each output file records the package and engine versions under `__meta`.
- Some calls end in an error by design, for example `ensureContrast(a, b, 7, { strict: true })` when no colour reaches the ratio. The error message is part of the output and has to match as well.
- The Python run takes much longer than the JavaScript run. `--quick` runs about one input in seven as a smoke test.
- The output files are large (about 18 MB each) and are not stored in the repository.
