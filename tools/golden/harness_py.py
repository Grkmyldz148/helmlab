"""Golden-output harness for helmlab Python. Same keys and inputs as harness_js.mjs.
Usage: python3 -I harness_py.py <package dir> <inputs.json> <out.json> [--quick]
<package dir> holds the `helmlab` package to test (for example an unpacked wheel) and goes
first on sys.path. <inputs.json> comes from `harness_js.mjs --inputs`.
--quick runs a subset (about every 7th input) as a smoke test.
"""
import json, math, platform, sys, warnings

if len(sys.argv) < 4:
    sys.exit(__doc__)
pkg_dir, in_path, out_path = sys.argv[1:4]
sys.path.insert(0, pkg_dir)

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")
import helmlab  # noqa: E402
from helmlab import Helmlab  # noqa: E402

inp = json.load(open(in_path))
QUICK = "--quick" in sys.argv
hl = Helmlab()
G, M, T = hl.gen, hl.metric, hl.tokens
GAMUTS = ["srgb", "display-p3", "rec2020"]
out = {}


def conv(x):
    if isinstance(x, np.ndarray):
        return conv(x.tolist())
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else f"__{'NaN' if math.isnan(x) else ('Infinity' if x > 0 else '-Infinity')}"
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {k: conv(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [conv(v) for v in x]
    return x


def rec(key, fn):
    try:
        out[key] = conv(fn())
    except Exception as e:  # noqa: BLE001
        out[key] = {"__error": str(e)[:200]}


hexes, wide, xyzs = inp["hexes"], inp["wide"], inp["xyzs"]
genLabs, metLabs = inp["genLabs"], inp["metLabs"]
pairs, wpairs, bases, special = inp["pairs"], inp["wpairs"], inp["bases"], inp["special"]
step = 7 if QUICK else 1

for i, h in enumerate(hexes[::step]):
    rec(f"hex.gen.fromHex:{h}", lambda: G.from_hex(h))
    rec(f"hex.gen.rt:{h}", lambda: G.to_hex(G.from_hex(h)))
    rec(f"hex.gen.toLch:{h}", lambda: G.to_lch(G.from_hex(h)))
    rec(f"hex.metric.fromHex:{h}", lambda: M.from_hex(h))
    rec(f"hex.metric.rt:{h}", lambda: M.to_hex(M.from_hex(h)))
    rec(f"hex.metric.info:{h}", lambda: M.info(h))
for c in wide[::step]:
    rec(f"wide.gen.fromHex:{c}", lambda: G.from_hex(c))
    rec(f"wide.metric.fromHex:{c}", lambda: M.from_hex(c))
    rec(f"wide.metric.info:{c}", lambda: M.info(c))
    for g in GAMUTS:
        rec(f"wide.gen.toCss.{g}:{c}", lambda: G.to_css(G.from_hex(c), g))
        rec(f"wide.metric.toCss.{g}:{c}", lambda: M.to_css(M.from_hex(c), g))
        rec(f"wide.gen.inGamut.{g}:{c}", lambda: G.in_gamut(c, g))
        rec(f"wide.metric.inGamut.{g}:{c}", lambda: M.in_gamut(c, g))
    rec(f"wide.gen.toHex:{c}", lambda: G.to_hex(G.from_hex(c)))
    rec(f"wide.metric.toHex:{c}", lambda: M.to_hex(M.from_hex(c)))
for i, x in enumerate(xyzs):
    if i % step:
        continue
    rec(f"xyz.metric.fromXyz:{i}", lambda: M.from_xyz(x))
    rec(f"xyz.metric.rt:{i}", lambda: M.to_xyz(M.from_xyz(x)))
for i, l in enumerate(genLabs):
    if i % step:
        continue
    L = G.lab(*l)
    rec(f"glab.toSrgb:{i}", lambda: G.to_srgb(L))
    rec(f"glab.toHex:{i}", lambda: G.to_hex(L))
    rec(f"glab.toLch:{i}", lambda: G.to_lch(L))
    rec(f"glab.lchrt:{i}", lambda: G.from_lch(G.to_lch(L)))
    for g in GAMUTS:
        rec(f"glab.toCss.{g}:{i}", lambda: G.to_css(L, g))
        rec(f"glab.inGamut.{g}:{i}", lambda: G.in_gamut(L, g))
        if i % 2 == 0:
            for m in ["chroma", "adaptive"]:
                rec(f"glab.gamutMap.{g}.{m}:{i}", lambda: G.gamut_map(L, g, m))
for i, l in enumerate(metLabs):
    if i % step:
        continue
    L = M.lab(*l)
    rec(f"mlab.toSrgb:{i}", lambda: M.to_srgb(L))
    rec(f"mlab.toHex:{i}", lambda: M.to_hex(L))
    rec(f"mlab.toXyz:{i}", lambda: M.to_xyz(L))
    rec(f"mlab.toLch:{i}", lambda: M.to_lch(L))
    rec(f"mlab.lchrt:{i}", lambda: M.from_lch(M.to_lch(L)))
    for g in GAMUTS:
        rec(f"mlab.toCss.{g}:{i}", lambda: M.to_css(L, g))
        rec(f"mlab.inGamut.{g}:{i}", lambda: M.in_gamut(L, g))


def jsnum(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))


for g in GAMUTS:
    for k in range(0, 720, step):
        rec(f"cusp.{g}:{jsnum(k / 2)}", lambda: G.cusp(k / 2, g))
    for Li in range(1, 20):
        for h in range(0, 360, 5 * step):
            rec(f"maxChroma.{g}:{Li}:{h}", lambda: G.max_chroma(Li * 0.05, h, g))
for b in bases:
    for g in GAMUTS:
        rec(f"vivid.{g}:{b}", lambda: G.vivid(b, gamut=g))

for i, (a, b) in enumerate(pairs[::step]):
    for g in GAMUTS:
        for n in [0, 1, 2, 5, 16]:
            if g != "srgb" and i % 3 != 0:
                continue
            rec(f"gradient.{g}.{n}:{a}>{b}", lambda: G.gradient(a, b, n, gamut=g))
    for t in [0, 0.25, 0.5, 0.9, 1, -0.5, 1.5]:
        rec(f"mix.{jsnum(t)}:{a}>{b}", lambda: G.mix(a, b, t))
for a, b in wpairs[:60:step]:
    for g in GAMUTS:
        rec(f"wgradient.{g}:{a}>{b}", lambda: G.gradient(a, b, 9, gamut=g))
for b in bases[::step]:
    for g in GAMUTS:
        rec(f"palette.{g}:{b}", lambda: G.palette(b, 10, gamut=g))
        rec(f"scale.{g}:{b}", lambda: G.scale(b, None, gamut=g))
    rec(f"scale.custom:{b}", lambda: G.scale(b, [100, 500, 900]))
    for k in ["complementary", "analogous", "triadic", "tetradic", "split_complementary"]:
        rec(f"harm.{k}:{b}", lambda: G.harmonies(b, k))
    for d in [30, 90, 180, -45, 360, 720.5]:
        rec(f"rot.{jsnum(d)}:{b}", lambda: G.rotate_hue(b, d))
for g in GAMUTS:
    for n in [1, 6, 12]:
        for Lx, Cx in [(0.6, 0.15), (0.3, 0.05), (0.9, 0.3)]:
            rec(f"hueRing.{g}.{n}.{jsnum(Lx)}.{jsnum(Cx)}", lambda: G.hue_ring(n, lightness=Lx, chroma=Cx, gamut=g))

for a, b in pairs[::step]:
    rec(f"cr:{a}>{b}", lambda: G.contrast_ratio(a, b))
    rec(f"meets.AA:{a}>{b}", lambda: G.meets_contrast(a, b, "AA"))
    rec(f"meets.AAA:{a}>{b}", lambda: G.meets_contrast(a, b, "AAA"))
    for r in [3, 4.5, 7]:
        rec(f"ensure.{jsnum(r)}:{a}>{b}", lambda: G.ensure_contrast(a, b, r))
    rec(f"ensure.strict:{a}>{b}", lambda: G.ensure_contrast(a, b, 7, strict=True))
    rec(f"mode.l2d:{a}", lambda: G.adapt_to_mode(a, "light", "dark"))
    rec(f"mode.d2l:{a}", lambda: G.adapt_to_mode(a, "dark", "light"))
    rec(f"pair:{a}>{b}", lambda: G.adapt_pair(a, b, "light", "dark"))

for a, b in (pairs + wpairs)[::step]:
    rec(f"diff:{a}>{b}", lambda: M.difference(a, b))
    rec(f"eucl:{a}>{b}", lambda: M.euclidean(a, b))
    rec(f"de00:{a}>{b}", lambda: M.ciede2000(a, b))
    rec(f"jnd:{a}>{b}", lambda: M.jnd(a, b))
    rec(f"dist:{a}>{b}", lambda: M.distance(M.from_hex(a), M.from_hex(b)))
    rec(f"conf:{a}>{b}", lambda: M.confidence(a, b))
pal = special[:10]
for t in (bases + wide[:40])[::step]:
    for m in ["ciede2000", "difference", "euclidean"]:
        rec(f"nearest.{m}:{t}", lambda: M.nearest(t, pal, m))

for i, h in enumerate(hexes):
    if i % 3 or (i // 3) % step:
        continue
    for f in ["hex", "rgb", "hsl", "oklch", "p3", "rec2020"]:
        rec(f"hex.tokens.css.{f}:{h}", lambda: T.css(h, f))
    rec(f"hex.tokens.android:{h}", lambda: T.android(h))
    rec(f"hex.tokens.swift:{h}", lambda: T.swift(h))
    rec(f"hex.tokens.iosP3:{h}", lambda: T.ios_p3(h))
for i, b in enumerate(bases[:30]):
    sc = G.scale(b)
    rec(f"tok.vars:{b}", lambda: T.css_variables(sc))
    rec(f"tok.tw:{b}", lambda: T.tailwind(sc, f"c{i}"))
    rec(f"tok.multi:{b}", lambda: T.multi_format(sc, f"c{i}"))
    rec(f"tok.json:{b}", lambda: T.json({f"c{i}": sc}))

n_keys = len(out)
n_err = sum(1 for v in out.values() if isinstance(v, dict) and "__error" in v)
out["__meta"] = {
    "harness": "py",
    "helmlab": getattr(helmlab, "__version__", None),
    "module": helmlab.__file__,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "platform": platform.platform(),
    "quick": QUICK,
}
json.dump(out, open(out_path, "w"))
print(f"keys={n_keys} errors={n_err} helmlab={out['__meta']['helmlab']} python={platform.python_version()}")
