"""Compare two golden-output JSONs. Usage: python3 compare.py A.json B.json [--tol 0] [--detail prefix] [--dump file]
Numbers: absolute difference (NaN==NaN); the default tolerance is 0, so any change counts.
Strings/bools/null: exact. Errors: message equality.
Reports per category (key prefix before ':'): count, exact, <=tol, worst diff, worst key.
Exit code 1 if any key differs beyond the tolerance or exists in only one file."""
import json, math, sys, collections

a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
meta_a, meta_b = a.pop("__meta", None), b.pop("__meta", None)
print("A", sys.argv[1], json.dumps(meta_a))
print("B", sys.argv[2], json.dumps(meta_b))
tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 0.0
detail = sys.argv[sys.argv.index("--detail") + 1] if "--detail" in sys.argv else None
dump = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else None


def diff(x, y):
    """Return (max numeric diff, n_nonnumeric_mismatch)."""
    if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None:
        return (0.0, 0) if x == y else (0.0, 1)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        if math.isnan(x) and math.isnan(y):
            return 0.0, 0
        return abs(x - y), 0
    if isinstance(x, str) or isinstance(y, str):
        return (0.0, 0) if x == y else (0.0, 1)
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            return 0.0, 1
        m, n = 0.0, 0
        for k in x:
            d, c = diff(x[k], y[k]); m = max(m, d); n += c
        return m, n
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return 0.0, 1
        m, n = 0.0, 0
        for u, v in zip(x, y):
            d, c = diff(u, v); m = max(m, d); n += c
        return m, n
    return 0.0, 1


common = sorted(set(a) & set(b))
only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
cat = collections.defaultdict(lambda: dict(n=0, exact=0, tol=0, nonnum=0, worst=0.0, wkey="", nkey=""))
bad = []
for k in common:
    c = cat[k.split(":")[0]]
    c["n"] += 1
    d, nn = diff(a[k], b[k])
    if nn:
        c["nonnum"] += 1; c["nkey"] = c["nkey"] or k; bad.append((k, "nonnum", d))
    elif d == 0:
        c["exact"] += 1
    elif d <= tol:
        c["tol"] += 1
    else:
        bad.append((k, "num", d))
    if d > c["worst"]:
        c["worst"], c["wkey"] = d, k
print(f"common keys {len(common)}  only-in-A {only_a}  only-in-B {only_b}  tol {tol:g}")
print(f"{'category':34s} {'n':>6s} {'exact':>6s} {'<=tol':>6s} {'>tol':>5s} {'str/bool':>8s}  worst")
for name in sorted(cat):
    c = cat[name]
    over = c["n"] - c["exact"] - c["tol"] - c["nonnum"]
    flag = "" if over == 0 and c["nonnum"] == 0 else "  <<<"
    print(f"{name:34s} {c['n']:6d} {c['exact']:6d} {c['tol']:6d} {over:5d} {c['nonnum']:8d}  {c['worst']:.2e}{flag}")
tot_over = sum(1 for x in bad)
print(f"TOTAL keys beyond tol or mismatched: {tot_over}")
if detail:
    for k, kind, d in bad:
        if k.startswith(detail):
            print(" ", kind, f"{d:.3e}", k, "|", json.dumps(a[k])[:160], "->", json.dumps(b[k])[:160])
if dump:
    json.dump([k for k, _, _ in bad], open(dump, "w"))
sys.exit(1 if (bad or only_a or only_b) else 0)
