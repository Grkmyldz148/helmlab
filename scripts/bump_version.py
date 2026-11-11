#!/usr/bin/env python3
"""Single-command version propagation for the helmlab monorepo.

Usage:  python scripts/bump_version.py 0.13.0
        python scripts/bump_version.py --check        # verify everything matches

The canonical Python version lives in pyproject.toml. This tool writes the same
version into every place that needs a literal copy, so they can never drift:

  - pyproject.toml                       (Python package — canonical)
  - src/helmlab/__init__.py              (dev fallback literal)
  - packages/helmlab-js/package.json     (npm package)

postcss-helmlab is versioned independently and is intentionally NOT touched here.
CI (.github/workflows/deploy.yml) runs `--check` and fails the build on any drift.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src" / "helmlab" / "__init__.py"
JS_PKG = ROOT / "packages" / "helmlab-js" / "package.json"

SEMVER = re.compile(r"^\d+\.\d+\.\d+([-+].+)?$")


def read_current() -> dict:
    """Read the version each file currently declares (None if unreadable)."""
    out = {}
    # pyproject
    m = re.search(r'(?m)^version\s*=\s*"([^"]+)"', PYPROJECT.read_text())
    out["pyproject.toml"] = m.group(1) if m else None
    # __init__ fallback literal
    m = re.search(r'__version__\s*=\s*"([^"]+)"\s*#\s*bump-version-fallback', INIT.read_text())
    out["src/helmlab/__init__.py"] = m.group(1) if m else None
    # js package.json
    try:
        out["packages/helmlab-js/package.json"] = json.loads(JS_PKG.read_text()).get("version")
    except Exception:
        out["packages/helmlab-js/package.json"] = None
    return out


def write_version(v: str) -> None:
    # pyproject.toml
    t = PYPROJECT.read_text()
    t = re.sub(r'(?m)^(version\s*=\s*")[^"]+(")', rf"\g<1>{v}\g<2>", t, count=1)
    PYPROJECT.write_text(t)
    # __init__ fallback
    t = INIT.read_text()
    t = re.sub(r'(__version__\s*=\s*")[^"]+("\s*#\s*bump-version-fallback)', rf"\g<1>{v}\g<2>", t, count=1)
    INIT.write_text(t)
    # js package.json (preserve formatting/indent=2)
    d = json.loads(JS_PKG.read_text())
    d["version"] = v
    JS_PKG.write_text(json.dumps(d, indent=2) + "\n")


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--check":
        cur = read_current()
        vals = {k: v for k, v in cur.items() if v is not None}
        unique = set(vals.values())
        for k, v in cur.items():
            print(f"  {v or 'MISSING':<12} {k}")
        if len(unique) == 1 and None not in cur.values():
            print(f"OK — all in sync at {unique.pop()}")
            return 0
        print(f"DRIFT — versions disagree: {sorted(unique)}")
        return 1
    v = args[0].lstrip("v")
    if not SEMVER.match(v):
        print(f"error: '{v}' is not semver (x.y.z)")
        return 2
    write_version(v)
    print(f"set version → {v} in:")
    for k in read_current():
        print(f"  - {k}")
    print("\nremember: postcss-helmlab is versioned separately (not touched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
