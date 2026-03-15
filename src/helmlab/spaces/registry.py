"""Color space registry: get_space("oklch") → instance."""

from helmlab.spaces.base import ColorSpace
from helmlab.spaces.metric import MetricSpace
from helmlab.spaces.gen import GenSpace


def _lazy_import(module_path: str, class_name: str):
    """Return a factory that lazily imports a class."""
    def factory():
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    return factory


# Eagerly available (no external deps beyond numpy/scipy)
_EAGER: dict[str, type[ColorSpace]] = {
    "analytical": MetricSpace,
    "helmlab": MetricSpace,
    "metric": MetricSpace,
    "gen": GenSpace,
}

# Lazily imported (require colour-science)
_LAZY: dict[str, tuple[str, str]] = {
    "srgb": ("helmlab.spaces.srgb", "SRGB"),
    "oklch": ("helmlab.spaces.oklch", "OKLCH"),
    "oklab": ("helmlab.spaces.oklch", "OKLCH"),
    "cam16ucs": ("helmlab.spaces.cam16ucs", "CAM16UCS"),
    "cam16-ucs": ("helmlab.spaces.cam16ucs", "CAM16UCS"),
    "ipt": ("helmlab.spaces.ipt", "IPT"),
    "jzazbz": ("helmlab.spaces.jzczhz", "JzAzBz"),
    "jzczhz": ("helmlab.spaces.jzczhz", "JzAzBz"),
}

# Singleton cache
_INSTANCES: dict[str, ColorSpace] = {}


def _resolve(key: str) -> type[ColorSpace]:
    """Resolve a key to a class, using lazy import if needed."""
    if key in _EAGER:
        return _EAGER[key]
    if key in _LAZY:
        mod_path, cls_name = _LAZY[key]
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, cls_name)
    available = sorted(set(
        list(_EAGER.keys()) + list(_LAZY.keys())
    ))
    raise ValueError(f"Unknown color space '{key}'. Available: {available}")


def get_space(name: str) -> ColorSpace:
    """Get a color space instance by name.

    Parameters
    ----------
    name : str
        Color space name (case-insensitive). Supported:
        "srgb", "oklch"/"oklab", "cam16ucs"/"cam16-ucs", "ipt", "jzazbz"/"jzczhz",
        "metric"/"helmlab"/"analytical", "gen"

    Returns
    -------
    ColorSpace
    """
    key = name.lower().strip()
    cls = _resolve(key)
    canonical = cls.__name__
    if canonical not in _INSTANCES:
        _INSTANCES[canonical] = cls()
    return _INSTANCES[canonical]


def all_spaces() -> list[ColorSpace]:
    """Return one instance of each unique color space."""
    seen = set()
    result = []
    all_keys = list(_EAGER.keys()) + list(_LAZY.keys())
    for key in all_keys:
        try:
            cls = _resolve(key)
        except (ImportError, ModuleNotFoundError):
            continue
        if cls.__name__ not in seen:
            seen.add(cls.__name__)
            result.append(get_space(key))
    return result
