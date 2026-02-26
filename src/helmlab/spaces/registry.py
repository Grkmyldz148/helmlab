"""Color space registry: get_space("oklch") → instance."""

from helmlab.spaces.base import ColorSpace
from helmlab.spaces.srgb import SRGB
from helmlab.spaces.oklch import OKLCH
from helmlab.spaces.cam16ucs import CAM16UCS
from helmlab.spaces.ipt import IPT
from helmlab.spaces.jzczhz import JzAzBz
from helmlab.spaces.analytical import AnalyticalSpace

_REGISTRY: dict[str, type[ColorSpace]] = {
    "srgb": SRGB,
    "oklch": OKLCH,
    "oklab": OKLCH,
    "cam16ucs": CAM16UCS,
    "cam16-ucs": CAM16UCS,
    "ipt": IPT,
    "jzazbz": JzAzBz,
    "jzczhz": JzAzBz,
    "analytical": AnalyticalSpace,
    "helmlab": AnalyticalSpace,
}

# Singleton cache
_INSTANCES: dict[str, ColorSpace] = {}


def get_space(name: str) -> ColorSpace:
    """Get a color space instance by name.

    Parameters
    ----------
    name : str
        Color space name (case-insensitive). Supported:
        "srgb", "oklch"/"oklab", "cam16ucs"/"cam16-ucs", "ipt", "jzazbz"/"jzczhz"

    Returns
    -------
    ColorSpace
    """
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = sorted(set(cls.name for cls in _REGISTRY.values()))
        raise ValueError(f"Unknown color space '{name}'. Available: {available}")

    cls = _REGISTRY[key]
    # Share instance across aliases for the same class
    canonical = cls.__name__
    if canonical not in _INSTANCES:
        _INSTANCES[canonical] = cls()
    return _INSTANCES[canonical]


def all_spaces() -> list[ColorSpace]:
    """Return one instance of each unique color space."""
    seen = set()
    result = []
    for cls in _REGISTRY.values():
        if cls.name not in seen:
            seen.add(cls.name)
            result.append(get_space(cls.name.lower().replace("-", "")))
    return result
