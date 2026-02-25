"""Shared test fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_XYZ():
    """Small array of valid XYZ values."""
    return np.array([
        [0.95047, 1.0, 1.08883],  # D65 white
        [0.5, 0.5, 0.5],
        [0.2, 0.1, 0.3],
        [0.05, 0.05, 0.05],
        [0.0, 0.0, 0.0],          # black
    ], dtype=np.float64)


@pytest.fixture
def sample_Lab():
    """Small array of CIE Lab values."""
    return np.array([
        [100.0, 0.0, 0.0],      # white
        [50.0, 30.0, -40.0],
        [75.0, -20.0, 60.0],
        [25.0, 50.0, 10.0],
        [0.0, 0.0, 0.0],         # black
    ], dtype=np.float64)


@pytest.fixture
def random_XYZ():
    """1000 random XYZ values in [0.01, 1.0]."""
    rng = np.random.default_rng(42)
    return rng.uniform(0.01, 1.0, size=(1000, 3))
