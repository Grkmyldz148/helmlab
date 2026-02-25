"""File I/O utilities: download, cache, parse."""

from pathlib import Path
from urllib.request import urlretrieve
import hashlib

import pandas as pd
from tqdm import tqdm

from colorspace.config import DATA_DIR


def ensure_data_dir() -> Path:
    """Create and return the data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


class _DownloadProgress(tqdm):
    """tqdm wrapper for urlretrieve reporthook."""

    def update_to(self, blocks=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def download_file(url: str, filename: str, *, md5: str | None = None) -> Path:
    """Download a file to data/ if it doesn't already exist.

    Parameters
    ----------
    url : str
        URL to download from.
    filename : str
        Local filename (stored in DATA_DIR).
    md5 : str, optional
        Expected MD5 hash for verification.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest = ensure_data_dir() / filename
    if dest.exists():
        if md5 is not None:
            actual = hashlib.md5(dest.read_bytes()).hexdigest()
            if actual == md5:
                return dest
        else:
            return dest

    with _DownloadProgress(unit="B", unit_scale=True, desc=filename) as t:
        urlretrieve(url, dest, reporthook=t.update_to)

    if md5 is not None:
        actual = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual != md5:
            dest.unlink()
            raise ValueError(
                f"MD5 mismatch for {filename}: expected {md5}, got {actual}"
            )
    return dest


def load_xlsx(path: Path | str, **kwargs) -> pd.DataFrame:
    """Load an Excel file into a DataFrame."""
    return pd.read_excel(path, engine="openpyxl", **kwargs)
