from pathlib import Path

import gdown
from scipy.io import loadmat
import h5py


def download_from_google_drive(url: str, out: Path) -> None:
    """
    Download a file from Google Drive.

    Downloads the file from the provided Google Drive URL to the specified output path,
    unless the file already exists.

    Parameters
    ----------
    url : str
        Google Drive file URL or ID.
    out : pathlib.Path
        Path to save the downloaded file.

    Returns
    -------
    None

    Notes
    -----
    If the file already exists at the output path, downloading is skipped.
    Uses `gdown` for downloading.
    """
    if out.exists():
        print(f"{out} already exists, not downloading.")
        return
    print(f"Downloading {out} from Google Drive...")
    gdown.download(url, str(out), quiet=False)


def load_matlab_file(path: str | Path) -> dict | h5py.File:
    """
    Loads a MATLAB .mat file.

    Attempts to load the file using `scipy.io.loadmat`. If this fails (e.g. for
    MATLAB v7.3+ files), falls back to using `h5py.File` for compatibility.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the .mat file to load.

    Returns
    -------
    data : dict or h5py.File
        The loaded data. Returns a dictionary if loaded with SciPy, or an
        `h5py.File` object if loaded with h5py.

    Raises
    ------
    ValueError
        If the file cannot be loaded by either method.
    """
    path = str(path)

    # Try SciPy first
    try:
        data = loadmat(path)
        return data
    except Exception as e:
        print("SciPy loadmat failed:", repr(e))

    # Try h5py for v7.3+
    try:
        f = h5py.File(path, "r")
        return f
    except Exception as e:
        print("h5py.File failed:", repr(e))

    raise ValueError("File is not a valid MAT file")