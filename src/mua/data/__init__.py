"""Dataset loaders. Paths can be overridden with the environment variables
MUA_CLIC_DIR (folder with clic_<n>px_{train,val,test}.h5) and MUA_MNIST (mnist.npz)."""
import os
from pathlib import Path

import numpy as np

CLIC_DIR = Path(os.environ.get("MUA_CLIC_DIR", Path.home() / "PSG" / "src" / "PSG" / "data"))
MNIST_FILE = Path(os.environ.get("MUA_MNIST", Path.home() / ".keras" / "datasets" / "mnist.npz"))


def load_clic(pixels: int, split: str = "train", folder: Path = CLIC_DIR) -> np.ndarray:
    """CLIC ECAL showers downsampled to `pixels` 1D pixels (desyqml/clic), shape (N, pixels)."""
    import h5py

    with h5py.File(Path(folder) / f"clic_{pixels}px_{split}.h5") as f:
        return f["ECAL"][:].astype(float)


def load_mnist(split: str = "train", path: Path = MNIST_FILE):
    """MNIST grey levels in [0, 1], flattened to (N, 784), and the labels."""
    with np.load(path) as f:
        x = f[f"x_{split}"].reshape(len(f[f"x_{split}"]), -1) / 255.0
        return x, f[f"y_{split}"]
