import numpy as np


def corr(x, y, keep=None) -> float:
    """Mean squared difference of the feature-feature Pearson correlation matrices.

    ``keep``: indices of the features to use (e.g. the non-constant pixels of the data);
    a constant feature has an undefined correlation, which is counted as 0."""
    x, y = np.asarray(x), np.asarray(y)
    if keep is not None:
        x, y = x[:, keep], y[:, keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        cx = np.nan_to_num(np.corrcoef(x, rowvar=False))
        cy = np.nan_to_num(np.corrcoef(y, rowvar=False))
    return float(np.mean((cx - cy) ** 2))
