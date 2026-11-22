"""Preprocess the 10% KDD Cup 1999 data set for the CSE558 assignment.

The resulting matrix contains only finite float32 values, has no constant
columns, and has every feature rescaled independently to the interval [0, 1].
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_kddcup99


CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = (
    "protocol_type",
    "service",
    "flag",
)


def preprocess_kdd_features(data: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot encode, de-constant, and min--max scale KDD feature rows.

    Parameters
    ----------
    data:
        The ``kdd.data`` DataFrame returned by ``fetch_kddcup99``.

    Returns
    -------
    D, feature_names:
        ``D`` is an ``(n, d)`` float32 array suitable for the assignment's
        k-means and regression tasks.  ``feature_names`` corresponds to its
        columns.  Byte-valued categories (for example ``b'tcp'``) are handled
        directly by pandas.
    """
    missing = set(CATEGORICAL_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"KDD data is missing categorical columns: {sorted(missing)}")

    # Get float32 dummy columns immediately: the full data set is large, and
    # pandas otherwise defaults to a wider integer type for these columns.
    categorical = pd.get_dummies(
        data.loc[:, CATEGORICAL_COLUMNS], dtype=np.float32
    )
    numeric_columns = data.columns.drop(list(CATEGORICAL_COLUMNS))
    numeric = data.loc[:, numeric_columns].astype(np.float32)
    features = pd.concat((numeric, categorical), axis=1, copy=False)

    # Include NaN in the count so a malformed all-NaN feature is not retained.
    nonconstant = features.nunique(dropna=False) > 1
    features = features.loc[:, nonconstant]
    if features.empty:
        raise ValueError("No non-constant KDD features remain after preprocessing")
    if features.isna().to_numpy().any():
        raise ValueError("KDD features contain NaN values; cannot min--max scale them")

    minima = features.min(axis=0)
    ranges = features.max(axis=0) - minima
    # The preceding filter guarantees positive ranges; this check keeps the
    # invariant explicit if the input DataFrame changes in the future.
    if (ranges <= 0).any():
        raise RuntimeError("A constant feature remained after filtering")

    scaled = (features - minima) / ranges
    D = scaled.to_numpy(dtype=np.float32, copy=False)
    return D, list(features.columns)


def fetch_and_preprocess_kddcup99(
    data_home: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fetch the assignment data and return ``D``, byte-string labels, and names.

    By default, the downloaded data is cached in ``.cache/scikit_learn`` next
    to this file.  This avoids relying on a writable home directory and keeps
    the cache out of the submitted code files.
    """
    if data_home is None:
        data_home = Path(__file__).resolve().parent / ".cache" / "scikit_learn"
    kdd = fetch_kddcup99(subset=None, as_frame=True, data_home=data_home)
    D, feature_names = preprocess_kdd_features(kdd.data)
    y = kdd.target.to_numpy(copy=True)
    return D, y, feature_names


if __name__ == "__main__":
    D, y, feature_names = fetch_and_preprocess_kddcup99()
    print(f"D shape: n={D.shape[0]}, d={D.shape[1]}")
    print(f"D dtype: {D.dtype}")
    print(f"labels shape: {y.shape}")
