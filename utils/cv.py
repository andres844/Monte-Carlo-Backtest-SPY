from typing import Iterator, Tuple
import numpy as np
import pandas as pd


def purged_time_series_cv(index: pd.DatetimeIndex, n_splits: int = 5, embargo: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Simplified purged CV: sequential splits without shuffling; applies an embargo
    of `embargo` bars on each side of the test fold removed from training indices.
    Yields (train_idx, test_idx) as integer arrays.
    """
    n = len(index)
    fold_size = n // n_splits
    for i in range(n_splits):
        test_start = i * fold_size
        test_end = n if i == n_splits - 1 else (i + 1) * fold_size
        test_idx = np.arange(test_start, test_end)
        left = max(0, test_start - embargo)
        right = min(n, test_end + embargo)
        train_idx = np.r_[np.arange(0, left), np.arange(right, n)]
        yield train_idx, test_idx

