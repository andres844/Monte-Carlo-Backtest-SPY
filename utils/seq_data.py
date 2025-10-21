import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, win: int):
        self.X = X
        self.y = y
        self.win = win

    def __len__(self):
        return max(0, len(self.y) - self.win + 1)

    def __getitem__(self, i):
        x = self.X[i : i + self.win]
        target = self.y[i + self.win - 1]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(target, dtype=torch.float32),
        )


def to_arrays(feats_df, y_ser):
    joined = feats_df.join(y_ser, how='inner').dropna()
    X = joined[feats_df.columns].to_numpy()
    y = joined[y_ser.name].to_numpy()
    return X, y, joined.index
