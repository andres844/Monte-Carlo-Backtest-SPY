import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Sequence, Optional


def plot_underwater(equity: pd.Series, ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    eq = equity.fillna(method='ffill').fillna(0)
    dd = eq - eq.cummax()
    if ax is None:
        plt.figure(figsize=(10, 3))
        ax = plt.gca()
    ax.fill_between(eq.index, dd.values, 0, color='red', alpha=0.4)
    ax.set_ylabel('Drawdown')
    if title:
        ax.set_title(title)
    return ax


def plot_fan_chart(real_cum: pd.Series, permutations: Sequence[pd.Series],
                   quantiles: Iterable[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
                   ax: Optional[plt.Axes] = None,
                   colors: Sequence[str] = ("#2ca02c", "#1f77b4", "#9467bd", "#1f77b4", "#2ca02c")):
    """
    Plot fan chart using quantiles over time of permuted cumulative returns
    plus the real cumulative return in red.
    """
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

    # Align all series to the real_cum index
    perm_mat = []
    for s in permutations:
        perm_mat.append(pd.Series(s).reindex(real_cum.index).values)
    if len(perm_mat) == 0:
        return ax
    perm_arr = np.vstack(perm_mat)

    qs = np.quantile(perm_arr, q=list(quantiles), axis=0)
    # Shade bands: [0.05,0.95], [0.25,0.75]
    if len(quantiles) >= 5:
        ax.fill_between(real_cum.index, qs[0], qs[4], color=colors[0], alpha=0.08, label='5–95%')
        ax.fill_between(real_cum.index, qs[1], qs[3], color=colors[1], alpha=0.12, label='25–75%')
        ax.plot(real_cum.index, qs[2], color=colors[2], alpha=0.5, linewidth=1.0, label='Median perm')
    ax.plot(real_cum.index, real_cum.values, color='red', linewidth=2.0, label='Real')
    ax.legend(loc='upper left')
    return ax

