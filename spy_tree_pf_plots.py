import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from spy_tree_strategy import train_tree, tree_strategy
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns
from utils.plots import plot_fan_chart

def plot_tree_permutation_fan(df, start_date="2000-01-01", end_date="2020-01-01",
                              lags=(6, 24, 168), n_permutations=1000):
    """
    Plot a 'fan chart' of cumulative log returns for a decision-tree strategy
    on real data (in red) vs. multiple permutations (in faint white).
    
    Parameters:
      df            : DataFrame with columns ['close'] (and optionally 'r' if precomputed).
      start_date    : Start of in-sample period.
      end_date      : End of in-sample period (non-inclusive).
      lags          : Tuple of lags for train_tree() and tree_strategy().
      n_permutations: How many permutations to run.
    """
    plt.style.use("dark_background")

    # 1) Filter the in-sample data
    in_sample_df = df[(df.index >= start_date) & (df.index < end_date)].copy()
    if 'r' not in in_sample_df.columns:
        in_sample_df['r'] = compute_forward_log_returns(in_sample_df['close'])

    # 2) Train the tree on real data
    real_model = train_tree(in_sample_df, lags)
    real_signal, real_pf = tree_strategy(in_sample_df, real_model, lags)
    print(f"Real PF: {real_pf:.2f}")

    # Compute cumulative log returns for the real strategy
    # (We assume 'r' is log return from bar i to i+1)
    real_cum = (real_signal * in_sample_df['r']).cumsum()

    # 3) Generate permutations and store each perm's cumulative returns
    perm_cum_returns = []
    for i in tqdm(range(n_permutations), desc="Permutations"):
        perm_df = get_permutation(in_sample_df, start_index=0, seed=i)
        perm_model = train_tree(perm_df, lags)
        perm_signal, _ = tree_strategy(perm_df, perm_model, lags)

        # Because 'r' was not in perm_df, recompute:
        perm_df['r'] = compute_forward_log_returns(perm_df['close'])
        perm_cum = (perm_signal * perm_df['r']).cumsum()
        perm_cum_returns.append(perm_cum)

    # 4) Plot all permutations in faint white
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fan_chart(real_cum, perm_cum_returns, ax=ax)
    ax.set_title(f"In-Sample Permutation Test (Optimized Tree Strategy)\nPF={real_pf:.2f}", color='green', fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")

    plt.show()

if __name__ == "__main__":
    # Example usage with a daily SPY dataset from 2000 to 2020:
    df = pd.read_csv("spy_daily_2000_2024.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)

    # plot the fan from 2000 to 2020
    plot_tree_permutation_fan(df,
                              start_date="2000-01-01",
                              end_date="2020-01-01",
                              lags=(6, 24, 168),
                              n_permutations=500)  # or 1000, but 500 for quicker runs
