import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from donchian import optimize_donchian, donchian_breakout
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns
from utils.plots import plot_fan_chart

# Number of total permutations to run
N_PERMUTATIONS = 1000

if __name__ == "__main__":

    # 1. Load your SPY dataset (daily/weekly/monthly)
    df = pd.read_csv('spy_monthly_2000_2024.csv', parse_dates=['date'])
    df.set_index('date', inplace=True)

    # 2. Filter for in-sample period
    train_df = df[(df.index >= "2000-01-01") & (df.index < "2020-01-01")]

    # 3. Optimize Donchian breakout on the real (unpermuted) in-sample data
    best_lookback, best_real_pf = optimize_donchian(train_df,frequency= 'monthly')
    print("In-sample PF:", best_real_pf, "Best Lookback:", best_lookback)

    # Generate real Donchian signal
    real_signal = donchian_breakout(train_df, best_lookback)
    # Real forward log returns
    real_r = compute_forward_log_returns(train_df['close'])
    real_strategy_rets = real_signal * real_r
    real_cum_log = real_strategy_rets.cumsum()

    # 4. Run permutations and store cumulative log returns
    perm_cum_logs = []
    perm_pfs = []

    print(f"\nRunning {N_PERMUTATIONS} permutations...\n")
    for _ in tqdm(range(N_PERMUTATIONS)):
        # Permute the data
        train_perm = get_permutation(train_df, start_index=0)
        # Optimize Donchian on permuted data
        best_lookback_perm, best_pf_perm = optimize_donchian(train_perm,frequency= 'monthly')

        # Generate permuted Donchian signal
        perm_signal = donchian_breakout(train_perm, best_lookback_perm)
        # Compute forward log returns
        perm_r = compute_forward_log_returns(train_perm['close'])
        perm_strategy_rets = perm_signal * perm_r
        perm_cum_log = perm_strategy_rets.cumsum()

        perm_cum_logs.append(perm_cum_log)
        perm_pfs.append(best_pf_perm)

    # 5. Plot all permutations + real data
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_fan_chart(real_cum_log, perm_cum_logs, ax=ax)
    ax.set_title(
        f"In-Sample Permutation Test (Optimized Donchian Strategy on Monthly)\nPF={best_real_pf:.2f}",
        color='green', fontsize=16
    )

    plt.show()
