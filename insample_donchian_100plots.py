import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from donchian import optimize_donchian, donchian_breakout
from bar_permute import get_permutation

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
    # Real log returns
    log_close_real = np.log(train_df['close'])
    real_r = log_close_real.diff().shift(-1)
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
        # Compute cumulative log returns
        log_close_perm = np.log(train_perm['close'])
        perm_r = log_close_perm.diff().shift(-1)
        perm_strategy_rets = perm_signal * perm_r
        perm_cum_log = perm_strategy_rets.cumsum()

        perm_cum_logs.append(perm_cum_log)
        perm_pfs.append(best_pf_perm)

    # 5. Plot all permutations + real data
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each permutation as a faint line
    for series in perm_cum_logs:
        ax.plot(series.index, series.values, color='white', alpha=0.15)

    # Plot the real data in red
    ax.plot(real_cum_log.index, real_cum_log.values,
            color='red', linewidth=2.0,
            label=f"Real Optimized (PF={best_real_pf:.2f})")

    ax.set_title("In-Sample Permutation Test (Optimized Donchian Strategy on Monthly)", color='green', fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Log Return")
    ax.legend(loc="upper left")

    plt.show()