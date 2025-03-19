import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from donchian import optimize_donchian, donchian_breakout
from bar_permute import get_permutation

def plot_real_vs_perm(df, in_sample_start="2000-01-01", in_sample_end="2020-01-01", seed=420):
    """
    1) Filter df for the chosen in-sample period.
    2) Optimize Donchian breakout on real data.
    3) Permute the data, re-optimize, and generate signals on the permuted data.
    4) Plot both cumulative log returns on the same figure.
    """

    # 1. Filter in-sample
    train_df = df[(df.index >= in_sample_start) & (df.index < in_sample_end)]

    # 2. Optimize on real data
    best_lookback, best_real_pf = optimize_donchian(train_df,frequency= 'weekly')
    print(f"Real Data -> Best Lookback: {best_lookback}, PF: {best_real_pf:.2f}")

    # Generate Donchian signals on real data
    signal_real = donchian_breakout(train_df, best_lookback)

    # Calculate cumulative log returns for real data
    # r = log(close[i+1]) - log(close[i])
    log_close_real = np.log(train_df['close'])
    r_real = log_close_real.diff().shift(-1)  # shift -1 so each bar has "next" bar return
    strategy_rets_real = signal_real * r_real
    cum_log_real = strategy_rets_real.cumsum()

    # 3. Permute the data and re-optimize
    train_perm = get_permutation(train_df, start_index=0, seed=seed)
    best_lookback_perm, best_perm_pf = optimize_donchian(train_perm, frequency= 'weekly')
    print(f"Perm Data -> Best Lookback: {best_lookback_perm}, PF: {best_perm_pf:.2f}")

    # Generate Donchian signals on permuted data
    signal_perm = donchian_breakout(train_perm, best_lookback_perm)

    # Calculate cumulative log returns for permuted data
    log_close_perm = np.log(train_perm['close'])
    r_perm = log_close_perm.diff().shift(-1)
    strategy_rets_perm = signal_perm * r_perm
    cum_log_perm = strategy_rets_perm.cumsum()

    # 4. Plot both lines
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(cum_log_real.index, cum_log_real, color='red', label=f"Real Donchian (PF={best_real_pf:.2f})")
    ax.plot(cum_log_perm.index, cum_log_perm, color='orange', label=f"Perm Donchian (PF={best_perm_pf:.2f})")

    ax.set_title("Compare Donchian on Real vs. Permuted Data")
    ax.set_ylabel("Cumulative Log Return")
    ax.set_xlabel("Date")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # 1. Load your daily CSV
    df = pd.read_csv("spy_weekly_2000_2024.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)

    # 2. Run the comparison plot
    plot_real_vs_perm(df, in_sample_start="2000-01-01", in_sample_end="2020-01-01")