import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns

# Import strategy functions and walk-forward functions
from wf_BB_RF import (
    bollinger_strategy,
    train_rf,
    rf_strategy,
    walkforward_bollinger,
    walkforward_rf,
    walkforward_pf,
    run_wf_permutation,
    in_sample_wf_permutation_test
)

plt.style.use("dark_background")

# ------------------------------
# Define Frequency Parameters
# ------------------------------
# Change the value of "frequency" below to "daily", "weekly", or "monthly"
frequency = "monthly"

# All parameters related to the timeframe are stored in this dictionary:
freq_params = {
    "daily": {
        "file": "spy_daily_2000_2024.csv",
        "frequency": "daily",
        "train_lookback": 365 * 4,  # ~4 years
        "train_step": 30,
        "num_std": 2,             # Bollinger standard deviation multiplier
        "plot_label": "Daily"
    },
    "weekly": {
        "file": "spy_weekly_2000_2024.csv",
        "frequency": "weekly",
        "train_lookback": 52 * 4,  # ~4 years of weeks
        "train_step": 4,
        "num_std": 1,
        "plot_label": "Weekly"
    },
    "monthly": {
        "file": "spy_monthly_2000_2024.csv",
        "frequency": "monthly",
        "train_lookback": 12 * 4,  # ~4 years of months
        "train_step": 3,
        "num_std": 1,
        "plot_label": "Monthly"
    }
}

params = freq_params[frequency]
# One-way transaction cost in bps (set to 0 if not desired)
cost_bps = 0.0

# ------------------------------
# Load Full Dataset (2000-2024)
# ------------------------------
df = pd.read_csv(params["file"], parse_dates=["date"])
df.set_index("date", inplace=True)
df['r'] = compute_forward_log_returns(df['close'])  # forward log returns for entire dataset

# ------------------------------
# Out-of-Sample Test Period: 2020-01-01 onward
# ------------------------------
opt_start_date = pd.Timestamp("2020-01-01")
test_mask = df.index >= opt_start_date

# ------------------------------
# Walk-Forward Test for Bollinger Bands Strategy (Out-of-Sample)
# ------------------------------
wf_signal_bb = walkforward_bollinger(
    df,
    frequency=params["frequency"],
    window=None,  # use default window based on frequency (20 for daily, etc.)
    num_std=params["num_std"],
    train_lookback=params["train_lookback"],
    train_step=params["train_step"],
    opt_start_date=opt_start_date
)
real_pf_bb = walkforward_pf(df, wf_signal_bb, cost_bps=cost_bps)
print(f"Bollinger Bands Test PF: {real_pf_bb:.4f}")

# Plot cumulative returns for the test period
test_cum_bb = (wf_signal_bb[test_mask] * df['r'][test_mask]).cumsum()
plt.figure(figsize=(10, 6))
plt.plot(test_cum_bb.index, test_cum_bb, color='red', linewidth=2, 
         label=f"Real Test ({params['plot_label']}) (PF={real_pf_bb:.2f})")
plt.title(f"Bollinger Bands Strategy ({params['plot_label']}) Out-Of-Sample Cumulative Returns\nPF={real_pf_bb:.2f}")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend(loc="upper left")
plt.show()

# Permutation test for Bollinger Bands on full dataset (walk-forward)
real_pf = real_pf_bb  # set global for permutation test function
# Note: strategy_args now includes frequency, window (None), num_std, train_lookback, train_step, and opt_start_date.
perm_pfs_bb, p_value_bb = in_sample_wf_permutation_test(
    walkforward_bollinger, df,
    strategy_args=(params["frequency"], None, params["num_std"], params["train_lookback"], params["train_step"], opt_start_date),
    real_pf=real_pf_bb,
    n_permutations=1000,
    cost_bps=cost_bps,
)
print(f"Bollinger Bands ({params['plot_label']}) WF MCPT P-Value: {p_value_bb:.4f}")
perm_pfs_bb_clean = [pf for pf in perm_pfs_bb if np.isfinite(pf)]

plt.figure(figsize=(10, 6))
pd.Series(perm_pfs_bb_clean).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
plt.axvline(real_pf_bb, color="red", linestyle="dashed", linewidth=2, label="Real")
plt.title(f"Bollinger Bands ({params['plot_label']}) WF MCPT\nPF={real_pf_bb:.2f}, p={p_value_bb:.4f}", color="green")
plt.xlabel("Profit Factor")
plt.legend()
plt.show()

# ------------------------------
# Walk-Forward Test for Random Forest Strategy (Out-of-Sample)
# ------------------------------
# For demonstration, we'll train the RF model on the full dataset.
rf_model = train_rf(df, frequency=params["frequency"])
wf_signal_rf = walkforward_rf(
    df,
    frequency=params["frequency"],
    train_lookback=params["train_lookback"],
    train_step=params["train_step"],
    opt_start_date=opt_start_date
)
real_pf_rf = walkforward_pf(df, wf_signal_rf, cost_bps=cost_bps)
print(f"Random Forest Test PF: {real_pf_rf:.2f}")

# Plot cumulative returns for the test period
test_cum_rf = (wf_signal_rf[test_mask] * df['r'][test_mask]).cumsum()
plt.figure(figsize=(10, 6))
plt.plot(test_cum_rf.index, test_cum_rf, color='orange', linewidth=2, 
         label=f"Real Test ({params['plot_label']}) (PF={real_pf_rf:.2f})")
plt.title(f"Random Forest Strategy ({params['plot_label']}) Out-Of-Sample Cumulative Returns\nPF={real_pf_rf:.2f}")
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend(loc="upper left")
plt.show()

# Permutation test for Random Forest strategy on full dataset (walk-forward)
real_pf = real_pf_rf  # set global for permutation test function
perm_pfs_rf, p_value_rf = in_sample_wf_permutation_test(
    walkforward_rf, df,
    strategy_args=(params["frequency"], params["train_lookback"], params["train_step"], opt_start_date),
    real_pf=real_pf_rf,
    n_permutations=1000,
    cost_bps=cost_bps,
)
print(f"Random Forest ({params['plot_label']}) WF MCPT P-Value: {p_value_rf:.4f}")

plt.figure(figsize=(10, 6))
pd.Series(perm_pfs_rf).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
plt.axvline(real_pf_rf, color="red", linestyle="dashed", linewidth=2, label="Real")
plt.title(f"Random Forest ({params['plot_label']}) WF MCPT\nPF={real_pf_rf:.2f}, p={p_value_rf:.4f}", color="green")
plt.xlabel("Profit Factor")
if real_pf_rf > 100:
    plt.xscale("log")
plt.legend()
plt.show()
