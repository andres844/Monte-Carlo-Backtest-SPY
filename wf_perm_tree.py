import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from strategies.spy_tree_strategy import train_tree, tree_strategy
from bar_permute import get_permutation
from walkforward_tree import walkforward_tree, walkforward_pf
from utils.metrics import compute_forward_log_returns
from utils.plots import plot_fan_chart

# 1) Load the full SPY daily dataset (2000–2024)
df = pd.read_csv("spy_data/spy_monthly_2000_2024.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# Use the full dataset and compute log returns (r)
full_df = df.copy()
full_df['r'] = compute_forward_log_returns(full_df['close'])

# 2) Run walk-forward on REAL data
#    Using an out-of-sample period starting 2020-01-01
wf_signal_real = walkforward_tree(
    full_df,
    lags=(2, 6, 24),
    train_lookback=12*4,   # ~4 years for daily data
    train_step=3,
    opt_start_date=pd.Timestamp("2020-01-01")
)
real_pf = walkforward_pf(full_df, wf_signal_real)
print(f"Real walk-forward PF: {real_pf:.2f}")

# Compute cumulative returns for the real strategy
real_cum = (wf_signal_real * full_df['r']).cumsum()

# 3) Run walk-forward permutation test and store cumulative returns
n_permutations = 1000  # Adjust as needed
perm_better_count = 1  # start count as in your original code
perm_pfs = []
perm_cum_returns = []

print("Running walk-forward permutation test...")
for i in tqdm(range(n_permutations)):
    # Permute the full dataset (which covers both training and test periods)
    perm_df = get_permutation(full_df, start_index=0, seed=i)
    # Recompute forward log returns for permuted data
    perm_df['r'] = compute_forward_log_returns(perm_df['close'])
    
    # Run walk-forward test on permuted data
    wf_signal_perm = walkforward_tree(
        perm_df,
        lags=(2, 6, 24),
        train_lookback=12*4,
        train_step=3,
        opt_start_date=pd.Timestamp("2020-01-01")
    )
    perm_pf = walkforward_pf(perm_df, wf_signal_perm)
    if perm_pf >= real_pf:
        perm_better_count += 1
    perm_pfs.append(perm_pf)
    
    # Compute cumulative returns for this permutation and store it
    perm_cum = (wf_signal_perm * perm_df['r']).cumsum()
    perm_cum_returns.append(perm_cum)

p_value = perm_better_count / n_permutations
print(f"Walk-forward Permutation Test P-Value: {p_value:.4f}")

# 4) Plot the fan chart of walk-forward cumulative returns
plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
plot_fan_chart(real_cum, perm_cum_returns)
plt.title(
    f"Walk-Forward Tree Strategy Fan Chart (Monthly)\nReal PF={real_pf:.2f}, p-value={p_value:.4f}",
    color="green",
)
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend(loc="upper left")
plt.show()

# 5) Optionally, also plot the distribution of permuted profit factors:
plt.figure(figsize=(10,6))
pd.Series(perm_pfs).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
plt.axvline(real_pf, color="red", linestyle="dashed", linewidth=2, label="Real")
plt.title(f"Walk-Forward Tree Strategy MCPT (Monthly)\nReal PF={real_pf:.2f}, p-value={p_value:.4f}", color="green")
plt.xlabel("Profit Factor")
plt.legend()
plt.show()
