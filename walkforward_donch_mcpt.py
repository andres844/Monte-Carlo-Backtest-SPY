import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bar_permute import get_permutation
from strategies.donchian import walkforward_donch  # Your walkforward function
from utils.metrics import compute_forward_log_returns
from utils.plots import plot_fan_chart

# ------------------------------------------------------------------
# IMPORTANT:
# Use the full dataset (2000-2024) so that the walk-forward function
# can use data before the out-of-sample period (e.g., 2016-2020) for training,
# and then generate signals from 2020 onward.
# ------------------------------------------------------------------

# 1. Load your SPY daily data (full period: 2000-2024)
df = pd.read_csv("spy_data/spy_monthly_2000_2024.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# Compute forward log returns for the full dataset
df['r'] = compute_forward_log_returns(df['close'])

# (Optional) You might want to create a copy of the full data if needed
full_df = df.copy()

# ------------------------------------------------------------------
# 2. Run walk-forward on REAL data
# We set the out-of-sample start date to 2020-01-01.
# With a train_lookback of 365*4, the training window for the first re-optimization
# will be roughly from 2016 to 2020.
# ------------------------------------------------------------------
wf_signal_real = walkforward_donch(
    full_df,
    frequency="monthly",            # re-optimize
    opt_start_date=pd.Timestamp("2020-01-01")
)
# Compute cumulative returns over the full period (but out-of-sample signals start in 2020)
real_cum = (wf_signal_real * full_df['r']).cumsum()

# Calculate profit factor for the real strategy (only consider non-NaN returns)
real_rets = wf_signal_real * full_df['r']
real_pf = real_rets[real_rets > 0].sum() / real_rets[real_rets < 0].abs().sum()
print(f"Real walk-forward PF: {real_pf:.2f}")

# ------------------------------------------------------------------
# 3. Run Walk-Forward Permutation Test
# We run the permutation on the full dataset as well. In each iteration, the 
# same walk-forward routine is applied, so the only difference comes from the permutation.
# ------------------------------------------------------------------
n_permutations = 1000  # adjust as needed
perm_better_count = 1  # starting at 1 (as in your original logic)
permuted_pfs = []
perm_cum_returns = []  # We'll also store cumulative returns for a fan chart

print("Running walk-forward permutation test...")
for perm_i in tqdm(range(1, n_permutations)):
    # Permute the full dataset (using a training window offset)
    # Here, we use start_index = train_lookback so that the first 4 years remain unchanged,
    # or you can choose to permute the entire dataset.
    wf_perm = get_permutation(full_df, start_index=12 * 4, seed=perm_i)
    
    # Recalculate forward log returns on the permuted data
    wf_perm['r'] = compute_forward_log_returns(wf_perm['close'])
    
    # Run walk-forward test on the permuted data with the same parameters
    wf_signal_perm = walkforward_donch(
        wf_perm,
        frequency="monthly",
        opt_start_date=pd.Timestamp("2020-01-01")
    )
    
    perm_rets = wf_signal_perm * wf_perm['r']
    perm_pf = perm_rets[perm_rets > 0].sum() / perm_rets[perm_rets < 0].abs().sum()
    
    if perm_pf >= real_pf:
        perm_better_count += 1
    permuted_pfs.append(perm_pf)
    
    # Also store cumulative returns for fan chart plotting
    perm_cum = perm_rets.cumsum()
    perm_cum_returns.append(perm_cum)

walkforward_mcpt_pval = perm_better_count / n_permutations
print(f"Walkforward MCPT P-Value: {walkforward_mcpt_pval:.4f}")

# ------------------------------------------------------------------
# 4. Plot the Fan Chart of Walk-Forward Cumulative Returns
# Plot the real strategy's cumulative returns (in red)
plt.style.use("dark_background")
plt.figure(figsize=(10, 6))
plot_fan_chart(real_cum, perm_cum_returns)
plt.title(
    f"Walk-Forward Donchian Fan Chart (Monthly)\nReal PF={real_pf:.2f}, p-value={walkforward_mcpt_pval:.4f}",
    color="green",
)
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.legend(loc="upper left")
plt.show()

# ------------------------------------------------------------------
# 5. Also plot the histogram of permuted PFs vs. real PF
plt.figure(figsize=(10, 6))
pd.Series(permuted_pfs).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
plt.axvline(real_pf, color="red", linestyle="dashed", linewidth=2, label="Real")
plt.title(f"Walk-Forward Donchian MCPT (Monthly)\nReal PF={real_pf:.2f}, p-value={walkforward_mcpt_pval:.4f}", color="green")
plt.xlabel("Profit Factor")
plt.legend()
plt.show()
