import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from donchian import optimize_donchian
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns

# 1. Load your SPY daily data
df = pd.read_csv("spy_daily_2000_2024.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# 2. Filter in-sample
train_df = df[(df.index >= "2000-01-01") & (df.index < "2020-01-01")]

# 3. Optimize Donchian on real data
best_lookback, best_real_pf = optimize_donchian(train_df)
print("In-sample PF", best_real_pf, "Best Lookback", best_lookback)

# 4. Monte Carlo Permutation
n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("In-Sample MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(train_df)
    _, best_perm_pf = optimize_donchian(train_perm)

    if best_perm_pf >= best_real_pf:
        perm_better_count += 1

    permuted_pfs.append(best_perm_pf)

insample_mcpt_pval = perm_better_count / n_permutations
print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

plt.style.use('dark_background')
pd.Series(permuted_pfs).hist(color='blue', label='Permutations')
plt.axvline(best_real_pf, color='red', label='Real')
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval}")
plt.grid(False)
plt.legend()
plt.show()
