import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from strategies.spy_tree_strategy import train_tree, tree_strategy
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns
from utils.plots import plot_fan_chart

# 1. Load the SPY daily dataset from CSV
df = pd.read_csv('spy_data/spy_monthly_2000_2024.csv', parse_dates=['date'])
df.set_index('date', inplace=True)

# 2. Calculate log returns (shifted so that each row gets the next bar's return)
df['r'] = compute_forward_log_returns(df['close'])

# 3. Filter training data for 2000–2019
train_df = df[(df.index.year >= 2000) & (df.index.year < 2020)]

# 4. Train the decision tree strategy on the real training data
real_tree = train_tree(train_df, (2, 6, 24))
real_is_signal, real_is_pf = tree_strategy(train_df, real_tree, (2, 6, 24))
print("Real Profit Factor:", real_is_pf)

# 5. Run Monte Carlo Permutation Test on the training data
n_permutations = 10000
perm_better_count = 1  # start at 1 (as in your original code)
permuted_pfs = []

print("In-Sample MCPT:")
for perm_i in tqdm(range(1, n_permutations)):
    # Permute the training data
    train_perm = get_permutation(train_df, start_index=0)
    
    # Train a new decision tree on the permuted data
    perm_nn = train_tree(train_perm, (2, 6, 24))
    _, perm_pf = tree_strategy(train_perm, perm_nn, (2, 6, 24))
    
    if perm_pf >= real_is_pf:
        perm_better_count += 1
    
    permuted_pfs.append(perm_pf)

insample_mcpt_pval = perm_better_count / n_permutations
print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

# 6. Plot the distribution of permuted profit factors vs. the real profit factor
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
pd.Series(permuted_pfs).hist(bins=30, color='blue', label='Permutations', alpha=0.8)
plt.axvline(real_is_pf, color='red', linestyle='dashed', linewidth=2, label='Real')
plt.xlabel("Profit Factor")
plt.title(f"In-sample Tree Classifier (Monthly) MCPT P-Value: {insample_mcpt_pval:.4f}")
plt.grid(False)
plt.legend()
plt.show()
