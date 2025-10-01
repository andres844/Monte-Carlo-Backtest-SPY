MC Permutation (SPY)

Monte Carlo Permutation Testing (MCPT) framework for SPY strategies with cost-aware evaluation, walk-forward testing, and robust null models.
Strategies: Donchian Breakout, Bollinger Bands mean-reversion, Decision Tree, Random Forest.
Centralized runner orchestrates experiments and plots across in-sample and walk-forward regimes.
Data Requirements

Place CSVs in repo root:
spy_daily_2000_2024.csv
spy_weekly_2000_2024.csv
spy_monthly_2000_2024.csv
Required columns: date, open, high, low, close, volume
Index: date (parsed to DatetimeIndex)
Install

Python 3.10+ recommended
pip install pandas numpy matplotlib scikit-learn tqdm
Highlights

Unified metrics via utils/metrics.py:
Profit Factor, Sharpe, Sortino, Max Drawdown, Calmar, Turnover
Forward returns r_t = log(C[t+1]) - log(C[t])
Transaction costs in bps (one-way; exit+enter counted), vol targeting
Permutation nulls:
bar: decouple gaps vs intrabars (legacy bar-wise)
block: stationary block bootstrap with configurable block_size
grouped-month, grouped-dow: permute within calendar groups
signflip: flip sign of bar direction, preserving bar geometry
Central runner (runner.py) with tasks for Donchian/Tree/Bollinger/RF, in-sample and walk-forward, with fan charts and PF histograms
Visuals: quantile fan charts, parameter stability heatmap (Donchian)
Quick Start

Donchian in-sample MCPT, monthly data, block-bootstrap, 1 bps:
python3 runner.py --tasks donchian_mcpt --frequency monthly --variant block --block_size 5 --n_permutations 1000 --cost_bps 1
Donchian stability heatmap (2000–2019):
python3 runner.py --tasks donchian_stability --frequency monthly --start 2000-01-01 --end 2020-01-01
Walk-forward Donchian MCPT (monthly):
python3 runner.py --tasks wf_donch_mcpt --frequency monthly --opt_start_date 2020-01-01 --variant grouped-month --n_permutations 1000 --cost_bps 2
Tree in-sample fan + MCPT (weekly):
python3 runner.py --tasks tree_insample_fan tree_insample_mcpt --frequency weekly --tree_lags 3 12 48 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1
Bollinger and RF in-sample MCPT (monthly):
python3 runner.py --tasks bb_insample_mcpt rf_insample_mcpt --frequency monthly --bb_num_std 1 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1
Walk-forward Bollinger/RF MCPT (weekly):
python3 runner.py --tasks wf_bb_mcpt wf_rf_mcpt --frequency weekly --opt_start_date 2020-01-01 --n_permutations 1000 --variant grouped-dow --cost_bps 1
Runner Tasks

donchian_mcpt: In-sample Donchian MCPT with fan chart + PF histogram
donchian_stability: PF stability heatmap by lookback×year
wf_donch_mcpt: Walk-forward Donchian MCPT (fan + histogram)
tree_insample_fan: In-sample Decision Tree fan chart
tree_insample_mcpt: In-sample Decision Tree PF distribution
wf_tree_mcpt: Walk-forward Tree MCPT (fan + histogram)
bb_insample_mcpt: In-sample Bollinger MCPT (PF distribution)
rf_insample_mcpt: In-sample Random Forest MCPT (PF distribution)
wf_bb_mcpt: Walk-forward Bollinger MCPT (PF distribution)
wf_rf_mcpt: Walk-forward RF MCPT (PF distribution)
Run multiple tasks at once: --tasks task1 task2 ...
Run everything: --tasks all
Common Flags

--frequency: daily|weekly|monthly
--start, --end: in-sample window (e.g., 2000-01-01, 2020-01-01)
--opt_start_date: walk-forward start (default 2020-01-01)
--n_permutations: number of permutations (compute heavy)
--cost_bps: one-way transaction cost (bps)
--variant: bar|block|grouped-month|grouped-dow|signflip
--block_size: block length for block bootstrap
--start_index: keep initial non-permuted segment
--tree_lags: e.g., --tree_lags 6 24 168
--bb_num_std: Bollinger std multiple (default 1)
--no_plots: disable plots (headless run)
Project Structure

runner.py central CLI for all experiments
donchian.py Donchian breakout + walk-forward
spy_tree_strategy.py Decision Tree model + strategy
wf_BB_RF.py Bollinger/RF strategies + walk-forward
bar_permute.py permutation null models (bar, block, grouped, signflip)
donchian_stability.py parameter robustness heatmap
utils/metrics.py returns, PF/Sharpe/Sortino/MDD/Calmar, turnover, costs, vol targeting
utils/plots.py fan chart and underwater plots
utils/cv.py purged time series CV (scaffold)
Examples/experiments: insample_*, wf_*, and plotting scripts
Methodology Notes

Returns: forward log returns r_t = log(C[t+1]) - log(C[t]) standard across the repo
PF calculation: net of costs when cost_bps > 0 (turnover-based)
Permutation invariants:
bar: decouple gap vs intrabar; extras are kept as-is (not synchronized)
block/grouped/signflip: day-level reordering; can sync extra columns if needed
Walk-forward defaults scale by frequency (lookback ~4 years; step ~1–3 units)
Performance Tips

Start with --n_permutations 200 to iterate quickly; scale up later
block bootstrap with moderate --block_size (e.g., 5–10) preserves short memory structure
Use --no_plots for headless or CI runs; add CSV reporting later if desired
Roadmap

Options/skew/smile features and plots (pending data source)
CSV reporting for runner outputs (PF, Sharpe, MDD, turnover, p-values)
Purged cross-validation for ML models and probability calibration + bet sizing
Caveats

Some tasks can be time-consuming (10^3–10^4 permutations)
Ensure data files include volume for RF features; extra columns are copied through permutations unless using day-level synced variants
