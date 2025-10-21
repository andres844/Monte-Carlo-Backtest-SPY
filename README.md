# MC Permutation (SPY)

Monte Carlo Permutation Testing (MCPT) framework for SPY strategies with cost-aware evaluation, walk-forward testing, and robust null models.

---

## Strategies
- **Donchian Breakout**
- **Bollinger Bands** (mean-reversion)
- **Decision Tree**
- **Random Forest**

Centralized runner orchestrates experiments and plots across in-sample and walk-forward regimes.

---

## Data Requirements
Place CSVs under `spy_data/`:
- `spy_data/spy_daily_2000_2024.csv`
- `spy_data/spy_weekly_2000_2024.csv`
- `spy_data/spy_monthly_2000_2024.csv`

Required format:
- Columns: `date, open, high, low, close, volume`
- Index: `date` (parsed to `DatetimeIndex`)

---

## Installation
Python 3.10+ recommended

```bash
pip install pandas numpy matplotlib scikit-learn tqdm
```

---

## Highlights
- Unified metrics via `utils/metrics.py`:
  - Profit Factor, Sharpe, Sortino, Max Drawdown, Calmar, Turnover
- Forward returns:  
  $$r_t = \\log(C_{t+1}) - \\log(C_t)$$
- Transaction costs in bps (one-way; exit+enter counted), volatility targeting
- Permutation nulls:
  - **bar**: decouple gaps vs intrabars (legacy bar-wise)
  - **block**: stationary block bootstrap (`block_size` configurable)
  - **grouped-month**, **grouped-dow**: permute within calendar groups
  - **signflip**: flip sign of bar direction, preserving geometry
- Central runner (`runner.py`) supports:
  - Donchian, Tree, Bollinger, RF
  - In-sample & walk-forward
  - Fan charts, PF histograms
- Visuals:
  - Quantile fan charts
  - Parameter stability heatmap (Donchian)

---

## Quick Start

### Donchian (In-sample MCPT, monthly, block-bootstrap, 1bps)
```bash
python3 runner.py --tasks donchian_mcpt --frequency monthly --variant block --block_size 5 --n_permutations 1000 --cost_bps 1
```

### Donchian Stability Heatmap (2000–2019)
```bash
python3 runner.py --tasks donchian_stability --frequency monthly --start 2000-01-01 --end 2020-01-01
```

### Walk-forward Donchian (monthly)
```bash
python3 runner.py --tasks wf_donch_mcpt --frequency monthly --opt_start_date 2020-01-01 --variant grouped-month --n_permutations 1000 --cost_bps 2
```

### Decision Tree (weekly, fan + MCPT)
```bash
python3 runner.py --tasks tree_insample_fan tree_insample_mcpt --frequency weekly --tree_lags 3 12 48 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1
```

### Bollinger & RF (monthly, in-sample MCPT)
```bash
python3 runner.py --tasks bb_insample_mcpt rf_insample_mcpt --frequency monthly --bb_num_std 1 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1
```

### Walk-forward Bollinger & RF (weekly)
```bash
python3 runner.py --tasks wf_bb_mcpt wf_rf_mcpt --frequency weekly --opt_start_date 2020-01-01 --n_permutations 1000 --variant grouped-dow --cost_bps 1
```

---

## Runner Tasks

- `donchian_mcpt` → In-sample Donchian MCPT (fan chart + PF histogram)  
- `donchian_stability` → PF stability heatmap by lookback × year  
- `wf_donch_mcpt` → Walk-forward Donchian MCPT  
- `tree_insample_fan` → In-sample Decision Tree fan chart  
- `tree_insample_mcpt` → In-sample Decision Tree PF distribution  
- `wf_tree_mcpt` → Walk-forward Tree MCPT  
- `bb_insample_mcpt` → In-sample Bollinger MCPT  
- `rf_insample_mcpt` → In-sample Random Forest MCPT  
- `wf_bb_mcpt` → Walk-forward Bollinger MCPT  
- `wf_rf_mcpt` → Walk-forward RF MCPT  

Run multiple tasks at once:
```bash
python3 runner.py --tasks task1 task2
```

Run everything:
```bash
python3 runner.py --tasks all
```

---

## Common Flags

- `--frequency` : `daily|weekly|monthly`  
- `--start`, `--end` : in-sample window (e.g. `2000-01-01`, `2020-01-01`)  
- `--opt_start_date` : walk-forward start (default `2020-01-01`)  
- `--n_permutations` : number of permutations (compute-heavy)  
- `--cost_bps` : one-way transaction cost (bps)  
- `--variant` : `bar|block|grouped-month|grouped-dow|signflip`  
- `--block_size` : block length for block bootstrap  
- `--start_index` : keep initial non-permuted segment  
- `--tree_lags` : e.g. `--tree_lags 6 24 168`  
- `--bb_num_std` : Bollinger std multiple (default = 1)  
- `--no_plots` : disable plots (headless run)

---

## Project Structure

```
runner.py                         # central CLI
strategies/
    donchian.py                   # Donchian breakout + walk-forward
    spy_tree_strategy.py          # Decision Tree model + strategy
bar_permute.py                    # permutation null models
wf_BB_RF.py                       # Bollinger / RF strategies + walk-forward
donchian_stability.py             # parameter robustness heatmap
utils/
    metrics.py         # returns, PF/Sharpe/Sortino/MDD/Calmar, turnover, costs
    plots.py           # fan chart, underwater plots
    cv.py              # purged time series CV (scaffold)
Examples/
    experiments: insample_*, wf_*, plotting scripts
```

---

## Methodology Notes

- **Returns**: forward log returns  
  $$r_t = \\log(C_{t+1}) - \\log(C_t)$$
- **Profit Factor (PF)**: net of costs when `cost_bps > 0` (turnover-based)
- **Permutation invariants**:
  - *bar*: decouple gap vs intrabar
  - *block/grouped/signflip*: day-level reordering
- **Walk-forward defaults**:  
  - lookback ≈ 4 years  
  - step ≈ 1–3 units (depending on frequency)

---

## Performance Tips
- Start with `--n_permutations 200` for iteration; scale up later
- Use block bootstrap with moderate `--block_size` (5–10) to preserve short memory
- Add `--no_plots` for headless/CI runs
- Add CSV reporting later for automated evaluation
