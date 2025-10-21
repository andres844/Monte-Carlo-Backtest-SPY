# Monte-Carlo-Backtest-SPY

End-to-end Monte Carlo Permutation Testing (MCPT) and backtesting framework for SPY across daily/weekly/monthly horizons. It includes classic rules-based strategies, tree/forest baselines, and sequence models (TCN/LSTM), with cost-aware evaluation, in-sample testing, and walk-forward workflows.

---

## What This Project Does
- Cost-aware backtesting with robust metrics (PF, Sharpe, Sortino, MDD, Calmar, turnover)
- Multiple strategies:
  - Donchian breakout (optimize, stability heatmap, walk-forward)
  - Bollinger Bands mean-reversion (in-sample + walk-forward)
  - Decision Tree (in-sample fan/MCPT + walk-forward MCPT)
  - Random Forest (in-sample + walk-forward MCPT)
  - RNN strategies (TCN/LSTM): in-sample MCPT, walk-forward with and without retraining
  - RNN Kelly variants: MC-Dropout Kelly and Dual-Head Directional Kelly (in-sample + walk-forward)
- MCPT null models for significance testing:
  - Bar-wise gap vs intrabar decoupling
  - Stationary block bootstrap
  - Grouped-by calendar (month, day-of-week)
  - Sign-flip preserving bar geometry
- Visualizations: PF histograms, fan charts, underwater (drawdown) plots
- Single CLI runner to orchestrate all experiments and plots

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
- Python 3.10+
- CPU-only: `pip install pandas numpy matplotlib scikit-learn tqdm torch`
- GPU (optional): install a CUDA-enabled PyTorch build from pytorch.org

---

## Core Modules
- `runner.py` – central CLI for all experiments
- `bar_permute.py` – permutation null models:
  - `get_permutation` (decouple gap vs intrabar)
  - `get_permutation_block_bootstrap` (block bootstrap)
  - `get_permutation_grouped` (month / day-of-week)
  - `get_permutation_sign_flip` (direction flips; geometry preserved)
- `strategies/donchian.py` – Donchian breakout, optimization, walk-forward
- `strategies/spy_tree_strategy.py` – Decision Tree features/fit/strategy
- `wf_BB_RF.py` – Bollinger Bands + Random Forest strategies and walk-forward wrappers
- `strategies/rnn_strategy.py` – sequence models (TCN/LSTM) with MCPT, Kelly sizing, and walk-forward
- `features/seq_features.py` – OHLCV-derived feature engineering (z-scored)
- `models/seq_models.py` – TCN, LSTM, and Dual-Head Kelly backbones
- `utils/metrics.py` – evaluation, costs, turnover, drawdowns, ratios
- `utils/plots.py` – fan chart, underwater plot helpers
- `utils/seq_data.py` – sequence dataset utilities for PyTorch
- `experiments/rnn_grid.py` – small RNN grid search script

---

## MCPT Null Models
- `bar`: shuffle gaps separately from intrabar legs; recompose OHLC coherently
- `block`: stationary block bootstrap over day-level bars (`--block_size`)
- `grouped-month`, `grouped-dow`: shuffle within calendar groups
- `signflip`: randomly flip sign of gaps/intrabar, swapping high/low as needed

---

## Runner Tasks (CLI)
Run one or multiple tasks via `--tasks`. Use `--tasks all` to run everything sequentially.

- Donchian
  - `donchian_mcpt` – In-sample MCPT (fan chart + PF histogram)
  - `donchian_stability` – PF heatmap by lookback × year
  - `wf_donch_mcpt` – Walk-forward MCPT
- Tree / Forest / Bollinger
  - `tree_insample_fan`, `tree_insample_mcpt`, `wf_tree_mcpt`
  - `bb_insample_mcpt`, `wf_bb_mcpt`
  - `rf_insample_mcpt`, `wf_rf_mcpt`
- RNN (TCN/LSTM)
  - `rnn_insample_mcpt` – Fit sequence model in-sample + MCPT
  - `wf_rnn_mcpt` – Train in-sample, test out-of-sample, MCPT on OOS window (fixed model)
  - `wf_rnn_retrain` – Full walk-forward retraining with calibrated scaling
  - `rnn_grid` – compact grid over freq/MCPT variants
  - `rnn_mc_kelly`, `wf_rnn_mc_kelly` – MC-Dropout Kelly (in-sample / walk-forward)
  - `rnn_dual_kelly`, `wf_rnn_dual_kelly` – Dual-Head Kelly (in-sample / walk-forward)

---

## Common Flags
- `--frequency`           `daily|weekly|monthly`
- `--start`, `--end`      in-sample window (e.g., `2000-01-01`, `2020-01-01`)
- `--opt_start_date`      walk-forward start (default `2020-01-01`)
- `--n_permutations`      number of permutations (compute-heavy)
- `--variant`             `bar|block|grouped-month|grouped-dow|signflip`
- `--block_size`          block length for block bootstrap
- `--start_index`         keep initial non-permuted segment
- `--cost_bps`            one-way transaction cost (bps)
- `--no_plots`            disable plots

Tree/Bollinger/RF specifics
- `--tree_lags`           e.g., `--tree_lags 6 24 168`
- `--bb_num_std`          Bollinger std multiple (default 1)

RNN specifics
- `--model_type`          `tcn|lstm`
- `--horizon`             forward horizon (bars)
- `--win`                 sequence window length
- `--mcpt_variant`        `block|grouped-month|grouped-dow|signflip`
- `--epochs`, `--batch_size`, `--lr`
- `--device`              `auto|cpu|cuda|mps`
- `--seed_model`          model seed
- `--no_calibrate`        skip in-sample scale calibration
- `--train_lookback`, `--train_step` (walk-forward retraining)
- `--mc_passes`           MC-dropout forward passes
- `--kelly_cap`           cap for Kelly sizing

---

## Quick Examples
- Donchian (monthly in-sample MCPT):
  `python3 runner.py --tasks donchian_mcpt --frequency monthly --variant block --block_size 5 --n_permutations 1000 --cost_bps 1`
- Donchian stability heatmap:
  `python3 runner.py --tasks donchian_stability --frequency monthly --start 2000-01-01 --end 2020-01-01`
- WF Donchian (monthly):
  `python3 runner.py --tasks wf_donch_mcpt --frequency monthly --opt_start_date 2020-01-01 --variant grouped-month --n_permutations 1000 --cost_bps 2`
- Decision Tree (weekly fan + MCPT):
  `python3 runner.py --tasks tree_insample_fan tree_insample_mcpt --frequency weekly --tree_lags 3 12 48 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1`
- Bollinger + RF (monthly in-sample):
  `python3 runner.py --tasks bb_insample_mcpt rf_insample_mcpt --frequency monthly --bb_num_std 1 --n_permutations 1000 --variant block --block_size 5 --cost_bps 1`
- WF Bollinger + RF (weekly):
  `python3 runner.py --tasks wf_bb_mcpt wf_rf_mcpt --frequency weekly --opt_start_date 2020-01-01 --n_permutations 1000 --variant grouped-dow --cost_bps 1`
- RNN in-sample MCPT (TCN):
  `python3 runner.py --tasks rnn_insample_mcpt --frequency weekly --horizon 5 --win 128 --model_type tcn --mcpt_variant block --n_permutations 300 --epochs 20 --batch_size 64 --lr 1e-3`
- WF RNN fixed-model MCPT:
  `python3 runner.py --tasks wf_rnn_mcpt --frequency weekly --horizon 5 --win 128 --model_type lstm --mcpt_variant block --n_permutations 300`
- WF RNN retrain:
  `python3 runner.py --tasks wf_rnn_retrain --frequency daily --horizon 5 --win 128 --model_type tcn --train_lookback 730 --train_step 30`
- RNN MC-Dropout Kelly (in-sample / WF):
  `python3 runner.py --tasks rnn_mc_kelly --frequency monthly --mc_passes 20 --kelly_cap 1.0`
  `python3 runner.py --tasks wf_rnn_mc_kelly --frequency monthly --mc_passes 20 --kelly_cap 1.0`
- RNN Dual-Head Kelly (in-sample / WF):
  `python3 runner.py --tasks rnn_dual_kelly --frequency weekly --kelly_cap 1.0`
  `python3 runner.py --tasks wf_rnn_dual_kelly --frequency weekly --kelly_cap 1.0`

---

## Visualizations
- PF histograms and MCPT fan charts (`utils/plots.py`)
- Drawdown (underwater) plots
- Saved example figures under `images/`

---

## Methodology Notes
- Forward returns: `r_t = log(C_{t+1}) - log(C_t)`
- Costs: turnover-based, one-way `--cost_bps` (exit+enter counted)
- Volatility targeting available via `utils/metrics.py:evaluate`
- Walk-forward defaults: lookback ≈ 4 years; step ≈ 1–3 units (by frequency)

---

## Project Structure
```
runner.py
bar_permute.py
strategies/
  donchian.py
  spy_tree_strategy.py
  rnn_strategy.py
wf_BB_RF.py
utils/
  metrics.py
  plots.py
  seq_data.py
features/
  seq_features.py
models/
  seq_models.py
experiments/
  rnn_grid.py
spy_data/    # input CSVs
results/     # experiment outputs (CSV)
images/      # example plots
```

---

## Tips
- Start with `--n_permutations 200` for iteration and increase later
- Prefer `block` null with `--block_size 5–10` to preserve short memory
- Use GPU (`--device cuda`) for RNN tasks when available
- Add `--no_plots` for headless/CI runs
