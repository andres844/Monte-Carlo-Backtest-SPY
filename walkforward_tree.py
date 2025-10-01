import pandas as pd
import numpy as np
from spy_tree_strategy import train_tree, tree_strategy
from utils.metrics import compute_forward_log_returns, evaluate

def walkforward_tree(
    ohlc: pd.DataFrame,
    lags: tuple = (6, 24, 168),
    train_lookback: int = 365*4,   # ~4 years if daily
    train_step: int = 30,         # re-optimize every 30 bars
    opt_start_date: pd.Timestamp = pd.Timestamp("2020-01-01")
):
    """
    Perform a walk-forward test for a decision-tree strategy.
    
    Parameters:
      ohlc          : DataFrame with 'close' column (and ideally 'r' for log returns).
      lags          : Tuple of lags for train_tree() and tree_strategy().
      train_lookback: Number of bars to use for each training segment.
      train_step    : Number of bars to move forward between re-train steps.
      opt_start_date: The date at which we do our first training/optimization.

    Returns:
      wf_signal: A NumPy array of signals (1 or -1) for each row in ohlc.
    """
    # Ensure we have log returns 'r'
    if 'r' not in ohlc.columns:
        ohlc['r'] = np.log(ohlc['close']).diff().shift(-1)

    # Find the first index where date >= opt_start_date
    start_opt = ohlc.index.searchsorted(opt_start_date)
    if start_opt >= len(ohlc):
        raise ValueError("No data available after the specified opt_start_date.")
    
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)

    next_train = start_opt
    while next_train < n:
        # The training window: the prior 'train_lookback' bars
        train_start = next_train - train_lookback
        if train_start < 0:
            train_start = 0
        
        # Slice the training data
        train_df = ohlc.iloc[train_start:next_train].copy()

        # Train the tree
        model = train_tree(train_df, lags)
        
        # Generate signals for the entire dataset with that model
        # Then we only "activate" those signals for the next segment
        full_signal, _ = tree_strategy(ohlc, model, lags)

        # Fill the signals for the next train_step bars
        end_segment = min(next_train + train_step, n)
        wf_signal[next_train:end_segment] = full_signal[next_train:end_segment]
        
        next_train += train_step

    # If the last segment didn't fill up to the end, forward-fill
    # or remain NaN if you prefer no trades at the tail end
    # wf_signal = pd.Series(wf_signal).fillna(method='ffill').to_numpy()

    return wf_signal

def walkforward_pf(ohlc, wf_signal, cost_bps: float = 0.0):
    """
    Given an ohlc DataFrame and a walk-forward signal,
    compute the profit factor on the entire in-sample period.
    """
    # Ensure we have forward log returns
    if 'r' not in ohlc.columns:
        ohlc['r'] = compute_forward_log_returns(ohlc['close'])

    stats, _, net, _ = evaluate(pd.Series(wf_signal, index=ohlc.index), ohlc['r'], cost_bps=cost_bps)
    return stats.pf
