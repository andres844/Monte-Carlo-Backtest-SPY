import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    # input df is assumed to have a 'close' column
    upper = ohlc['close'].rolling(lookback - 1).max().shift(1)
    lower = ohlc['close'].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc['close'] > upper] = 1
    signal.loc[ohlc['close'] < lower] = -1
    signal = signal.ffill()
    return signal

def optimize_donchian(ohlc: pd.DataFrame, frequency: str = 'daily'):
    """
    Finds the best Donchian lookback that maximizes profit factor.
    frequency: 'daily', 'weekly', or 'monthly' 
               -> determines the lookback range we search over.
    """
    # Decide on the range of lookbacks based on frequency
    if frequency == 'daily':
        lookback_range = range(12, 169)     # e.g., 12..168
    elif frequency == 'weekly':
        lookback_range = range(4, 53)       # e.g., 4..52
    elif frequency == 'monthly':
        lookback_range = range(3, 37)       # e.g., 3..36
    else:
        raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")

    best_pf = 0.0
    best_lookback = -1

    # log returns, shifting so each row has the next bar's return
    r = np.log(ohlc['close']).diff().shift(-1)

    for lookback in lookback_range:
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r
        
        pos_sum = sig_rets[sig_rets > 0].sum(skipna=True)
        neg_sum = sig_rets[sig_rets < 0].sum(skipna=True)

        if abs(neg_sum) < 1e-15:
            sig_pf = np.inf if pos_sum > 0 else 0
        else:
            sig_pf = pos_sum / abs(neg_sum)

        if sig_pf > best_pf:
            best_pf = sig_pf
            best_lookback = lookback

    return best_lookback, best_pf


def walkforward_donch(ohlc: pd.DataFrame,
                      frequency: str = 'daily',
                      train_lookback: int = None,
                      train_step: int = None,
                      opt_start_date: pd.Timestamp = pd.Timestamp("2004-01-01")):
    """
    Walk-forward optimization for the Donchian breakout strategy.
    
    Parameters:
      ohlc          : DataFrame with columns ['open', 'high', 'low', 'close']
      frequency     : 'daily', 'weekly', or 'monthly'
      train_lookback: Number of bars to use for each training period.
                      If None, defaults are used based on frequency:
                        - daily   : 365*4
                        - weekly  : 52*4
                        - monthly : 12*4
      train_step    : Number of bars to move forward between re-optimizations.
                      If None, defaults are used:
                        - daily   : 30
                        - weekly  : 4
                        - monthly : 3
      opt_start_date: The date at which the first re-optimization is performed.
    
    Returns:
      wf_signal: A NumPy array of walk-forward signals with the same length as ohlc.
    """
    # Set default training parameters based on frequency if not provided
    if train_lookback is None:
        if frequency == 'daily':
            train_lookback = 365 * 4
        elif frequency == 'weekly':
            train_lookback = 52 * 4
        elif frequency == 'monthly':
            train_lookback = 12 * 4
        else:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
    
    if train_step is None:
        if frequency == 'daily':
            train_step = 30
        elif frequency == 'weekly':
            train_step = 4
        elif frequency == 'monthly':
            train_step = 3
        else:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
    
    # Find the index for the first date >= opt_start_date
    start_opt = ohlc.index.searchsorted(opt_start_date)
    if start_opt >= len(ohlc):
        raise ValueError("No data available after the specified opt_start_date.")
    
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    
    next_train = start_opt
    while next_train < n:
        # Determine the training window: the previous 'train_lookback' bars (or as many as available)
        train_start = next_train - train_lookback
        if train_start < 0:
            train_start = 0
        
        train_df = ohlc.iloc[train_start:next_train]
        
        # Optimize the Donchian strategy on the training window,
        # using the appropriate lookback range for the given frequency.
        best_lookback, _ = optimize_donchian(train_df, frequency=frequency)
        # Generate the full signal for the entire dataset using the optimized lookback
        full_signal = donchian_breakout(ohlc, best_lookback)
        
        # Apply the signal for the next segment (next 'train_step' bars)
        end_segment = min(next_train + train_step, n)
        wf_signal[next_train:end_segment] = full_signal[next_train:end_segment]
        
        next_train += train_step
    
    return wf_signal

if __name__ == '__main__':
    df = pd.read_csv("spy_daily_2000_2024.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)

    df = df[(df.index.year >= 2000) & (df.index.year < 2020)] 
    best_lookback, best_real_pf = optimize_donchian(df)


    signal = donchian_breakout(df, best_lookback) 

    df['r'] = np.log(df['close']).diff().shift(-1)
    df['donch_r'] = df['r'] * signal

    plt.style.use("dark_background")
    df['donch_r'].cumsum().plot(color='red')
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel('Cumulative Log Return')
    plt.show()