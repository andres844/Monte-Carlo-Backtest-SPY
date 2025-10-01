import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from bar_permute import get_permutation
from utils.metrics import compute_forward_log_returns, evaluate

# ------------------------------
# Strategy 1: Bollinger Bands Mean Reversion
# ------------------------------
def bollinger_strategy(ohlc: pd.DataFrame, frequency="daily", window=None, num_std=1, cost_bps: float = 0.0):
    """
    Bollinger Bands Mean Reversion Strategy:
      - Computes the moving average (MA) and standard deviation (std) of 'close'
      - Defines upper band = MA + num_std * std and lower band = MA - num_std * std
      - Signal = 1 when price < lower band (buy) and -1 when price > upper band (sell/short)
      - The signal is forward-filled.
    
    Parameters:
      ohlc      : DataFrame containing at least a 'close' column.
      frequency : One of "daily", "weekly", or "monthly". If window is None, defaults are:
                    daily   -> 20,
                    weekly  -> 5,
                    monthly -> 3.
      window    : (Optional) The rolling window length. If None, it is set based on frequency.
      num_std   : Number of standard deviations for the bands.
    
    Returns:
      signal    : Series of trading signals (1 or -1).
      pf        : Profit Factor = (sum of positive returns) / (sum of absolute negative returns)
    """
    if window is None:
        if frequency == "daily":
            window = 20
        elif frequency == "weekly":
            window = 5
        elif frequency == "monthly":
            window = 3
        else:
            raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
    
    ma = ohlc['close'].rolling(window=window).mean()
    std = ohlc['close'].rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    signal = pd.Series(np.nan, index=ohlc.index)
    signal[ohlc['close'] < lower] = 1
    signal[ohlc['close'] > upper] = -1
    signal = signal.ffill().fillna(0)
    
    r = compute_forward_log_returns(ohlc['close'])
    stats, _, net, _ = evaluate(signal, r, cost_bps=cost_bps)
    return signal, stats.pf

# ------------------------------
# Strategy 2: Random Forest–Based Strategy
# ------------------------------
def train_rf(ohlc: pd.DataFrame, frequency="daily"):
    if frequency == "daily":
        r1, r5, r10, vol_win = 1, 5, 10, 20
    elif frequency == "weekly":
        r1, r5, r10, vol_win = 1, 3, 5, 10
    elif frequency == "monthly":
        r1, r5, r10, vol_win = 1, 2, 3, 6
    else:
        raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
    
    log_c = np.log(ohlc['close'])
    ret1 = log_c.diff(r1)
    ret5 = log_c.diff(r5)
    ret10 = log_c.diff(r10)
    vol_ratio = ohlc['volume'] / ohlc['volume'].rolling(window=vol_win).mean()
    
    features = pd.concat([ret1, ret5, ret10, vol_ratio], axis=1)
    features.columns = ['ret1', 'ret5', 'ret10', 'vol_ratio']
    
    target = np.sign(log_c.diff().shift(-1))
    target = ((target + 1) / 2)
    
    dataset = pd.concat([features, target.rename("target")], axis=1)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    dataset['target'] = dataset['target'].astype(int)
    
    X = dataset[['ret1', 'ret5', 'ret10', 'vol_ratio']].to_numpy()
    y = dataset['target'].to_numpy()
    
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=69)
    model.fit(X, y)
    return model

def rf_strategy(ohlc: pd.DataFrame, model, frequency="daily", cost_bps: float = 0.0):
    if frequency == "daily":
        r1, r5, r10, vol_win = 1, 5, 10, 20
    elif frequency == "weekly":
        r1, r5, r10, vol_win = 1, 3, 5, 10
    elif frequency == "monthly":
        r1, r5, r10, vol_win = 1, 2, 3, 6
    else:
        raise ValueError("frequency must be 'daily', 'weekly', or 'monthly'")
    
    log_c = np.log(ohlc['close'])
    ret1 = log_c.diff(r1)
    ret5 = log_c.diff(r5)
    ret10 = log_c.diff(r10)
    vol_ratio = ohlc['volume'] / ohlc['volume'].rolling(window=vol_win).mean()
    
    features = pd.concat([ret1, ret5, ret10, vol_ratio], axis=1)
    features.columns = ['ret1', 'ret5', 'ret10', 'vol_ratio']
    features = features.dropna()
    
    X = features.to_numpy()
    pred = model.predict(X)
    pred_series = pd.Series(pred, index=features.index)
    pred_unique = pred_series.groupby(pred_series.index).first()
    pred_final = pred_unique.reindex(ohlc.index, method='ffill')
    
    signal = pred_final.apply(lambda x: 1 if x == 1 else -1)
    r = compute_forward_log_returns(ohlc['close'])
    stats, _, net, _ = evaluate(signal, r, cost_bps=cost_bps)
    return signal, stats.pf

# ------------------------------
# Walk-Forward Test Functions
# ------------------------------
def walkforward_bollinger(ohlc: pd.DataFrame, frequency="daily", window=None, num_std=1,
                           train_lookback=365*4, train_step=30, opt_start_date=pd.Timestamp("2020-01-01")):
    """
    Walk-forward test for the Bollinger Bands strategy.
    
    At each re-optimization:
      - Use the previous train_lookback bars to compute Bollinger bands,
      - Then hold the last computed signal for the next train_step bars.
    
    Returns:
      wf_signal: Series of signals for the full dataset.
    """
    if 'r' not in ohlc.columns:
        ohlc['r'] = compute_forward_log_returns(ohlc['close'])
    start_opt = ohlc.index.searchsorted(opt_start_date)
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    next_train = start_opt
    while next_train < n:
        train_start = max(0, next_train - train_lookback)
        train_df = ohlc.iloc[train_start:next_train].copy()
        signal_train, _ = bollinger_strategy(train_df, frequency=frequency, window=window, num_std=num_std)
        last_signal = signal_train.iloc[-1]
        end_segment = min(next_train + train_step, n)
        wf_signal[next_train:end_segment] = last_signal
        next_train += train_step
    wf_signal = pd.Series(wf_signal, index=ohlc.index).ffill()
    return wf_signal

def walkforward_rf(ohlc: pd.DataFrame, frequency="daily", train_lookback=365*4, train_step=30, opt_start_date=pd.Timestamp("2020-01-01")):
    """
    Walk-forward test for the Random Forest strategy.
    
    At each re-optimization:
      - Train the RF model on the previous train_lookback bars,
      - Then apply the model to generate signals for the next train_step bars.
    
    Returns:
      wf_signal: Series of signals for the full dataset.
    """
    if 'r' not in ohlc.columns:
        ohlc['r'] = compute_forward_log_returns(ohlc['close'])
    start_opt = ohlc.index.searchsorted(opt_start_date)
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    next_train = start_opt
    while next_train < n:
        train_start = max(0, next_train - train_lookback)
        train_df = ohlc.iloc[train_start:next_train].copy()
        model = train_rf(train_df, frequency=frequency)
        full_signal, _ = rf_strategy(ohlc, model, frequency=frequency)
        end_segment = min(next_train + train_step, n)
        wf_signal[next_train:end_segment] = full_signal[next_train:end_segment]
        next_train += train_step
    wf_signal = pd.Series(wf_signal, index=ohlc.index).ffill()
    return wf_signal

def walkforward_pf(ohlc: pd.DataFrame, wf_signal: pd.Series, cost_bps: float = 0.0):
    if 'r' not in ohlc.columns:
        ohlc['r'] = compute_forward_log_returns(ohlc['close'])
    stats, _, net, _ = evaluate(wf_signal, ohlc['r'], cost_bps=cost_bps)
    return stats.pf

# ------------------------------
# Walk-Forward Permutation Test Functions
# ------------------------------
def run_wf_permutation(strategy_func, ohlc, strategy_args=(), seed=0, cost_bps: float = 0.0):
    permuted = get_permutation(ohlc, start_index=0, seed=seed)
    permuted['r'] = compute_forward_log_returns(permuted['close'])
    wf_signal = strategy_func(permuted, *strategy_args)
    pf = walkforward_pf(permuted, wf_signal, cost_bps=cost_bps)
    return pf

def in_sample_wf_permutation_test(strategy_func, ohlc, strategy_args=(), real_pf=None, n_permutations=200, cost_bps: float = 0.0):
    if real_pf is None:
        raise ValueError("Please provide the real profit factor (real_pf) as a parameter.")
    perm_better_count = 1
    perm_pfs = []
    for i in tqdm(range(1, n_permutations), desc="WF Permutation Test"):
        pf = run_wf_permutation(strategy_func, ohlc, strategy_args, seed=i, cost_bps=cost_bps)
        if pf >= real_pf:  # now real_pf is provided as an argument
            perm_better_count += 1
        perm_pfs.append(pf)
    p_value = perm_better_count / n_permutations
    return perm_pfs, p_value
