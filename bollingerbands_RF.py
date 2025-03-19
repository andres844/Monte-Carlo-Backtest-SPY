import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ------------------------------
# Strategy 1: Bollinger Bands Mean Reversion
# ------------------------------
def bollinger_strategy(ohlc: pd.DataFrame, frequency="daily", window=None, num_std=1):
    """
    Bollinger Bands Mean Reversion Strategy:
      - Computes the moving average (MA) and standard deviation (std) of 'close'
      - Defines the upper band = MA + num_std * std and lower band = MA - num_std * std
      - Signal = 1 when price < lower band (buy) and -1 when price > upper band (sell/short)
      - The signal is forward-filled.
    
    Parameters:
      ohlc      : DataFrame containing at least a 'close' column.
      frequency : One of "daily", "weekly", or "monthly". If window is None,
                  the default window is set as:
                    daily   -> 20
                    weekly  -> 10
                    monthly -> 6
      window    : (Optional) The rolling window length. If None, it is set based on frequency.
      num_std   : Number of standard deviations for the bands (default is 2).
    
    Returns:
      signal    : Series of trading signals (1 or -1).
      pf        : Profit Factor computed as (sum of positive returns) / (sum of absolute negative returns).
    """
    # Set default window if not provided
    if window is None:
        if frequency == "daily":
            window = 20
        elif frequency == "weekly":
            window = 5
        elif frequency == "monthly":
            window = 3
        else:
            raise ValueError("frequency must be one of 'daily', 'weekly', or 'monthly'")
    
    ma = ohlc['close'].rolling(window=window).mean()
    std = ohlc['close'].rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std

    signal = pd.Series(np.nan, index=ohlc.index)
    signal[ohlc['close'] < lower] = 1
    signal[ohlc['close'] > upper] = -1
    signal = signal.ffill().fillna(0)
    
    log_c = np.log(ohlc['close'])
    r = log_c.diff().shift(-1)
    strat_rets = signal * r
    pos_sum = strat_rets[strat_rets > 0].sum(skipna=True)
    neg_sum = strat_rets[strat_rets < 0].abs().sum(skipna=True)
    pf = pos_sum / (neg_sum + 1e-8)  # small constant to avoid division by zero
    return signal, pf

# ------------------------------
# Strategy 2: Random Forest–Based Strategy
# ------------------------------
def train_rf(ohlc: pd.DataFrame, frequency="daily"):
    """
    Train a Random Forest classifier on features derived from OHLC and volume.
    
    Features used depend on frequency:
      - For "daily": use 1-day, 5-day, and 10-day log returns and volume ratio with a 20-day rolling window.
      - For "weekly": use 1-week, 3-week, and 5-week log returns and volume ratio with a 10-week rolling window.
      - For "monthly": use 1-month, 2-month, and 3-month log returns and volume ratio with a 6-month rolling window.
    
    Target:
      - 1 if next period’s log return is positive, 0 otherwise.
    
    Returns:
      A fitted RandomForestClassifier model.
    """
    # Set default lags and volume rolling window based on frequency
    if frequency == "daily":
        r1, r5, r10, vol_win = 1, 5, 10, 20
    elif frequency == "weekly":
        r1, r5, r10, vol_win = 1, 3, 5, 10
    elif frequency == "monthly":
        r1, r5, r10, vol_win = 1, 2, 3, 6
    else:
        raise ValueError("frequency must be one of 'daily', 'weekly', or 'monthly'")
    
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

def rf_strategy(ohlc: pd.DataFrame, model, frequency="daily"):
    """
    Generate trading signals using the trained Random Forest model.
    
    Uses the same feature set as train_rf, with parameters automatically set based on frequency.
    
    Returns:
      signal: Series of trading signals (1 for long, -1 for short)
      pf    : Profit Factor computed as (sum of positive returns) / (sum of absolute negative returns)
    """
    if frequency == "daily":
        r1, r5, r10, vol_win = 1, 5, 10, 20
    elif frequency == "weekly":
        r1, r5, r10, vol_win = 1, 3, 5, 10
    elif frequency == "monthly":
        r1, r5, r10, vol_win = 1, 2, 3, 6
    else:
        raise ValueError("frequency must be one of 'daily', 'weekly', or 'monthly'")
    
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
    # Group by index to collapse duplicates, then forward-fill to full index
    pred_unique = pred_series.groupby(pred_series.index).first()
    pred_final = pred_unique.reindex(ohlc.index, method='ffill')
    
    signal = pred_final.apply(lambda x: 1 if x == 1 else -1)
    
    r = log_c.diff().shift(-1)
    strat_rets = signal * r
    pos_sum = strat_rets[strat_rets > 0].sum(skipna=True)
    neg_sum = strat_rets[strat_rets < 0].abs().sum(skipna=True)
    pf = pos_sum / neg_sum if neg_sum != 0 else np.inf
    return signal, pf

# ------------------------------
# Permutation Function (unchanged)
# ------------------------------
from bar_permute import get_permutation

def run_permutation(strategy_func, ohlc, strategy_args=(), seed=0):
    permuted = get_permutation(ohlc, start_index=0, seed=seed)
    permuted['r'] = np.log(permuted['close']).diff().shift(-1)
    signal, pf = strategy_func(permuted, *strategy_args)
    return pf

def in_sample_permutation_test(strategy_func, ohlc, strategy_args=(), n_permutations=200):
    perm_better_count = 1
    perm_pfs = []
    from tqdm import tqdm
    for i in tqdm(range(1, n_permutations), desc="Permutation Test"):
        pf = run_permutation(strategy_func, ohlc, strategy_args, seed=i)
        if pf >= real_pf:  # real_pf is defined in the outer scope
            perm_better_count += 1
        perm_pfs.append(pf)
    p_value = perm_better_count / n_permutations
    return perm_pfs, p_value

# ------------------------------
# Main Script: In-Sample Tests for Both Strategies (Daily/Weekly/Monthly)
# ------------------------------
if __name__ == "__main__":
    plt.style.use("dark_background")
    
    # Example: we'll run for the weekly dataset
    freq_params = {
        'daily': {
            'file': 'spy_daily_2000_2024.csv',
            'frequency': 'daily',
            'lags': None  # Not used for BB strategy; for RF, defaults in function
        },
        'weekly': {
            'file': 'spy_weekly_2000_2024.csv',
            'frequency': 'weekly',
            'lags': None
        },
        'monthly': {
            'file': 'spy_monthly_2000_2024.csv',
            'frequency': 'monthly',
            'lags': None
        }
    }
    
    # Choose one frequency to test; for example, "weekly"
    chosen_freq = "monthly"
    params = freq_params[chosen_freq]
    
    df = pd.read_csv(params['file'], parse_dates=["date"])
    df.set_index("date", inplace=True)
    df['r'] = np.log(df['close']).diff().shift(-1)
    
    # In-sample period: 2000-01-01 to 2020-01-01
    in_sample = df[(df.index >= "2000-01-01") & (df.index < "2020-01-01")].copy()
    print("Columns in dataset:", in_sample.columns)
    
    # ----- Strategy 1: Bollinger Bands Mean Reversion -----
    # For Bollinger, we now pass the frequency parameter.
    signal_bb, real_pf_bb = bollinger_strategy(in_sample, frequency=params['frequency'])
    print("Bollinger Bands Real PF:", real_pf_bb)
    
    cum_returns_bb = (signal_bb * in_sample['r']).cumsum()
    plt.figure(figsize=(10,6))
    plt.plot(cum_returns_bb.index, cum_returns_bb, color='red', label=f"BB Strategy (PF={real_pf_bb:.2f})")
    plt.title(f"Bollinger Bands Strategy ({chosen_freq.capitalize()}) Cumulative Returns (In-Sample 2000-2019)\nPF={real_pf_bb:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(loc="upper left")
    plt.show()
    
    # Permutation test for Bollinger Bands strategy
    real_pf = real_pf_bb
    perm_pfs_bb, p_value_bb = in_sample_permutation_test(bollinger_strategy, in_sample, strategy_args=(20, 2), n_permutations=10000)
    print(f"Bollinger Bands ({chosen_freq.capitalize()}) In-Sample MCPT P-Value: {p_value_bb:.4f}")
    perm_pfs_bb_clean = [pf for pf in perm_pfs_bb if np.isfinite(pf)]

    plt.figure(figsize=(10,6))
    pd.Series(perm_pfs_bb_clean).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
    plt.axvline(real_pf_bb, color="red", linestyle="dashed", linewidth=2, label="Real")
    plt.title(f"Bollinger Bands ({chosen_freq.capitalize()}) MCPT (PF={real_pf_bb:.2f}, p={p_value_bb:.4f})", color="green")
    plt.xlabel("Profit Factor")
    plt.legend()
    plt.show()
    
    # ----- Strategy 2: Random Forest–Based Strategy -----
    rf_model = train_rf(in_sample, frequency=params['frequency'])
    signal_rf, real_pf_rf = rf_strategy(in_sample, rf_model, frequency=params['frequency'])
    print("Random Forest Real PF:", real_pf_rf)
    
    cum_returns_rf = (signal_rf * in_sample['r']).cumsum()
    plt.figure(figsize=(10,6))
    plt.plot(cum_returns_rf.index, cum_returns_rf, color='orange', label=f"RF Strategy (PF={real_pf_rf:.2f})")
    plt.title(f"Random Forest Strategy ({chosen_freq.capitalize()}) Cumulative Returns (In-Sample 2000-2019)\nPF={real_pf_rf:.2f}")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Log Return")
    plt.legend(loc="upper left")
    plt.show()
    
    # Permutation test for Random Forest strategy
    real_pf = real_pf_rf
    perm_pfs_rf, p_value_rf = in_sample_permutation_test(rf_strategy, in_sample, strategy_args=(rf_model, params['frequency']), n_permutations=10000)
    print(f"Random Forest ({chosen_freq.capitalize()}) In-Sample MCPT P-Value: {p_value_rf:.4f}")
    
    plt.figure(figsize=(10,6))
    pd.Series(perm_pfs_rf).hist(bins=30, color="blue", alpha=0.7, label="Permutations")
    plt.axvline(real_pf_rf, color="red", linestyle="dashed", linewidth=2, label="Real")
    plt.title(f"Random Forest ({chosen_freq.capitalize()}) MCPT (PF={real_pf_rf:.2f}, p={p_value_rf:.4f})", color="green")
    plt.xlabel("Profit Factor")
    plt.legend()
    plt.show()