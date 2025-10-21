import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

def train_tree(ohlc: pd.DataFrame, lags: tuple):
    """
    Train a Decision Tree using log differences at given lags.
    
    Parameters:
      ohlc: DataFrame with a 'close' column.
      lags: Tuple of three integers (lag1, lag2, lag3) specifying the differences to compute.
    
    Returns:
      A fitted DecisionTreeClassifier model.
    """
    lag1, lag2, lag3 = lags
    log_c = np.log(ohlc['close'])
    
    diff1 = log_c.diff(lag1)
    diff2 = log_c.diff(lag2)
    diff3 = log_c.diff(lag3)
    
    # Define target as the sign of the change over lag2 periods shifted by lag2 
    target = np.sign(log_c.diff(lag2).shift(-lag2))
    target = (target + 1) / 2  # transform from {-1, 1} to {0, 1}
    
    dataset = pd.concat([diff1, diff2, diff3, target], axis=1)
    dataset.columns = [f'diff{lag1}', f'diff{lag2}', f'diff{lag3}', 'target']
    
    train_data = dataset.dropna()
    train_x = train_data[[f'diff{lag1}', f'diff{lag2}', f'diff{lag3}']].to_numpy()
    train_y = train_data['target'].astype(int).to_numpy()
    
    model = DecisionTreeClassifier(min_samples_leaf=5, random_state=69)
    model.fit(train_x, train_y)
    return model

def tree_strategy(ohlc: pd.DataFrame, model, lags: tuple):
    """
    Generate trading signals using the trained decision tree model.
    
    Parameters:
      ohlc: DataFrame with a 'close' column.
      model: Trained DecisionTreeClassifier.
      lags: Tuple of integers (lag1, lag2, lag3).
    
    Returns:
      insample_signal: Series of trading signals (1 or -1).
      insample_pf: Profit factor of the resulting signal.
    """
    lag1, lag2, lag3 = lags
    log_c = np.log(ohlc['close'])
    
    diff1 = log_c.diff(lag1)
    diff2 = log_c.diff(lag2)
    diff3 = log_c.diff(lag3)
    
    dataset = pd.concat([diff1, diff2, diff3], axis=1)
    dataset.columns = [f'diff{lag1}', f'diff{lag2}', f'diff{lag3}']
    dataset = dataset.dropna()
    
    insample_pred = model.predict(dataset.to_numpy())
    insample_pred = pd.Series(insample_pred, index=dataset.index)
    
    # Instead of reindexing directly (which fails if the index has duplicates),
    # we group by the index to collapse duplicates and take the first value.
    insample_pred_unique = insample_pred.groupby(insample_pred.index).first()
    # Now reindex back to the original ohlc.index; duplicates will get filled forward.
    insample_pred_final = insample_pred_unique.reindex(ohlc.index, method='ffill')
    
    # Convert predictions to trading signals: 1 if prediction > 0, otherwise -1
    insample_signal = np.where(insample_pred_final > 0, 1, -1)
    insample_signal = pd.Series(insample_signal, index=ohlc.index)
    
    # Compute profit factor of the signal
    r = log_c.diff().shift(-1)
    rets = insample_signal * r
    pos_sum = rets[rets > 0].sum(skipna=True)
    neg_sum = rets[rets < 0].abs().sum(skipna=True)
    insample_pf = pos_sum / neg_sum if neg_sum != 0 else np.inf
    return insample_signal, insample_pf
    

if __name__ == '__main__':
    # Define dataset parameters for each frequency
    freq_params = {
        'daily': {
            'file': 'spy_data/spy_daily_2000_2024.csv',
            'lags': (6, 24, 168)
        },
        'weekly': {
            'file': 'spy_data/spy_weekly_2000_2024.csv',
            'lags': (3, 12, 48)  # example values; adjust as needed
        },
        'monthly': {
            'file': 'spy_data/spy_monthly_2000_2024.csv',
            'lags': (2, 6, 24)   # example values; adjust as needed
        }
    }
    
    results = {}
    
    for freq, params in freq_params.items():
        # Load dataset
        df = pd.read_csv(params['file'], parse_dates=['date'])
        df.set_index('date', inplace=True)
        
        # Filter in-sample period: (2000-2019)
        train_df = df[(df.index >= "2000-01-01") & (df.index < "2020-01-01")]
        
        # Train decision tree on training data
        model = train_tree(train_df, params['lags'])
        
        # Apply strategy on training data
        signal, pf = tree_strategy(train_df, model, params['lags'])
        results[freq] = pf
        print(f"{freq.capitalize()} Profit Factor: {pf:.4f}")
        
        # Plot cumulative returns for training data
        train_df = train_df.copy()  # to avoid SettingWithCopyWarning
        train_df['r'] = np.log(train_df['close']).diff().shift(-1)
        cum_returns = (signal * train_df['r']).cumsum()
        plt.style.use("dark_background")
        plt.figure(figsize=(10,6))
        plt.plot(cum_returns.index, cum_returns, label=f'{freq.capitalize()} Tree Strategy')
        plt.title(f"{freq.capitalize()} Tree Strategy Cumulative Returns (In-Sample: 2000-2019)\nProfit Factor: {pf:.4f}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Log Return")
        plt.legend()
        plt.show()
