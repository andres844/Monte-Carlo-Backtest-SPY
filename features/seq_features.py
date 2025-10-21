import numpy as np
import pandas as pd


def _log(x):
    return np.log(x.astype(float))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    px = df.copy()
    if not isinstance(px.index, pd.DatetimeIndex):
        px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated(keep='last')]

    px['ret1'] = _log(px['close']).diff()
    px['hl_range'] = np.log(px['high'] / px['low'])
    px['co'] = _log(px['close']) - _log(px['open'])
    px['oc_next'] = _log(px['open'].shift(-1)) - _log(px['close'])
    px['parkinson_var'] = (px['hl_range'] ** 2) / (4 * np.log(2))

    for col in ['ret1', 'co', 'oc_next', 'hl_range']:
        roll = px[col].rolling(252, min_periods=50)
        px[f'{col}_z'] = (px[col] - roll.mean()) / (roll.std() + 1e-12)

    log_vol = np.log(px['volume'].replace(0, np.nan))
    roll_vol = log_vol.rolling(252, min_periods=50)
    px['vol_z'] = (log_vol - roll_vol.mean()) / (roll_vol.std() + 1e-12)

    dow = pd.get_dummies(px.index.dayofweek, prefix='dow').astype(float)
    # Ensure a stable 7-column one-hot (dow_0..dow_6) so train/test dims match
    expected = [f'dow_{i}' for i in range(7)]
    for c in expected:
        if c not in dow.columns:
            dow[c] = 0.0
    dow = dow[expected]

    features = px[
        ['ret1_z', 'co_z', 'oc_next_z', 'hl_range_z', 'vol_z', 'parkinson_var']
    ].fillna(0.0)
    feats = pd.concat([features, dow], axis=1).fillna(0.0)
    return feats


def make_targets_close2close(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    c = np.log(df['close'].astype(float))
    y = c.shift(-horizon) - c
    return y.rename(f'fwd_ret_{horizon}')
