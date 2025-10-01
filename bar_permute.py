import numpy as np
import pandas as pd
from typing import List, Union, Optional, Literal

def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: Optional[int] = None,
):
    """
    Bar-wise permutation that shuffles the gap (open) sequence separately from
    intrabar (high/low/close) sequences after a fixed start_index. Uses
    numpy's Generator for reproducibility.

    Note: Extra (non-price) columns are copied from the original and not
    permuted. If you need extras permuted in sync with day reordering, consider
    using block/grouped permutations provided below.
    """
    assert start_index >= 0
    rng = np.random.default_rng(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    # We'll only process the price columns; extra columns will be added later.
    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        # Process only price columns
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']])
        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        # Calculate differences for gap and intrabar moves
        r_o = (log_bars['open'] - log_bars['close'].shift()).to_numpy()
        r_h = (log_bars['high'] - log_bars['open']).to_numpy()
        r_l = (log_bars['low'] - log_bars['open']).to_numpy()
        r_c = (log_bars['close'] - log_bars['open']).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)
    # Shuffle intrabar relative values (high/low/close)
    perm1 = rng.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]
    # Shuffle gap (open) separately
    perm2 = rng.permutation(idx)
    relative_open = relative_open[:, perm2]

    perm_ohlc = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))
        # Original price log values
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        # Create DataFrame for the permuted prices
        perm_df = pd.DataFrame(perm_bars, index=time_index, columns=['open', 'high', 'low', 'close'])
        # Reattach any extra columns (like volume, pct_chg, etc.) from the original reg_bars
        for col in reg_bars.columns:
            if col not in ['open', 'high', 'low', 'close']:
                # Use the original column values (assumed to be already aligned with time_index)
                perm_df[col] = reg_bars[col].values
        perm_ohlc.append(perm_df)

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]


def get_permutation_block_bootstrap(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    block_size: int = 5,
    start_index: int = 0,
    seed: Optional[int] = None,
    extras: Literal['copy', 'sync', 'drop'] = 'copy',
    extra_cols: Optional[List[str]] = None,
):
    """
    Stationary block bootstrap permutation over day-level bars. Preserves short-term
    autocorrelation by sampling blocks of length `block_size` from the permutable
    region [start_index+1 : end]. The same reordering is applied to open/gap and
    intrabar moves, producing a coherent day sequence.

    extras:
      - 'copy': copy non-price columns from original (default)
      - 'sync': reorder specified extra_cols using the same day order
      - 'drop': drop non-price columns
    """
    assert start_index >= 0 and block_size >= 1
    rng = np.random.default_rng(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        markets = ohlc
    else:
        time_index = ohlc.index
        markets = [ohlc]

    n_bars = len(markets[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    if perm_n <= 0:
        return ohlc

    # Build a day-level index order using block bootstrap
    base_idx = np.arange(perm_n)
    num_blocks = int(np.ceil(perm_n / block_size))
    starts = rng.integers(0, perm_n - block_size + 1, size=num_blocks)
    order = np.concatenate([np.arange(s, s + block_size) for s in starts])[:perm_n]

    out = []
    for reg_bars in markets:
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy()
        # Extract relative moves
        r_o = (log_bars[:, 0] - np.r_[np.nan, log_bars[:-1, 3]])  # gap vs prev close
        r_h = (log_bars[:, 1] - log_bars[:, 0])
        r_l = (log_bars[:, 2] - log_bars[:, 0])
        r_c = (log_bars[:, 3] - log_bars[:, 0])

        # Day-level permutation order in permutable region
        r_o_perm = r_o[perm_index:][order]
        r_h_perm = r_h[perm_index:][order]
        r_l_perm = r_l[perm_index:][order]
        r_c_perm = r_c[perm_index:][order]

        # Reconstruct log bars
        perm = np.zeros_like(log_bars)
        perm[:start_index, :] = log_bars[:start_index, :]
        perm[start_index, :] = log_bars[start_index, :]
        prev_close = perm[start_index, 3]
        for k in range(perm_n):
            i = perm_index + k
            lo = prev_close + r_o_perm[k]
            perm[i, 0] = lo
            perm[i, 1] = lo + r_h_perm[k]
            perm[i, 2] = lo + r_l_perm[k]
            perm[i, 3] = lo + r_c_perm[k]
            prev_close = perm[i, 3]

        perm = np.exp(perm)
        perm_df = pd.DataFrame(perm, index=time_index, columns=['open', 'high', 'low', 'close'])

        if extras == 'copy':
            for col in reg_bars.columns:
                if col not in ['open', 'high', 'low', 'close']:
                    perm_df[col] = reg_bars[col].values
        elif extras == 'sync':
            cols = [c for c in (extra_cols or []) if c in reg_bars.columns]
            for col in cols:
                arr = reg_bars[col].to_numpy().copy()
                arr_perm = arr.copy()
                arr_perm[perm_index:] = arr[perm_index:][order]
                perm_df[col] = arr_perm
        # else: drop extras

        out.append(perm_df)

    return out if len(out) > 1 else out[0]


def get_permutation_grouped(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    groupby: Literal['dow', 'month'] = 'month',
    start_index: int = 0,
    seed: Optional[int] = None,
    extras: Literal['copy', 'sync', 'drop'] = 'copy',
    extra_cols: Optional[List[str]] = None,
):
    """
    Permute within calendar groups (month or day-of-week). For each group in the
    permutable region, shuffle day order; keeps group-level seasonality and mixes
    days within groups. Applies a single day-level permutation to open/intrabar.
    """
    assert start_index >= 0
    rng = np.random.default_rng(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        markets = ohlc
    else:
        time_index = ohlc.index
        markets = [ohlc]

    n_bars = len(markets[0])
    perm_index = start_index + 1
    if perm_index >= n_bars:
        return ohlc

    idx = pd.Index(time_index)
    region = idx[perm_index:]
    if groupby == 'dow':
        keys = region.dayofweek
    else:
        keys = region.month

    # Build a permutation order by concatenating within-group shuffles
    order_parts = []
    for key in pd.Series(np.arange(len(region)), index=region).groupby(keys).indices.values():
        local = np.array(list(key))
        order_parts.append(rng.permutation(local))
    order = np.concatenate(order_parts)

    out = []
    for reg_bars in markets:
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy()
        r_o = (log_bars[:, 0] - np.r_[np.nan, log_bars[:-1, 3]])
        r_h = (log_bars[:, 1] - log_bars[:, 0])
        r_l = (log_bars[:, 2] - log_bars[:, 0])
        r_c = (log_bars[:, 3] - log_bars[:, 0])

        r_o_perm = r_o[perm_index:][order]
        r_h_perm = r_h[perm_index:][order]
        r_l_perm = r_l[perm_index:][order]
        r_c_perm = r_c[perm_index:][order]

        perm = np.zeros_like(log_bars)
        perm[:start_index, :] = log_bars[:start_index, :]
        perm[start_index, :] = log_bars[start_index, :]
        prev_close = perm[start_index, 3]
        for k in range(len(order)):
            i = perm_index + k
            lo = prev_close + r_o_perm[k]
            perm[i, 0] = lo
            perm[i, 1] = lo + r_h_perm[k]
            perm[i, 2] = lo + r_l_perm[k]
            perm[i, 3] = lo + r_c_perm[k]
            prev_close = perm[i, 3]

        perm = np.exp(perm)
        perm_df = pd.DataFrame(perm, index=time_index, columns=['open', 'high', 'low', 'close'])

        if extras == 'copy':
            for col in reg_bars.columns:
                if col not in ['open', 'high', 'low', 'close']:
                    perm_df[col] = reg_bars[col].values
        elif extras == 'sync':
            cols = [c for c in (extra_cols or []) if c in reg_bars.columns]
            for col in cols:
                arr = reg_bars[col].to_numpy().copy()
                arr_perm = arr.copy()
                arr_perm[perm_index:] = arr[perm_index:][order]
                perm_df[col] = arr_perm

        out.append(perm_df)

    return out if len(out) > 1 else out[0]


def get_permutation_sign_flip(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]],
    start_index: int = 0,
    seed: Optional[int] = None,
):
    """
    Sign-flip null: preserves absolute bar magnitudes but flips the sign (direction)
    of gap and intrabar moves at random. For a negative flip, high/low legs are
    swapped appropriately to maintain bar geometry.
    """
    rng = np.random.default_rng(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        markets = ohlc
    else:
        time_index = ohlc.index
        markets = [ohlc]

    n_bars = len(markets[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    if perm_n <= 0:
        return ohlc

    signs = rng.choice([-1.0, 1.0], size=perm_n)

    out = []
    for reg_bars in markets:
        log_bars = np.log(reg_bars[['open', 'high', 'low', 'close']]).to_numpy()
        r_o = (log_bars[:, 0] - np.r_[np.nan, log_bars[:-1, 3]])
        r_h = (log_bars[:, 1] - log_bars[:, 0])
        r_l = (log_bars[:, 2] - log_bars[:, 0])
        r_c = (log_bars[:, 3] - log_bars[:, 0])

        r_o_perm = r_o.copy()
        r_h_perm = r_h.copy()
        r_l_perm = r_l.copy()
        r_c_perm = r_c.copy()

        # Apply sign flips to the permutable region
        for k in range(perm_n):
            s = signs[k]
            i = perm_index + k
            if s < 0:
                r_o_perm[i] = -r_o[i]
                r_c_perm[i] = -r_c[i]
                # Swap high/low legs with sign change
                r_h_perm[i] = -r_l[i]
                r_l_perm[i] = -r_h[i]

        perm = np.zeros_like(log_bars)
        perm[:start_index, :] = log_bars[:start_index, :]
        perm[start_index, :] = log_bars[start_index, :]
        prev_close = perm[start_index, 3]
        for i in range(perm_index, n_bars):
            lo = prev_close + r_o_perm[i]
            perm[i, 0] = lo
            perm[i, 1] = lo + r_h_perm[i]
            perm[i, 2] = lo + r_l_perm[i]
            perm[i, 3] = lo + r_c_perm[i]
            prev_close = perm[i, 3]

        perm = np.exp(perm)
        perm_df = pd.DataFrame(perm, index=time_index, columns=['open', 'high', 'low', 'close'])
        # Copy extras
        for col in reg_bars.columns:
            if col not in ['open', 'high', 'low', 'close']:
                perm_df[col] = reg_bars[col].values
        out.append(perm_df)

    return out if len(out) > 1 else out[0]
