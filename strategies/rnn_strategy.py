import numpy as np, pandas as pd, torch, torch.nn as nn, random
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, Iterable
from features.seq_features import build_features, make_targets_close2close
from utils.seq_data import SequenceDataset, to_arrays
from models.seq_models import TCNModel, LSTMModel
from models.seq_models import DualHeadModel
from bar_permute import (
    get_permutation_block_bootstrap,
    get_permutation_grouped,
    get_permutation_sign_flip
)
from tqdm import tqdm

# --- simple metrics (local & robust) ---
def _nanstd(x): return np.sqrt(np.nanmean((x - np.nanmean(x))**2) + 1e-12)
def sharpe_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    mu, sd = np.nanmean(returns), _nanstd(returns)
    return float((mu / (sd + 1e-12)) * np.sqrt(ann_factor))
def sortino_ratio(returns: pd.Series, ann_factor: float = 252.0) -> float:
    mu = np.nanmean(returns)
    downside = np.sqrt(np.nanmean(np.square(np.minimum(returns, 0.0))) + 1e-12)
    return float((mu / (downside + 1e-12)) * np.sqrt(ann_factor))
def profit_factor(returns: pd.Series) -> float:
    gross_win = returns[returns > 0].sum()
    gross_loss = -returns[returns < 0].sum()
    return float(gross_win / (gross_loss + 1e-12))

def _auto_device(device: Optional[str] = None) -> str:
    if device in ("cpu","cuda","mps"): return device  # type: ignore
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def _set_seed(seed: Optional[int] = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False    # type: ignore[attr-defined]
    except Exception:
        pass

def train_model(X, y, win=128, model_type='tcn', epochs=20, bs=64, lr=1e-3, device=None, model_kwargs=None, seed_model: Optional[int] = None):
    device = _auto_device(device)
    _set_seed(seed_model)
    model_kwargs = model_kwargs or {}
    ds = SequenceDataset(X, y, win)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    if model_type == 'tcn':
        m = TCNModel(X.shape[1], **model_kwargs)
    else:
        m = LSTMModel(X.shape[1], **model_kwargs)
    m = m.to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lossf = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = m(xb); loss = lossf(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return m, device

def predict_series(m, X, y, win=128, device='cpu'):
    m.eval()
    ds = SequenceDataset(X, y, win)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            preds.append(m(xb).detach().cpu().numpy())
    if len(preds)==0: return np.array([]), None
    preds = np.concatenate(preds, axis=0)
    return preds, len(y) - len(preds)  # offset


def predict_mc(m, X, y, win=128, device='cpu', passes: int = 20):
    """
    Monte Carlo dropout predictions: returns (mu, var, offset).
    """
    ds = SequenceDataset(X, y, win)
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    preds_all = []
    was_training = m.training
    m.train()  # enable dropout
    with torch.no_grad():
        for _ in range(max(1, passes)):
            preds = []
            for xb, _ in loader:
                xb = xb.to(device)
                out = m(xb)
                preds.append(out.detach().cpu().numpy())
            preds_all.append(np.concatenate(preds, axis=0))
    if not was_training:
        m.eval()
    arr = np.stack(preds_all, axis=0)  # [P, N]
    mu = arr.mean(axis=0)
    var = arr.var(axis=0) + 1e-12
    offset = len(y) - mu.shape[0]
    return mu, var, offset

def positions_from_preds(preds, ret_vol, scale=10.0, clip=1.0):
    vol = np.maximum(ret_vol, 1e-8)
    raw = np.tanh(scale * preds / vol)
    return np.clip(raw, -clip, clip)

def _calibrate_scale(preds: np.ndarray, vol: np.ndarray, fwd: pd.Series, cost_bps: float,
                     grid: Iterable[float] = (2.0, 3.0, 5.0, 7.5, 10.0)) -> float:
    best_s, best_pf = 5.0, -np.inf
    for s in grid:
        pos = positions_from_preds(preds, vol, scale=s)
        r = compute_strategy_returns(fwd, pos, cost_bps=cost_bps)
        pf = profit_factor(r)
        if pf > best_pf:
            best_pf, best_s = pf, s
    return float(best_s)

def compute_strategy_returns(fwd_ret: pd.Series, pos: np.ndarray, cost_bps=1.0) -> pd.Series:
    pos = pd.Series(pos, index=fwd_ret.index)
    strat = pos.shift(1).fillna(0.0) * fwd_ret
    turnover = (pos.diff().abs()).fillna(0.0)
    cost = (cost_bps/1e4) * turnover
    return (strat - cost).rename('r_strategy')

def _vol_proxy(feats_df: pd.DataFrame, idx_all, offset: int) -> pd.Series:
    """
    Build a volatility proxy aligned to the same joined index used by X/y.
    idx_all is the full joined index before slicing by offset; the returned
    Series is indexed by idx_all[offset:] so it matches predictions/targets.
    """
    base = feats_df['ret1_z'].rolling(21).std()
    arr = base.reindex(idx_all).to_numpy()
    arr = np.abs(arr) + 1e-6
    return pd.Series(arr[offset:], index=idx_all[offset:])

def _mcpt_pf(model, device, df: pd.DataFrame,
             horizon: int, win: int, cost_bps: float,
             variant: str, n_permutations: int, block_size: int, seed: int,
             feature_columns: Optional[list] = None, scale: float = 10.0):
    rng = np.random.default_rng(seed)
    pf_null = []
    iterator = tqdm(range(n_permutations), desc='RNN MCPT', leave=False, disable=n_permutations<=1)
    for _ in iterator:
        if variant == 'block':
            perm = get_permutation_block_bootstrap(df, block_size=block_size, start_index=0, seed=int(rng.integers(0,1e9)))
        elif variant == 'grouped-month':
            perm = get_permutation_grouped(df, groupby='month', start_index=0, seed=int(rng.integers(0,1e9)))
        elif variant == 'grouped-dow':
            perm = get_permutation_grouped(df, groupby='dow', start_index=0, seed=int(rng.integers(0,1e9)))
        else:
            perm = get_permutation_sign_flip(df, start_index=0, seed=int(rng.integers(0,1e9)))

        f_perm = build_features(perm)
        if feature_columns is not None:
            f_perm = f_perm.reindex(columns=feature_columns, fill_value=0.0)
        y_perm = make_targets_close2close(perm, horizon=horizon)
        Xp, yp, ip = to_arrays(f_perm, y_perm)
        pr, off = predict_series(model, Xp, yp, win=win, device=device)
        if pr.size == 0: continue
        ip_used = ip[off:]
        fwd_p = pd.Series(yp[off:], index=ip_used)
        vol_p = _vol_proxy(f_perm, ip, off)
        pos_p = positions_from_preds(pr, vol_p.to_numpy(), scale=scale)
        r_p = compute_strategy_returns(fwd_p, pos_p, cost_bps=cost_bps)
        pf_null.append(profit_factor(r_p))
    return pf_null


def _mcpt_pf_mc(model, device, df: pd.DataFrame,
                horizon: int, win: int, cost_bps: float,
                variant: str, n_permutations: int, block_size: int, seed: int,
                feature_columns: Optional[list] = None, kelly_k: float = 1.0, mc_passes: int = 20,
                cap: float = 1.0) -> list:
    rng = np.random.default_rng(seed)
    pf_null = []
    iterator = tqdm(range(n_permutations), desc='RNN MCPT (MC)', leave=False, disable=n_permutations<=1)
    for _ in iterator:
        if variant == 'block':
            perm = get_permutation_block_bootstrap(df, block_size=block_size, start_index=0, seed=int(rng.integers(0,1e9)))
        elif variant == 'grouped-month':
            perm = get_permutation_grouped(df, groupby='month', start_index=0, seed=int(rng.integers(0,1e9)))
        elif variant == 'grouped-dow':
            perm = get_permutation_grouped(df, groupby='dow', start_index=0, seed=int(rng.integers(0,1e9)))
        else:
            perm = get_permutation_sign_flip(df, start_index=0, seed=int(rng.integers(0,1e9)))

        f_perm = build_features(perm)
        if feature_columns is not None:
            f_perm = f_perm.reindex(columns=feature_columns, fill_value=0.0)
        y_perm = make_targets_close2close(perm, horizon=horizon)
        Xp, yp, ip = to_arrays(f_perm, y_perm)
        mu, var, off = predict_mc(model, Xp, yp, win=win, device=device, passes=mc_passes)
        if mu.size == 0:
            continue
        ip_used = ip[off:]
        fwd_p = pd.Series(yp[off:], index=ip_used)
        f = np.clip(kelly_k * mu / (var + 1e-12), -cap, cap)
        r_p = compute_strategy_returns(fwd_p, f, cost_bps=cost_bps)
        pf_null.append(profit_factor(r_p))
    return pf_null

def fit_and_eval(df: pd.DataFrame,
                 start='2000-01-01', end='2020-01-01',
                 horizon=5, win=128, model_type='tcn',
                 cost_bps=1.0, mcpt_variant='block', n_permutations=300, block_size=5,
                 seed=42, device=None, model_kwargs=None, epochs=20, bs=64, lr=1e-3,
                 seed_model: Optional[int] = None, calibrate_scale: bool = True,
                 scale_grid: Iterable[float] = (2.0, 3.0, 5.0, 7.5, 10.0)) -> Dict:
    device = _auto_device(device)
    data = df.loc[start:end].copy()
    feats = build_features(data)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)

    model, device = train_model(X, y_arr, win=win, model_type=model_type,
                                epochs=epochs, bs=bs, lr=lr, device=device,
                                model_kwargs=model_kwargs, seed_model=seed_model)
    preds, offset = predict_series(model, X, y_arr, win=win, device=device)
    if preds.size == 0: return {'error': 'not enough data after windowing'}

    idx_used = idx[offset:]
    fwd = pd.Series(y_arr[offset:], index=idx_used, name='fwd_ret')
    vol = _vol_proxy(feats, idx, offset)
    scale = _calibrate_scale(preds, vol.to_numpy(), fwd, cost_bps, grid=scale_grid) if calibrate_scale else 10.0
    pos = positions_from_preds(preds, vol.to_numpy(), scale=scale)
    r_strat = compute_strategy_returns(fwd, pos, cost_bps=cost_bps)

    pf_real = profit_factor(r_strat)
    sr = sharpe_ratio(r_strat)
    so = sortino_ratio(r_strat)

    pf_null = _mcpt_pf(model, device, data, horizon, win, cost_bps,
                       mcpt_variant, n_permutations, block_size, seed,
                       feature_columns=feats.columns.tolist(), scale=scale)
    pval = (1.0 + np.sum(np.array(pf_null) >= pf_real)) / (1.0 + max(1, len(pf_null)))
    return {
        'pf_real': float(pf_real),
        'sharpe': float(sr),
        'sortino': float(so),
        'pval_pf': float(pval),
        'n_perm': int(len(pf_null)),
        'equity_curve': r_strat.cumsum(),
        'r_series': r_strat,       # raw returns (for OOS comparison)
        'model': model,
        'device': device,
        'feature_columns': feats.columns.tolist(),
        'scale': scale,
    }


def fit_and_eval_kelly_mc(df: pd.DataFrame,
                          start='2000-01-01', end='2020-01-01',
                          horizon=5, win=128, model_type='tcn',
                          cost_bps=1.0, mcpt_variant='block', n_permutations=300, block_size=5,
                          seed=42, device=None, model_kwargs=None, epochs=20, bs=64, lr=1e-3,
                          seed_model: Optional[int] = None, mc_passes: int = 20,
                          cap: float = 1.0, k_grid: Iterable[float] = (0.5, 1.0, 2.0, 3.0)) -> Dict:
    device = _auto_device(device)
    data = df.loc[start:end].copy()
    feats = build_features(data)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)

    model, device = train_model(X, y_arr, win=win, model_type=model_type,
                                epochs=epochs, bs=bs, lr=lr, device=device,
                                model_kwargs=model_kwargs, seed_model=seed_model)
    mu, var, offset = predict_mc(model, X, y_arr, win=win, device=device, passes=mc_passes)
    if mu.size == 0:
        return {'error': 'not enough data after windowing'}
    idx_used = idx[offset:]
    fwd = pd.Series(y_arr[offset:], index=idx_used, name='fwd_ret')
    best_k, best_pf = 1.0, -np.inf
    for k in k_grid:
        f = np.clip(k * mu / (var + 1e-12), -cap, cap)
        pf = profit_factor(compute_strategy_returns(fwd, f, cost_bps=cost_bps))
        if pf > best_pf:
            best_pf, best_k = pf, k
    f = np.clip(best_k * mu / (var + 1e-12), -cap, cap)
    r_strat = compute_strategy_returns(fwd, f, cost_bps=cost_bps)

    pf_real = profit_factor(r_strat)
    sr = sharpe_ratio(r_strat)
    so = sortino_ratio(r_strat)

    pf_null = _mcpt_pf_mc(model, device, data, horizon, win, cost_bps,
                          mcpt_variant, n_permutations, block_size, seed,
                          feature_columns=feats.columns.tolist(), kelly_k=best_k,
                          mc_passes=mc_passes, cap=cap)
    pval = (1.0 + np.sum(np.array(pf_null) >= pf_real)) / (1.0 + max(1, len(pf_null)))
    return {
        'pf_real': float(pf_real),
        'sharpe': float(sr),
        'sortino': float(so),
        'pval_pf': float(pval),
        'n_perm': int(len(pf_null)),
        'equity_curve': r_strat.cumsum(),
        'r_series': r_strat,
        'model': model,
        'device': device,
        'feature_columns': feats.columns.tolist(),
        'kelly_k': best_k,
        'mc_passes': mc_passes,
        'cap': cap,
    }


def apply_model_on_kelly_mc(df: pd.DataFrame, model, device,
                            start='2020-01-01', end='2025-01-01',
                            horizon=5, win=128, cost_bps=1.0,
                            feature_columns: Optional[list] = None,
                            kelly_k: float = 1.0, mc_passes: int = 20, cap: float = 1.0) -> Dict:
    data = df.loc[start:end].copy()
    feats = build_features(data)
    if feature_columns is not None:
        feats = feats.reindex(columns=feature_columns, fill_value=0.0)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)
    mu, var, offset = predict_mc(model, X, y_arr, win=win, device=device, passes=mc_passes)
    if mu.size == 0:
        return {'error': 'not enough data after windowing'}
    idx_used = idx[offset:]
    fwd = pd.Series(y_arr[offset:], index=idx_used, name='fwd_ret')
    f = np.clip(kelly_k * mu / (var + 1e-12), -cap, cap)
    r_strat = compute_strategy_returns(fwd, f, cost_bps=cost_bps)
    return {
        'pf_real': profit_factor(r_strat),
        'sharpe': sharpe_ratio(r_strat),
        'sortino': sortino_ratio(r_strat),
        'equity_curve': r_strat.cumsum(),
        'r_series': r_strat,
    }

def apply_model_on(df: pd.DataFrame, model, device,
                   start='2020-01-01', end='2025-01-01',
                   horizon=5, win=128, cost_bps=1.0,
                   feature_columns: Optional[list] = None,
                   scale: float = 10.0) -> Dict:
    data = df.loc[start:end].copy()
    feats = build_features(data)
    if feature_columns is not None:
        # Align to training feature space/order
        feats = feats.reindex(columns=feature_columns, fill_value=0.0)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)
    preds, offset = predict_series(model, X, y_arr, win=win, device=device)
    if preds.size == 0: return {'error': 'not enough data after windowing'}
    idx_used = idx[offset:]
    fwd = pd.Series(y_arr[offset:], index=idx_used, name='fwd_ret')
    vol = _vol_proxy(feats, idx, offset)
    pos = positions_from_preds(preds, vol.to_numpy(), scale=scale)
    r_strat = compute_strategy_returns(fwd, pos, cost_bps=cost_bps)
    return {
        'pf_real': profit_factor(r_strat),
        'sharpe': sharpe_ratio(r_strat),
        'sortino': sortino_ratio(r_strat),
        'equity_curve': r_strat.cumsum(),
        'r_series': r_strat
    }


# ----- Dual-Head Directional Kelly -----
def train_dualhead(X, y, win=128, model_type='tcn', device=None, epochs=20, bs=64, lr=1e-3, model_kwargs=None, seed_model: Optional[int] = None):
    device = _auto_device(device)
    model_kwargs = model_kwargs or {}
    if seed_model is not None:
        import random as _r
        _r.seed(seed_model); np.random.seed(seed_model); torch.manual_seed(seed_model)
    backbone = 'tcn' if model_type == 'tcn' else 'lstm'
    m = DualHeadModel(X.shape[1], backbone=backbone, **model_kwargs).to(device)
    ds = SequenceDataset(X, y, win)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()
    m.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            out = m(xb)
            p_logit = out['logit']; pos = out['pos']; neg = out['neg']
            label = (yb > 0).float(); y_pos = torch.relu(yb); y_neg = torch.relu(-yb)
            loss = bce(p_logit, label) + mse(pos, y_pos) + mse(neg, y_neg)
            opt.zero_grad(); loss.backward(); opt.step()
    return m, device


def predict_dual(m, X, y, win=128, device='cpu'):
    ds = SequenceDataset(X, y, win)
    dl = DataLoader(ds, batch_size=512, shuffle=False)
    m.eval()
    p_list, pos_list, neg_list = [], [], []
    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            out = m(xb)
            p_list.append(out['p'].detach().cpu().numpy())
            pos_list.append(out['pos'].detach().cpu().numpy())
            neg_list.append(out['neg'].detach().cpu().numpy())
    if not p_list:
        return None, None, None, None
    p = np.concatenate(p_list); rp = np.concatenate(pos_list); rn = np.concatenate(neg_list)
    off = len(y) - len(p)
    return p, rp, rn, off


def kelly_from_dual(p: np.ndarray, rpos: np.ndarray, rneg: np.ndarray, cap: float = 1.0) -> np.ndarray:
    b = rpos / (rneg + 1e-12)
    f = (b * p - (1.0 - p)) / (b + 1e-12)
    return np.clip(f, -cap, cap)


def fit_and_eval_dual_kelly(df: pd.DataFrame,
                            start='2000-01-01', end='2020-01-01',
                            horizon=5, win=128, model_type='tcn',
                            cost_bps=1.0, mcpt_variant='block', n_permutations=300, block_size=5,
                            seed=42, device=None, model_kwargs=None, epochs=20, bs=64, lr=1e-3,
                            seed_model: Optional[int] = None, cap: float = 1.0) -> Dict:
    device = _auto_device(device)
    data = df.loc[start:end].copy()
    feats = build_features(data)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)
    m, device = train_dualhead(X, y_arr, win=win, model_type=model_type, device=device, epochs=epochs, bs=bs, lr=lr, model_kwargs=model_kwargs, seed_model=seed_model)
    p, rpos, rneg, off = predict_dual(m, X, y_arr, win=win, device=device)
    if p is None:
        return {'error': 'not enough data after windowing'}
    idx_used = idx[off:]
    fwd = pd.Series(y_arr[off:], index=idx_used)
    f = kelly_from_dual(p, rpos, rneg, cap=cap)
    r_strat = compute_strategy_returns(fwd, f, cost_bps=cost_bps)
    pf_real = profit_factor(r_strat)
    sr = sharpe_ratio(r_strat); so = sortino_ratio(r_strat)

    rng = np.random.default_rng(seed)
    pf_null = []
    iterator = tqdm(range(n_permutations), desc='RNN MCPT (dual)', leave=False, disable=n_permutations<=1)
    for _ in iterator:
        if mcpt_variant == 'block':
            perm = get_permutation_block_bootstrap(data, block_size=block_size, start_index=0, seed=int(rng.integers(0,1e9)))
        elif mcpt_variant == 'grouped-month':
            perm = get_permutation_grouped(data, groupby='month', start_index=0, seed=int(rng.integers(0,1e9)))
        elif mcpt_variant == 'grouped-dow':
            perm = get_permutation_grouped(data, groupby='dow', start_index=0, seed=int(rng.integers(0,1e9)))
        else:
            perm = get_permutation_sign_flip(data, start_index=0, seed=int(rng.integers(0,1e9)))
        f_perm = build_features(perm).reindex(columns=feats.columns, fill_value=0.0)
        y_perm = make_targets_close2close(perm, horizon=horizon)
        Xp, yp, ip = to_arrays(f_perm, y_perm)
        p_p, rp_p, rn_p, off_p = predict_dual(m, Xp, yp, win=win, device=device)
        if p_p is None:
            continue
        ip_used = ip[off_p:]
        fwd_p = pd.Series(yp[off_p:], index=ip_used)
        f_p = kelly_from_dual(p_p, rp_p, rn_p, cap=cap)
        r_p = compute_strategy_returns(fwd_p, f_p, cost_bps=cost_bps)
        pf_null.append(profit_factor(r_p))
    pval = (1.0 + np.sum(np.array(pf_null) >= pf_real)) / (1.0 + max(1, len(pf_null)))
    return {
        'pf_real': float(pf_real),
        'sharpe': float(sr),
        'sortino': float(so),
        'pval_pf': float(pval),
        'n_perm': int(len(pf_null)),
        'equity_curve': r_strat.cumsum(),
        'r_series': r_strat,
        'model': m, 'device': device, 'feature_columns': feats.columns.tolist(), 'cap': cap,
    }


def apply_model_on_dual(df: pd.DataFrame, model, device,
                        start='2020-01-01', end='2025-01-01',
                        horizon=5, win=128, cost_bps=1.0,
                        feature_columns: Optional[list] = None, cap: float = 1.0) -> Dict:
    data = df.loc[start:end].copy()
    feats = build_features(data)
    if feature_columns is not None:
        feats = feats.reindex(columns=feature_columns, fill_value=0.0)
    y = make_targets_close2close(data, horizon=horizon)
    X, y_arr, idx = to_arrays(feats, y)
    p, rp, rn, off = predict_dual(model, X, y_arr, win=win, device=device)
    if p is None:
        return {'error': 'not enough data after windowing'}
    idx_used = idx[off:]
    fwd = pd.Series(y_arr[off:], index=idx_used)
    f = kelly_from_dual(p, rp, rn, cap=cap)
    r_strat = compute_strategy_returns(fwd, f, cost_bps=cost_bps)
    return {
        'pf_real': profit_factor(r_strat),
        'sharpe': sharpe_ratio(r_strat),
        'sortino': sortino_ratio(r_strat),
        'equity_curve': r_strat.cumsum(), 'r_series': r_strat
    }


def mcpt_fixed_model_on_window(model, device, df: pd.DataFrame,
                               start: str, end: str,
                               horizon: int, win: int, cost_bps: float,
                               variant: str, n_permutations: int, block_size: int, seed: int,
                               feature_columns: Optional[list] = None,
                               scale: float = 10.0) -> Tuple[float, list]:
    data = df.loc[start:end].copy()
    pf_null = _mcpt_pf(model, device, data, horizon, win, cost_bps,
                       variant, n_permutations, block_size, seed,
                       feature_columns=feature_columns, scale=scale)
    return float(np.mean(pf_null) if pf_null else np.nan), pf_null


def walkforward_rnn(df: pd.DataFrame,
                    horizon: int = 5,
                    win: int = 128,
                    model_type: str = 'tcn',
                    cost_bps: float = 1.0,
                    train_lookback: int = 365 * 4,
                    train_step: int = 30,
                    opt_start_date: str = '2020-01-01',
                    device: Optional[str] = None,
                    model_kwargs: Optional[dict] = None,
                    epochs: int = 20,
                    bs: int = 64,
                    lr: float = 1e-3,
                    seed_model: Optional[int] = None,
                    calibrate_scale: bool = True,
                    scale_grid: Iterable[float] = (2.0, 3.0, 5.0, 7.5, 10.0)) -> Dict:
    device = _auto_device(device)
    feats_full = build_features(df)
    y_full = make_targets_close2close(df, horizon=horizon)

    pos_wf = pd.Series(np.nan, index=df.index)
    start_opt = df.index.searchsorted(pd.Timestamp(opt_start_date))
    n = len(df)
    next_train = start_opt
    last_scale = 10.0
    feature_columns = None

    while next_train < n:
        train_start = max(0, next_train - train_lookback)
        train_df = df.iloc[train_start:next_train]

        feats_tr = build_features(train_df)
        y_tr = make_targets_close2close(train_df, horizon=horizon)
        X_tr, y_tr_arr, idx_tr = to_arrays(feats_tr, y_tr)
        if len(X_tr) < win:
            next_train += train_step
            continue

        model, device = train_model(X_tr, y_tr_arr, win=win, model_type=model_type,
                                    epochs=epochs, bs=bs, lr=lr, device=device,
                                    model_kwargs=model_kwargs, seed_model=seed_model)

        feature_columns = feats_tr.columns.tolist()
        feats_full_aligned = feats_full.reindex(columns=feature_columns, fill_value=0.0)
        X_full, y_full_arr, idx_full = to_arrays(feats_full_aligned, y_full)
        preds_full, off_full = predict_series(model, X_full, y_full_arr, win=win, device=device)
        if preds_full.size == 0:
            next_train += train_step
            continue
        idx_used_full = idx_full[off_full:]
        fwd_full = pd.Series(y_full_arr[off_full:], index=idx_used_full)
        vol_full = _vol_proxy(feats_full_aligned, idx_full, off_full).to_numpy()

        if calibrate_scale:
            mask_tr = (idx_used_full >= train_df.index.min()) & (idx_used_full <= train_df.index.max())
            if mask_tr.any():
                last_scale = _calibrate_scale(preds_full[mask_tr], vol_full[mask_tr], fwd_full[mask_tr], cost_bps, grid=scale_grid)

        pos_full = positions_from_preds(preds_full, vol_full, scale=last_scale)
        pos_full_ser = pd.Series(pos_full, index=idx_used_full)

        end_segment = min(next_train + train_step, n)
        seg_index = df.index[next_train:end_segment]
        pos_wf.loc[seg_index] = pos_full_ser.reindex(seg_index)

        next_train += train_step

    fwd_all = make_targets_close2close(df, horizon=horizon)
    fwd_all = fwd_all.loc[pos_wf.index]
    pos_wf = pos_wf.reindex(fwd_all.index)
    r_wf = compute_strategy_returns(fwd_all, pos_wf.to_numpy(), cost_bps=cost_bps)

    return {
        'pf_real': profit_factor(r_wf),
        'sharpe': sharpe_ratio(r_wf),
        'sortino': sortino_ratio(r_wf),
        'equity_curve': r_wf.cumsum(),
        'r_series': r_wf,
    }
