import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


def compute_forward_log_returns(close: pd.Series) -> pd.Series:
    """
    Forward one-step log returns: r_t = log(C_{t+1}) - log(C_t).
    """
    log_c = np.log(close)
    return log_c.diff().shift(-1)


def profit_factor(returns: pd.Series) -> float:
    pos_sum = returns[returns > 0].sum(skipna=True)
    neg_sum = returns[returns < 0].abs().sum(skipna=True)
    if neg_sum == 0:
        return np.inf if pos_sum > 0 else 0.0
    return float(pos_sum / neg_sum)


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    sr = (mu / sd) * np.sqrt(periods_per_year)
    return float(sr)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    downside = r[r < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return np.nan
    mu = r.mean()
    return float((mu / dd) * np.sqrt(periods_per_year))


def equity_curve(returns: pd.Series) -> pd.Series:
    return returns.fillna(0).cumsum()


def max_drawdown(equity: pd.Series) -> Tuple[float, Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    cummax = equity.cummax()
    dd = equity - cummax
    mdd = dd.min()
    if len(dd) == 0 or np.isnan(mdd):
        return 0.0, None, None
    end = dd.idxmin()
    start = equity.loc[:end].idxmax()
    return float(mdd), start, end


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    eq = equity_curve(returns)
    mdd, _, _ = max_drawdown(eq)
    if mdd == 0:
        return np.nan
    cagr = returns.fillna(0).mean() * periods_per_year
    return float(-cagr / mdd)


def turnover(signal: pd.Series) -> pd.Series:
    """
    Simple turnover proxy for a discrete {-1, 0, 1} signal.
    turnover_t = 0.5 * |pos_t - pos_{t-1}|; a flip -1->+1 yields 1.0.
    """
    s = pd.Series(signal).fillna(0)
    return 0.5 * (s - s.shift(1).fillna(0)).abs()


def apply_costs(returns: pd.Series, signal: pd.Series, cost_bps: float = 0.0) -> Tuple[pd.Series, pd.Series]:
    """
    Apply transaction costs based on turnover. cost_bps is one-way cost (bps).
    Total cost per change = 2 * cost_bps bps (exit + enter).
    Returns a tuple: (net_returns, cost_series)
    """
    if cost_bps <= 0:
        zero = pd.Series(0.0, index=returns.index)
        return returns, zero
    t = turnover(pd.Series(signal, index=returns.index))
    cost_per_change = 2.0 * (cost_bps * 1e-4)
    costs = -t * cost_per_change
    return returns.add(costs, fill_value=0.0), costs


def vol_target_positions(signal: pd.Series, returns: pd.Series, target_vol: Optional[float] = None,
                         lookback: int = 20) -> pd.Series:
    """
    Scale binary signal to reach target_vol annualized using rolling realized vol.
    If target_vol is None, returns the original signal.
    """
    if not target_vol:
        return pd.Series(signal, index=returns.index)
    daily_vol = returns.rolling(lookback).std()
    ann_factor = np.sqrt(252)
    scale = (target_vol / (daily_vol * ann_factor)).replace([np.inf, -np.inf], np.nan).clip(lower=0, upper=10)
    scale = scale.fillna(method='ffill').fillna(0)
    return pd.Series(signal, index=returns.index) * scale


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float = 0.0, kurt: float = 3.0, sr_benchmark: float = 0.0) -> float:
    """
    Probabilistic Sharpe Ratio (Bailey & López de Prado, 2012).
    Approximates P(SR > SR*) given sample moments.
    """
    if n <= 1 or np.isnan(sr):
        return np.nan
    from math import sqrt
    num = (sr - sr_benchmark) * sqrt(n - 1)
    den = np.sqrt(1 - skew * sr + (kurt - 1) * (sr ** 2) / 4)
    if den == 0 or np.isnan(den):
        return np.nan
    z = num / den
    # standard normal CDF without SciPy
    from math import erf
    return float(0.5 * (1.0 + erf(z / np.sqrt(2.0))))


@dataclass
class Stats:
    pf: float
    sharpe: float
    sortino: float
    mdd: float
    calmar: float
    gross_cagr: float
    net_cagr: float
    avg_turnover: float
    cost_bps: float


def evaluate(signal: pd.Series, returns: pd.Series, cost_bps: float = 0.0,
             target_vol: Optional[float] = None, vol_lookback: int = 20,
             periods_per_year: int = 252) -> Tuple[Stats, pd.Series, pd.Series, pd.Series]:
    """
    Evaluate a strategy signal against forward returns.
    Returns (stats, gross_returns, net_returns, equity_net)
    """
    returns = returns.astype(float)
    if target_vol:
        pos = vol_target_positions(signal, returns, target_vol, vol_lookback)
    else:
        pos = pd.Series(signal, index=returns.index)

    gross = (pos * returns).rename('gross')
    net, costs = apply_costs(gross, pos, cost_bps=cost_bps)
    eq_net = equity_curve(net)
    pf = profit_factor(net)
    sr = sharpe_ratio(net, periods_per_year)
    so = sortino_ratio(net, periods_per_year)
    mdd, _, _ = max_drawdown(eq_net)
    cal = calmar_ratio(net, periods_per_year)
    avg_turn = float(turnover(pos).mean(skipna=True))

    gross_cagr = float(gross.fillna(0).mean() * periods_per_year)
    net_cagr = float(net.fillna(0).mean() * periods_per_year)
    stats = Stats(pf=pf, sharpe=sr, sortino=so, mdd=mdd, calmar=cal,
                  gross_cagr=gross_cagr, net_cagr=net_cagr,
                  avg_turnover=avg_turn, cost_bps=cost_bps)
    return stats, gross, net, eq_net
