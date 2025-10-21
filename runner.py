import argparse
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from strategies.donchian import optimize_donchian, donchian_breakout, walkforward_donch
from walkforward_tree import walkforward_tree, walkforward_pf as walkforward_pf_tree
from strategies.spy_tree_strategy import train_tree, tree_strategy
from wf_BB_RF import (
    bollinger_strategy,
    train_rf,
    rf_strategy,
    walkforward_bollinger,
    walkforward_rf,
    walkforward_pf as walkforward_pf_bb_rf,
)
from donchian_stability import donchian_stability_heatmap
from bar_permute import (
    get_permutation,
    get_permutation_block_bootstrap,
    get_permutation_grouped,
    get_permutation_sign_flip,
)
from utils.metrics import compute_forward_log_returns, evaluate
from utils.plots import plot_fan_chart


def _progress(iterable, desc: str):
    return tqdm(iterable, desc=desc, leave=False)


def load_dataset(frequency: str) -> pd.DataFrame:
    files = {
        'daily': 'spy_data/spy_daily_2000_2024.csv',
        'weekly': 'spy_data/spy_weekly_2000_2024.csv',
        'monthly': 'spy_data/spy_monthly_2000_2024.csv',
    }
    df = pd.read_csv(files[frequency], parse_dates=['date'])
    df.set_index('date', inplace=True)
    df['r'] = compute_forward_log_returns(df['close'])
    return df


def permute_df(df: pd.DataFrame, variant: str, seed: int, start_index: int, block_size: int) -> pd.DataFrame:
    if variant == 'bar':
        return get_permutation(df, start_index=start_index, seed=seed)
    elif variant == 'block':
        return get_permutation_block_bootstrap(df, block_size=block_size, start_index=start_index, seed=seed)
    elif variant == 'grouped-month':
        return get_permutation_grouped(df, groupby='month', start_index=start_index, seed=seed)
    elif variant == 'grouped-dow':
        return get_permutation_grouped(df, groupby='dow', start_index=start_index, seed=seed)
    elif variant == 'signflip':
        return get_permutation_sign_flip(df, start_index=start_index, seed=seed)
    else:
        raise ValueError(f"Unknown permutation variant: {variant}")


def run_donchian_mcpt(frequency: str, start: str, end: str, cost_bps: float, n_perm: int,
                       variant: str, start_index: int, block_size: int, show_plots: bool = True):
    plt.style.use('dark_background')
    df = load_dataset(frequency)
    is_df = df[(df.index >= start) & (df.index < end)].copy()

    best_lookback, _ = optimize_donchian(is_df, frequency=frequency, cost_bps=cost_bps)
    signal = donchian_breakout(is_df, best_lookback)
    stats, gross, net, eq = evaluate(signal, is_df['r'], cost_bps=cost_bps)
    real_cum = net.cumsum()

    perm_pfs: List[float] = []
    perm_cums: List[pd.Series] = []
    iterator = _progress(range(1, n_perm + 1), f"{frequency.upper()} Donchian perms") if n_perm else []
    for i in iterator:
        p = permute_df(is_df, variant, i, start_index, block_size)
        bl, _ = optimize_donchian(p, frequency=frequency, cost_bps=cost_bps)
        psig = donchian_breakout(p, bl)
        stats_perm, _, pnet, _ = evaluate(psig, compute_forward_log_returns(p['close']), cost_bps=cost_bps)
        perm_pfs.append(stats_perm.pf)
        perm_cums.append(pnet.cumsum())

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= stats.pf for x in perm_pfs)) / denom if denom > 0 else np.nan

    print("Donchian In-Sample MCPT:")
    print(f"  Lookback: {best_lookback}")
    print(f"  PF (net): {stats.pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    print(f"  Sharpe: {stats.sharpe:.3f}  Sortino: {stats.sortino:.3f}")
    print(f"  CAGR gross/net: {stats.gross_cagr:.2%} / {stats.net_cagr:.2%}")
    print(f"  MDD: {stats.mdd:.2%}  Avg turnover: {stats.avg_turnover:.3f}")

    if show_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fan_chart(real_cum, perm_cums, ax=ax)
        ax.set_title(
            f"Donchian MCPT ({frequency}) {start}–{end}, PF={stats.pf:.2f}, p={pval:.4f}, cost={cost_bps}bps",
            color='green')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Log Return')
        plt.show()

        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7, label='Permutations')
        plt.axvline(stats.pf, color='red', linestyle='dashed', linewidth=2, label='Real')
        plt.title(f"PF Distribution (p={pval:.4f})", color='green')
        plt.xlabel('Profit Factor')
        plt.legend()
        plt.show()


def run_donchian_stability(frequency: str, start: str, end: str, cost_bps: float, show_plots: bool = True):
    df = load_dataset(frequency)
    df = df[(df.index >= start) & (df.index < end)].copy()
    if show_plots:
        donchian_stability_heatmap(df, frequency=frequency, cost_bps=cost_bps)


def run_walkforward_donch_mcpt(frequency: str, opt_start_date: str, cost_bps: float,
                               n_perm: int, variant: str, start_index: int, block_size: int,
                               show_plots: bool = True):
    plt.style.use('dark_background')
    df = load_dataset(frequency)

    wf_signal_real = walkforward_donch(df, frequency=frequency, opt_start_date=pd.Timestamp(opt_start_date))
    stats, _, net, _ = evaluate(pd.Series(wf_signal_real, index=df.index), df['r'], cost_bps=cost_bps)
    real_pf = stats.pf
    real_cum = net.cumsum()

    perm_pfs: List[float] = []
    perm_cums: List[pd.Series] = []
    iterator = _progress(range(1, n_perm + 1), f"WF Donchian perms") if n_perm else []
    for i in iterator:
        p = permute_df(df, variant, i, start_index, block_size)
        p['r'] = compute_forward_log_returns(p['close'])
        wf_signal_perm = walkforward_donch(p, frequency=frequency, opt_start_date=pd.Timestamp(opt_start_date))
        st, _, nnet, _ = evaluate(pd.Series(wf_signal_perm, index=p.index), p['r'], cost_bps=cost_bps)
        perm_pfs.append(st.pf)
        perm_cums.append(nnet.cumsum())

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"WF Donchian PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")

    if show_plots:
        plt.figure(figsize=(10, 6))
        plot_fan_chart(real_cum, perm_cums)
        plt.title(f"WF Donchian Fan ({frequency}) PF={real_pf:.2f}, p={pval:.4f}", color='green')
        plt.xlabel('Date'); plt.ylabel('Cumulative Log Return')
        plt.legend(loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"WF Donchian MCPT PF Dist (p={pval:.4f})", color='green')
        plt.xlabel('Profit Factor'); plt.show()


def run_tree_insample_fan(frequency: str, start: str, end: str, lags: Tuple[int, int, int],
                          n_perm: int, variant: str, start_index: int, block_size: int,
                          cost_bps: float, show_plots: bool = True):
    plt.style.use('dark_background')
    df = load_dataset(frequency)
    is_df = df[(df.index >= start) & (df.index < end)].copy()
    model = train_tree(is_df, lags)
    signal, real_pf = tree_strategy(is_df, model, lags)
    stats, _, net, _ = evaluate(signal, is_df['r'], cost_bps=cost_bps)
    real_cum = net.cumsum()

    perm_cums = []
    iterator = _progress(range(n_perm), f"Tree fan perms") if n_perm else []
    for i in iterator:
        p = permute_df(is_df, variant, i, start_index, block_size)
        m = train_tree(p, lags)
        sig, _ = tree_strategy(p, m, lags)
        st, _, nnet, _ = evaluate(sig, compute_forward_log_returns(p['close']), cost_bps=cost_bps)
        perm_cums.append(nnet.cumsum())

    if show_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_fan_chart(real_cum, perm_cums, ax=ax)
        ax.set_title(f"In-Sample Tree Fan ({frequency}) PF={stats.pf:.2f}", color='green')
        ax.set_xlabel('Date'); ax.set_ylabel('Cumulative Log Return')
        plt.show()


def run_tree_insample_mcpt(frequency: str, start: str, end: str, lags: Tuple[int, int, int],
                           n_perm: int, variant: str, start_index: int, block_size: int,
                           cost_bps: float, show_plots: bool = True):
    df = load_dataset(frequency)
    is_df = df[(df.index >= start) & (df.index < end)].copy()
    model = train_tree(is_df, lags)
    signal, real_pf = tree_strategy(is_df, model, lags)
    stats, _, net, _ = evaluate(signal, is_df['r'], cost_bps=cost_bps)
    real_pf = stats.pf

    perm_pfs = []
    iterator = _progress(range(1, n_perm + 1), f"Tree MCPT perms") if n_perm else []
    for i in iterator:
        p = permute_df(is_df, variant, i, start_index, block_size)
        m = train_tree(p, lags)
        sig, _ = tree_strategy(p, m, lags)
        st, _, nnet, _ = evaluate(sig, compute_forward_log_returns(p['close']), cost_bps=cost_bps)
        perm_pfs.append(st.pf)

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"Tree In-Sample PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"Tree MCPT (p={pval:.4f})", color='green'); plt.xlabel('Profit Factor'); plt.show()


def run_wf_tree_mcpt(frequency: str, opt_start_date: str, lags: Tuple[int, int, int],
                     n_perm: int, variant: str, start_index: int, block_size: int,
                     cost_bps: float, show_plots: bool = True):
    df = load_dataset(frequency)
    wf_signal = walkforward_tree(df, lags=lags, train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                 train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                 opt_start_date=pd.Timestamp(opt_start_date))
    real_pf = walkforward_pf_tree(df, wf_signal, cost_bps=cost_bps)
    real_cum = (pd.Series(wf_signal, index=df.index) * df['r']).cumsum()

    perm_pfs = []
    perm_cums = []
    iterator = _progress(range(n_perm), "WF Tree perms") if n_perm else []
    for i in iterator:
        p = permute_df(df, variant, i, start_index, block_size)
        p['r'] = compute_forward_log_returns(p['close'])
        wf_sig_p = walkforward_tree(p, lags=lags, train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                    train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                    opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_tree(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)
        perm_cums.append((pd.Series(wf_sig_p, index=p.index) * p['r']).cumsum())

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"WF Tree PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        plot_fan_chart(real_cum, perm_cums)
        plt.title(f"WF Tree Fan ({frequency}) PF={real_pf:.2f}, p={pval:.4f}", color='green')
        plt.xlabel('Date'); plt.ylabel('Cumulative Log Return'); plt.legend(loc='upper left'); plt.show()


def run_bb_insample_mcpt(frequency: str, start: str, end: str, cost_bps: float, n_perm: int,
                          variant: str, start_index: int, block_size: int, num_std: int,
                          show_plots: bool = True):
    df = load_dataset(frequency)
    is_df = df[(df.index >= start) & (df.index < end)].copy()
    signal, pf = bollinger_strategy(is_df, frequency=frequency, num_std=num_std, cost_bps=cost_bps)
    stats, _, net, _ = evaluate(signal, is_df['r'], cost_bps=cost_bps)
    real_pf = stats.pf

    perm_pfs = []
    iterator = _progress(range(1, n_perm + 1), "Bollinger perms") if n_perm else []
    for i in iterator:
        p = permute_df(is_df, variant, i, start_index, block_size)
        sig, ppf = bollinger_strategy(p, frequency=frequency, num_std=num_std, cost_bps=cost_bps)
        perm_pfs.append(ppf)

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"Bollinger In-Sample PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"Bollinger MCPT (p={pval:.4f})", color='green'); plt.xlabel('Profit Factor'); plt.show()


def run_rf_insample_mcpt(frequency: str, start: str, end: str, cost_bps: float, n_perm: int,
                          variant: str, start_index: int, block_size: int, show_plots: bool = True):
    df = load_dataset(frequency)
    is_df = df[(df.index >= start) & (df.index < end)].copy()
    model = train_rf(is_df, frequency=frequency)
    signal, pf = rf_strategy(is_df, model, frequency=frequency, cost_bps=cost_bps)
    real_pf = pf

    perm_pfs = []
    iterator = _progress(range(1, n_perm + 1), "RF perms") if n_perm else []
    for i in iterator:
        p = permute_df(is_df, variant, i, start_index, block_size)
        m = train_rf(p, frequency=frequency)
        sig, ppf = rf_strategy(p, m, frequency=frequency, cost_bps=cost_bps)
        perm_pfs.append(ppf)

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"RF In-Sample PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"RF MCPT (p={pval:.4f})", color='green'); plt.xlabel('Profit Factor'); plt.show()


def run_wf_bb_mcpt(frequency: str, opt_start_date: str, cost_bps: float, n_perm: int,
                    variant: str, start_index: int, block_size: int, num_std: int,
                    show_plots: bool = True):
    df = load_dataset(frequency)
    wf_signal = walkforward_bollinger(df, frequency=frequency, window=None, num_std=num_std,
                                      train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                      train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                      opt_start_date=pd.Timestamp(opt_start_date))
    real_pf = walkforward_pf_bb_rf(df, wf_signal, cost_bps=cost_bps)

    perm_pfs = []
    iterator = _progress(range(1, n_perm + 1), "WF Bollinger perms") if n_perm else []
    for i in iterator:
        p = permute_df(df, variant, i, start_index, block_size)
        wf_sig_p = walkforward_bollinger(p, frequency=frequency, window=None, num_std=num_std,
                                         train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                         train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                         opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_bb_rf(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"WF Bollinger PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"WF Bollinger MCPT (p={pval:.4f})", color='green'); plt.xlabel('Profit Factor'); plt.show()


def run_wf_rf_mcpt(frequency: str, opt_start_date: str, cost_bps: float, n_perm: int,
                    variant: str, start_index: int, block_size: int,
                    show_plots: bool = True):
    df = load_dataset(frequency)
    wf_signal = walkforward_rf(df, frequency=frequency,
                               train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                               train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                               opt_start_date=pd.Timestamp(opt_start_date))
    real_pf = walkforward_pf_bb_rf(df, wf_signal, cost_bps=cost_bps)

    perm_pfs = []
    iterator = _progress(range(1, n_perm + 1), "WF RF perms") if n_perm else []
    for i in iterator:
        p = permute_df(df, variant, i, start_index, block_size)
        wf_sig_p = walkforward_rf(p, frequency=frequency,
                                  train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                  train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                  opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_bb_rf(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)

    denom = len(perm_pfs) + 1
    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / denom if denom > 0 else np.nan
    print(f"WF RF PF={real_pf:.3f}  p={pval:.4f}  cost={cost_bps}bps")
    if show_plots:
        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))
        pd.Series(perm_pfs).hist(bins=30, color='blue', alpha=0.7)
        plt.axvline(real_pf, color='red', linestyle='dashed', linewidth=2)
        plt.title(f"WF RF MCPT (p={pval:.4f})", color='green'); plt.xlabel('Profit Factor'); plt.show()


def main():
    tasks_all = [
        'donchian_mcpt',
        'donchian_stability',
        'wf_donch_mcpt',
        'tree_insample_fan',
        'tree_insample_mcpt',
        'wf_tree_mcpt',
        'bb_insample_mcpt',
        'rf_insample_mcpt',
        'wf_bb_mcpt',
        'wf_rf_mcpt',
        'rnn_insample_mcpt',
        'wf_rnn_mcpt',
        'wf_rnn_retrain',
        'rnn_grid',
        'rnn_mc_kelly',
        'wf_rnn_mc_kelly',
        'rnn_dual_kelly',
        'wf_rnn_dual_kelly',
    ]

    ap = argparse.ArgumentParser(description='Central runner for MCPT experiments and plots')
    ap.add_argument('--tasks', nargs='+', default=[], help='Tasks to run (or "all").')
    ap.add_argument('--frequency', default='daily', choices=['daily', 'weekly', 'monthly'])
    ap.add_argument('--start', default='2000-01-01')
    ap.add_argument('--end', default='2020-01-01')
    ap.add_argument('--opt_start_date', default='2020-01-01')
    ap.add_argument('--n_permutations', type=int, default=1000)
    ap.add_argument('--cost_bps', type=float, default=0.25)
    ap.add_argument('--variant', default='bar', choices=['bar', 'block', 'grouped-month', 'grouped-dow', 'signflip'])
    ap.add_argument('--block_size', type=int, default=5)
    ap.add_argument('--start_index', type=int, default=0)
    ap.add_argument('--tree_lags', nargs=3, type=int, default=[6, 24, 168])
    ap.add_argument('--bb_num_std', type=int, default=1)
    ap.add_argument('--model_type', choices=['tcn', 'lstm'], default='tcn')
    ap.add_argument('--horizon', type=int, default=5)
    ap.add_argument('--win', type=int, default=128)
    ap.add_argument('--mcpt_variant', default='block', choices=['block', 'grouped-month', 'grouped-dow', 'signflip'])
    ap.add_argument('--no_plots', action='store_true')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed_model', type=int, default=42)
    ap.add_argument('--no_calibrate', action='store_true')
    ap.add_argument('--train_lookback', type=int, default=365*4)
    ap.add_argument('--train_step', type=int, default=30)
    ap.add_argument('--mc_passes', type=int, default=20)
    ap.add_argument('--kelly_cap', type=float, default=1.0)
    args = ap.parse_args()

    tasks = tasks_all if (len(args.tasks) == 1 and args.tasks[0].lower() == 'all') else args.tasks
    show_plots = not args.no_plots
    lags = tuple(args.tree_lags)  # type: ignore

    for t in tasks:
        if t == 'donchian_mcpt':
            run_donchian_mcpt(args.frequency, args.start, args.end, args.cost_bps,
                               args.n_permutations, args.variant, args.start_index, args.block_size, show_plots)
        elif t == 'donchian_stability':
            run_donchian_stability(args.frequency, args.start, args.end, args.cost_bps, show_plots)
        elif t == 'wf_donch_mcpt':
            run_walkforward_donch_mcpt(args.frequency, args.opt_start_date, args.cost_bps,
                                       args.n_permutations, args.variant, args.start_index, args.block_size,
                                       show_plots)
        elif t == 'tree_insample_fan':
            run_tree_insample_fan(args.frequency, args.start, args.end, lags,
                                  args.n_permutations, args.variant, args.start_index, args.block_size,
                                  args.cost_bps, show_plots)
        elif t == 'tree_insample_mcpt':
            run_tree_insample_mcpt(args.frequency, args.start, args.end, lags,
                                   args.n_permutations, args.variant, args.start_index, args.block_size,
                                   args.cost_bps, show_plots)
        elif t == 'wf_tree_mcpt':
            run_wf_tree_mcpt(args.frequency, args.opt_start_date, lags,
                             args.n_permutations, args.variant, args.start_index, args.block_size,
                             args.cost_bps, show_plots)
        elif t == 'bb_insample_mcpt':
            run_bb_insample_mcpt(args.frequency, args.start, args.end, args.cost_bps, args.n_permutations,
                                 args.variant, args.start_index, args.block_size, args.bb_num_std, show_plots)
        elif t == 'rf_insample_mcpt':
            run_rf_insample_mcpt(args.frequency, args.start, args.end, args.cost_bps, args.n_permutations,
                                 args.variant, args.start_index, args.block_size, show_plots)
        elif t == 'wf_bb_mcpt':
            run_wf_bb_mcpt(args.frequency, args.opt_start_date, args.cost_bps, args.n_permutations,
                           args.variant, args.start_index, args.block_size, args.bb_num_std, show_plots)
        elif t == 'wf_rf_mcpt':
            run_wf_rf_mcpt(args.frequency, args.opt_start_date, args.cost_bps, args.n_permutations,
                           args.variant, args.start_index, args.block_size, show_plots)
        elif t == 'rnn_insample_mcpt':
            from strategies.rnn_strategy import fit_and_eval

            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res = fit_and_eval(
                df,
                start=args.start or '2000-01-01',
                end=args.end or '2020-01-01',
                horizon=args.horizon,
                win=args.win,
                model_type=args.model_type,
                cost_bps=args.cost_bps,
                mcpt_variant=args.mcpt_variant,
                n_permutations=args.n_permutations,
                block_size=args.block_size or 5,
                seed=42,
                epochs=args.epochs,
                bs=args.batch_size,
                lr=args.lr,
                device=(None if args.device=='auto' else args.device),
                seed_model=args.seed_model,
                calibrate_scale=(not args.no_calibrate),
            )
            if 'error' in res:
                print('RNN error:', res['error'])
                continue
            print('PF:', res.get('pf_real'), 'Sharpe:', res.get('sharpe'), 'Sortino:', res.get('sortino'), 'p-value:', res.get('pval_pf'))
            if show_plots and 'equity_curve' in res and isinstance(res['equity_curve'], pd.Series):
                plt.style.use('dark_background')
                plt.figure(figsize=(10, 6))
                res['equity_curve'].plot(title=f"RNN {args.model_type} equity (in-sample)")
                plt.xlabel('Date')
                plt.ylabel('Cumulative Log Return')
                plt.grid(False)
                plt.tight_layout()
                plt.show()
        elif t == 'wf_rnn_mcpt':
            from strategies.rnn_strategy import fit_and_eval, apply_model_on, mcpt_fixed_model_on_window

            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res_tr = fit_and_eval(
                df,
                start='2000-01-01',
                end='2020-01-01',
                horizon=args.horizon,
                win=args.win,
                model_type=args.model_type,
                cost_bps=args.cost_bps,
                mcpt_variant=args.mcpt_variant,
                n_permutations=args.n_permutations,
                block_size=args.block_size or 5,
                epochs=args.epochs,
                bs=args.batch_size,
                lr=args.lr,
                device=(None if args.device=='auto' else args.device),
                seed_model=args.seed_model,
                calibrate_scale=(not args.no_calibrate),
            )
            res_te = apply_model_on(
                df,
                model=res_tr.get('model'),
                device=res_tr.get('device'),
                start='2020-01-01',
                end='2025-01-01',
                horizon=args.horizon,
                win=args.win,
                cost_bps=args.cost_bps,
                feature_columns=res_tr.get('feature_columns'),
                scale=res_tr.get('scale', 10.0),
            )
            # OOS MCPT with the fixed model on the test window
            _, pf_null_te = mcpt_fixed_model_on_window(
                res_tr.get('model'), res_tr.get('device'), df,
                start='2020-01-01', end='2025-01-01',
                horizon=args.horizon, win=args.win, cost_bps=args.cost_bps,
                variant=args.mcpt_variant, n_permutations=args.n_permutations,
                block_size=args.block_size or 5, seed=42,
                feature_columns=res_tr.get('feature_columns'),
                scale=res_tr.get('scale', 10.0),
            )
            te_pf = res_te.get('pf_real')
            te_pval = (1.0 + sum(x >= te_pf for x in pf_null_te)) / (1.0 + max(1, len(pf_null_te))) if pf_null_te else float('nan')
            if 'error' in res_tr:
                print('Train error:', res_tr['error'])
            if 'error' in res_te:
                print('Test error:', res_te['error'])
            print("Train PF:", res_tr.get('pf_real'), "Train p:", res_tr.get('pval_pf'), "Test PF:", te_pf, "Test p:", te_pval)
            if show_plots:
                for label, res_x in [('Train', res_tr), ('Test', res_te)]:
                    eq = res_x.get('equity_curve')
                    if not isinstance(eq, pd.Series):
                        continue
                    plt.style.use('dark_background')
                    plt.figure(figsize=(10, 6))
                    eq.plot(title=f"RNN {args.model_type} equity ({label.lower()})")
                    plt.xlabel('Date')
                    plt.ylabel('Cumulative Log Return')
                    plt.grid(False)
                    plt.tight_layout()
                    plt.show()
        elif t == 'rnn_grid':
            # Run the compact grid search experiment script
            from types import SimpleNamespace
            from experiments.rnn_grid import main as rnn_grid_main
            ns = SimpleNamespace(
                frequency=args.frequency,
                mcpt_variant=args.mcpt_variant,
                block_size=args.block_size,
                n_permutations=args.n_permutations,
                cost_bps=args.cost_bps,
                device=args.device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
            )
            rnn_grid_main(ns)
        elif t == 'wf_rnn_retrain':
            from strategies.rnn_strategy import walkforward_rnn
            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res = walkforward_rnn(
                df,
                horizon=args.horizon,
                win=args.win,
                model_type=args.model_type,
                cost_bps=args.cost_bps,
                train_lookback=args.train_lookback,
                train_step=args.train_step,
                opt_start_date=args.opt_start_date,
                device=(None if args.device=='auto' else args.device),
                epochs=args.epochs,
                bs=args.batch_size,
                lr=args.lr,
                seed_model=args.seed_model,
                calibrate_scale=(not args.no_calibrate),
            )
            print(f"WF-RNN PF={res.get('pf_real'):.3f} Sharpe={res.get('sharpe'):.2f}")
            if show_plots and isinstance(res.get('equity_curve'), pd.Series):
                plt.style.use('dark_background')
                res['equity_curve'].plot(title=f"RNN {args.model_type} equity (walk-forward)")
                plt.xlabel('Date'); plt.ylabel('Cumulative Log Return'); plt.grid(False); plt.show()
        elif t == 'rnn_mc_kelly':
            from strategies.rnn_strategy import fit_and_eval_kelly_mc
            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res = fit_and_eval_kelly_mc(
                df, start=args.start, end=args.end, horizon=args.horizon, win=args.win, model_type=args.model_type,
                cost_bps=args.cost_bps, mcpt_variant=args.mcpt_variant, n_permutations=args.n_permutations,
                block_size=args.block_size or 5, seed=42, device=(None if args.device=='auto' else args.device),
                model_kwargs=None, epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed_model=args.seed_model,
                mc_passes=args.mc_passes, cap=args.kelly_cap
            )
            print('PF:', res.get('pf_real'), 'p:', res.get('pval_pf'))
            if show_plots and isinstance(res.get('equity_curve'), pd.Series):
                plt.style.use('dark_background'); res['equity_curve'].plot(title='MC-Dropout Kelly (in-sample)'); plt.show()
        elif t == 'wf_rnn_mc_kelly':
            from strategies.rnn_strategy import fit_and_eval_kelly_mc, apply_model_on_kelly_mc
            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res_tr = fit_and_eval_kelly_mc(
                df, start='2000-01-01', end='2020-01-01', horizon=args.horizon, win=args.win, model_type=args.model_type,
                cost_bps=args.cost_bps, mcpt_variant=args.mcpt_variant, n_permutations=args.n_permutations,
                block_size=args.block_size or 5, seed=42, device=(None if args.device=='auto' else args.device),
                epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed_model=args.seed_model,
                mc_passes=args.mc_passes, cap=args.kelly_cap
            )
            res_te = apply_model_on_kelly_mc(
                df, model=res_tr.get('model'), device=res_tr.get('device'), start='2020-01-01', end='2025-01-01',
                horizon=args.horizon, win=args.win, cost_bps=args.cost_bps, feature_columns=res_tr.get('feature_columns'),
                kelly_k=res_tr.get('kelly_k', 1.0), mc_passes=args.mc_passes, cap=args.kelly_cap
            )
            print('Train PF:', res_tr.get('pf_real'), 'Test PF:', res_te.get('pf_real'))
            if show_plots and isinstance(res_te.get('equity_curve'), pd.Series):
                plt.style.use('dark_background'); res_te['equity_curve'].plot(title='MC-Dropout Kelly (test)'); plt.show()
        elif t == 'rnn_dual_kelly':
            from strategies.rnn_strategy import fit_and_eval_dual_kelly
            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res = fit_and_eval_dual_kelly(
                df, start=args.start, end=args.end, horizon=args.horizon, win=args.win, model_type=args.model_type,
                cost_bps=args.cost_bps, mcpt_variant=args.mcpt_variant, n_permutations=args.n_permutations,
                block_size=args.block_size or 5, seed=42, device=(None if args.device=='auto' else args.device),
                epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed_model=args.seed_model, cap=args.kelly_cap
            )
            print('PF:', res.get('pf_real'), 'p:', res.get('pval_pf'))
            if show_plots and isinstance(res.get('equity_curve'), pd.Series):
                plt.style.use('dark_background'); res['equity_curve'].plot(title='Dual-Head Kelly (in-sample)'); plt.show()
        elif t == 'wf_rnn_dual_kelly':
            from strategies.rnn_strategy import fit_and_eval_dual_kelly, apply_model_on_dual
            df = pd.read_csv(f"spy_data/spy_{args.frequency}_2000_2024.csv", parse_dates=['date']).set_index('date')
            res_tr = fit_and_eval_dual_kelly(
                df, start='2000-01-01', end='2020-01-01', horizon=args.horizon, win=args.win, model_type=args.model_type,
                cost_bps=args.cost_bps, mcpt_variant=args.mcpt_variant, n_permutations=args.n_permutations,
                block_size=args.block_size or 5, seed=42, device=(None if args.device=='auto' else args.device),
                epochs=args.epochs, bs=args.batch_size, lr=args.lr, seed_model=args.seed_model, cap=args.kelly_cap
            )
            res_te = apply_model_on_dual(
                df, model=res_tr.get('model'), device=res_tr.get('device'), start='2020-01-01', end='2025-01-01',
                horizon=args.horizon, win=args.win, cost_bps=args.cost_bps, feature_columns=res_tr.get('feature_columns'), cap=args.kelly_cap
            )
            print('Train PF:', res_tr.get('pf_real'), 'Test PF:', res_te.get('pf_real'))
            if show_plots and isinstance(res_te.get('equity_curve'), pd.Series):
                plt.style.use('dark_background'); res_te['equity_curve'].plot(title='Dual-Head Kelly (test)'); plt.show()
        else:
            print(f"Unknown task: {t}")


if __name__ == '__main__':
    main()
