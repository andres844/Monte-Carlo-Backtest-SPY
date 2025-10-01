import argparse
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from donchian import optimize_donchian, donchian_breakout, walkforward_donch
from walkforward_tree import walkforward_tree, walkforward_pf as walkforward_pf_tree
from spy_tree_strategy import train_tree, tree_strategy
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


def load_dataset(frequency: str) -> pd.DataFrame:
    files = {
        'daily': 'spy_daily_2000_2024.csv',
        'weekly': 'spy_weekly_2000_2024.csv',
        'monthly': 'spy_monthly_2000_2024.csv',
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
    for i in range(1, n_perm + 1):
        p = permute_df(is_df, variant, i, start_index, block_size)
        bl, _ = optimize_donchian(p, frequency=frequency, cost_bps=cost_bps)
        psig = donchian_breakout(p, bl)
        _, _, pnet, _ = evaluate(psig, compute_forward_log_returns(p['close']), cost_bps=cost_bps)
        pf = float((pnet[pnet > 0].sum()) / max(pnet[pnet < 0].abs().sum(), 1e-12))
        perm_pfs.append(pf)
        perm_cums.append(pnet.cumsum())

    pval = (1 + sum(x >= stats.pf for x in perm_pfs)) / (n_perm + 1)

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
    for i in range(1, n_perm):
        p = permute_df(df, variant, i, start_index, block_size)
        p['r'] = compute_forward_log_returns(p['close'])
        wf_signal_perm = walkforward_donch(p, frequency=frequency, opt_start_date=pd.Timestamp(opt_start_date))
        st, _, nnet, _ = evaluate(pd.Series(wf_signal_perm, index=p.index), p['r'], cost_bps=cost_bps)
        perm_pfs.append(st.pf)
        perm_cums.append(nnet.cumsum())

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    for i in range(n_perm):
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
    for i in range(1, n_perm):
        p = permute_df(is_df, variant, i, start_index, block_size)
        m = train_tree(p, lags)
        sig, _ = tree_strategy(p, m, lags)
        st, _, nnet, _ = evaluate(sig, compute_forward_log_returns(p['close']), cost_bps=cost_bps)
        perm_pfs.append(st.pf)

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    for i in range(n_perm):
        p = permute_df(df, variant, i, start_index, block_size)
        p['r'] = compute_forward_log_returns(p['close'])
        wf_sig_p = walkforward_tree(p, lags=lags, train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                    train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                    opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_tree(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)
        perm_cums.append((pd.Series(wf_sig_p, index=p.index) * p['r']).cumsum())

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm + 1)
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
    for i in range(1, n_perm):
        p = permute_df(is_df, variant, i, start_index, block_size)
        sig, ppf = bollinger_strategy(p, frequency=frequency, num_std=num_std, cost_bps=cost_bps)
        perm_pfs.append(ppf)

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    for i in range(1, n_perm):
        p = permute_df(is_df, variant, i, start_index, block_size)
        m = train_rf(p, frequency=frequency)
        sig, ppf = rf_strategy(p, m, frequency=frequency, cost_bps=cost_bps)
        perm_pfs.append(ppf)

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    for i in range(1, n_perm):
        p = permute_df(df, variant, i, start_index, block_size)
        wf_sig_p = walkforward_bollinger(p, frequency=frequency, window=None, num_std=num_std,
                                         train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                         train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                         opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_bb_rf(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    for i in range(1, n_perm):
        p = permute_df(df, variant, i, start_index, block_size)
        wf_sig_p = walkforward_rf(p, frequency=frequency,
                                  train_lookback=(12*4 if frequency=='monthly' else 52*4 if frequency=='weekly' else 365*4),
                                  train_step=(3 if frequency=='monthly' else 4 if frequency=='weekly' else 30),
                                  opt_start_date=pd.Timestamp(opt_start_date))
        pf_p = walkforward_pf_bb_rf(p, wf_sig_p, cost_bps=cost_bps)
        perm_pfs.append(pf_p)

    pval = (1 + sum(x >= real_pf for x in perm_pfs)) / (n_perm)
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
    ]

    ap = argparse.ArgumentParser(description='Central runner for MCPT experiments and plots')
    ap.add_argument('--tasks', nargs='+', default=['wf_donch_mcpt'], help='Tasks to run (or "all").')
    ap.add_argument('--frequency', default='monthly', choices=['daily', 'weekly', 'monthly'])
    ap.add_argument('--start', default='2000-01-01')
    ap.add_argument('--end', default='2020-01-01')
    ap.add_argument('--opt_start_date', default='2020-01-01')
    ap.add_argument('--n_permutations', type=int, default=1000)
    ap.add_argument('--cost_bps', type=float, default=0.0)
    ap.add_argument('--variant', default='bar', choices=['bar', 'block', 'grouped-month', 'grouped-dow', 'signflip'])
    ap.add_argument('--block_size', type=int, default=5)
    ap.add_argument('--start_index', type=int, default=0)
    ap.add_argument('--tree_lags', nargs=3, type=int, default=[6, 24, 168])
    ap.add_argument('--bb_num_std', type=int, default=1)
    ap.add_argument('--no_plots', action='store_true')
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
        else:
            print(f"Unknown task: {t}")


if __name__ == '__main__':
    main()
