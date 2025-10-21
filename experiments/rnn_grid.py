import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from strategies.rnn_strategy import fit_and_eval, apply_model_on


def combos():
    return [
        # --- TCNs ---
        dict(name="TCN_win128_h5_d123",  model_type="tcn", win=128, horizon=5,
             model_kwargs=dict(channels=(32, 32, 32), dilations=(1, 2, 4), p=0.10)),
        dict(name="TCN_win256_h5_d12345", model_type="tcn", win=256, horizon=5,
             model_kwargs=dict(channels=(32, 32, 32, 32, 32), dilations=(1, 2, 4, 8, 16), p=0.10)),
        dict(name="TCN_win256_h21_d12345", model_type="tcn", win=256, horizon=21,
             model_kwargs=dict(channels=(32, 32, 32, 32, 32), dilations=(1, 2, 4, 8, 16), p=0.10)),

        # --- LSTMs ---
        dict(name="LSTM_win128_h5_h64_L1", model_type="lstm", win=128, horizon=5,
             model_kwargs=dict(hidden=64, layers=1, p=0.10)),
        dict(name="LSTM_win256_h5_h128_L2", model_type="lstm", win=256, horizon=5,
             model_kwargs=dict(hidden=128, layers=2, p=0.10)),
        dict(name="LSTM_win96_h1_h64_L1",  model_type="lstm", win=96, horizon=1,
             model_kwargs=dict(hidden=64, layers=1, p=0.00)),
    ]


def main(args):
    csv = Path(f"spy_data/spy_{args.frequency}_2000_2024.csv")
    df = pd.read_csv(csv, parse_dates=['date']).set_index('date').sort_index()

    rows = []
    for cfg in combos():
        print(f"\n=== {cfg['name']} ===")
        res_tr = fit_and_eval(
            df, start='2000-01-01', end='2020-01-01',
            horizon=cfg['horizon'], win=cfg['win'], model_type=cfg['model_type'],
            cost_bps=args.cost_bps, mcpt_variant=args.mcpt_variant,
            n_permutations=args.n_permutations, block_size=args.block_size,
            seed=42, device=(None if args.device == 'auto' else args.device),
            model_kwargs=cfg['model_kwargs'],
            epochs=args.epochs, bs=args.batch_size, lr=args.lr,
        )

        res_te = apply_model_on(
            df, model=res_tr['model'], device=res_tr['device'],
            start='2020-01-01', end='2025-01-01',
            horizon=cfg['horizon'], win=cfg['win'], cost_bps=args.cost_bps,
            feature_columns=res_tr.get('feature_columns'),
        )

        row = dict(
            name=cfg['name'], model_type=cfg['model_type'],
            win=cfg['win'], horizon=cfg['horizon'],
            tr_pf=res_tr['pf_real'], tr_sharpe=res_tr['sharpe'], tr_sortino=res_tr['sortino'],
            tr_pval_pf=res_tr['pval_pf'], tr_nperm=res_tr['n_perm'],
            te_pf=res_te['pf_real'], te_sharpe=res_te['sharpe'], te_sortino=res_te['sortino'],
        )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(['te_pf', 'tr_pval_pf'], ascending=[False, True])
    out.to_csv('grid_results_rnn.csv', index=False)
    print("\nTop by OOS PF (tie-break by lower p-value in-sample):")
    print(out.head(6).to_string(index=False))
    print("\nSaved: grid_results_rnn.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--frequency', choices=['daily', 'weekly', 'monthly'], default='weekly')
    p.add_argument('--mcpt_variant', choices=['block', 'grouped-month', 'grouped-dow', 'signflip'], default='block')
    p.add_argument('--block_size', type=int, default=5)
    p.add_argument('--n_permutations', type=int, default=300)
    p.add_argument('--cost_bps', type=float, default=1.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    args = p.parse_args()
    main(args)
