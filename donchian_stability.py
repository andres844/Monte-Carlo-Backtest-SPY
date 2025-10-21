import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.donchian import donchian_breakout
from utils.metrics import compute_forward_log_returns, evaluate


def donchian_stability_heatmap(df: pd.DataFrame, frequency: str = 'monthly', cost_bps: float = 0.0):
    if frequency == 'daily':
        lookbacks = range(12, 169)
        periods_per_year = 252
    elif frequency == 'weekly':
        lookbacks = range(4, 53)
        periods_per_year = 52
    else:
        lookbacks = range(3, 37)
        periods_per_year = 12

    df = df.copy()
    df['r'] = compute_forward_log_returns(df['close'])

    years = sorted(set(df.index.year))
    # Only full years inside 2000-2019
    years = [y for y in years if 2000 <= y <= 2019]

    heat = np.zeros((len(lookbacks), len(years)))
    for j, y in enumerate(years):
        sub = df[(df.index.year == y)].copy()
        for i, lb in enumerate(lookbacks):
            sig = donchian_breakout(sub, lb)
            stats, _, net, _ = evaluate(sig, sub['r'], cost_bps=cost_bps, periods_per_year=periods_per_year)
            heat[i, j] = stats.pf if np.isfinite(stats.pf) else 0.0

    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6))
    im = plt.imshow(heat, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(im, label='Profit Factor (net)')
    plt.yticks(ticks=np.arange(0, len(lookbacks), max(1, len(lookbacks)//10)),
               labels=[str(lb) for lb in list(lookbacks)[::max(1, len(lookbacks)//10)]])
    plt.xticks(ticks=np.arange(len(years)), labels=[str(y) for y in years], rotation=45)
    plt.title(f'Donchian PF Stability Heatmap ({frequency})')
    plt.xlabel('Year')
    plt.ylabel('Lookback')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    freq = 'daily'
    files = {
        'daily': 'spy_data/spy_daily_2000_2024.csv',
        'weekly': 'spy_data/spy_weekly_2000_2024.csv',
        'monthly': 'spy_data/spy_monthly_2000_2024.csv',
    }
    df = pd.read_csv(files[freq], parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = df[(df.index >= '2000-01-01') & (df.index < '2020-01-01')]
    donchian_stability_heatmap(df, frequency=freq, cost_bps=0.0)
