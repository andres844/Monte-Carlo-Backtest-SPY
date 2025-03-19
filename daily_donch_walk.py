import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from donchian import walkforward_donch

# Load your daily data (ensure 'date' column is parsed and set as index)
df = pd.read_csv("spy_daily_2000_2024.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# Run walk-forward optimization using default parameters for daily frequency.
wf_signal = walkforward_donch(df, frequency='daily', opt_start_date=pd.Timestamp("2020-01-01"))

df['wf_signal'] = wf_signal
df['log_close'] = np.log(df['close'])
df['r'] = df['log_close'].diff().shift(-1)
df['wf_rets'] = df['wf_signal'] * df['r']

plt.style.use("dark_background")
plt.plot(df.index, df['wf_rets'].cumsum(), color='orange', label='Walk-Forward Donchian')
plt.xlabel("Date")
plt.ylabel("Cumulative Log Return")
plt.title("Walk-Forward Donchian (Daily)")
plt.legend()
plt.show()