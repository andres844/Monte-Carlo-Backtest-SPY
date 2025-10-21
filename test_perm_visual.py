import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bar_permute import get_permutation  # Assumes you have get_permutation defined here

def main():
    plt.style.use("dark_background")

    # 1) Load your SPY daily dataset
    df = pd.read_csv("spy_data/spy_daily_2000_2024.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)

    # 2) Split into train (2000-2020) and test (2020+)
    train_df = df[df.index < "2020-01-01"]
    test_df  = df[df.index >= "2020-01-01"]

    # 3) Plot the train data (in blue) and real test data (in red) using log-scale
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the log of the close for the training portion
    ax.plot(train_df.index, np.log(train_df["close"]), color="blue", label="Train (2000–2019)")

    # Plot the log of the close for the real test portion
    ax.plot(test_df.index, np.log(test_df["close"]), color="red", label="Real Test (2020+)")

    # 4) Generate permutations of the test data and plot them
    N_PERMUTATIONS = 20  # or however many you want
    for i in tqdm(range(N_PERMUTATIONS), desc="Permuting test data"):
        # Permute the test data's OHLC (start_index=0, seed=i for reproducibility)
        perm_test = get_permutation(test_df, start_index=0, seed=i)

        # Plot the log of the permuted close in faint white
        ax.plot(perm_test.index, np.log(perm_test["close"]), color="white", alpha=0.2)
    ax.set_xlim(pd.Timestamp("2005-01-01"), pd.Timestamp("2024-12-31"))

    # 5) Final styling
    ax.set_title("Train vs. Test with 20 Permutations of Test (SPY)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log of Close Price")
    ax.legend(loc="upper left")

    plt.show()

if __name__ == "__main__":
    main()
