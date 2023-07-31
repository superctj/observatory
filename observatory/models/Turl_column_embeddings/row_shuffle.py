# Ranomly select a subset rows and shuffle the indices for them, finally
# insert them back while keeping the row positions for other rows unchanged

import argparse
import pandas as pd
import os
import numpy as np


def row_shuffle(table: pd.DataFrame, percentage: float = 1.0) -> pd.DataFrame:
    if percentage == 1:
        return table.sample(frac=1).reset_index(drop=True)
    else:
        # Get the fraction of rows to shuffle
        shuffle_frac = int(len(table) * percentage)

        # Select a random subset of indices
        indices = np.random.choice(table.index, replace=False, size=shuffle_frac)

        # Take the subset of rows to shuffle
        subset = table.loc[indices]

        # Shuffle the subset
        subset = subset.sample(frac=1)

        # Replace the original rows with the shuffled rows
        table.loc[indices] = subset.values

    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script loads csv files from a given directory, shuffles a percentage of rows in each table, and saves the shuffled tables in another directory."
    )
    parser.add_argument(
        "-r",
        "--read_dir",
        type=str,
        required=True,
        help="Directory from where the csv files will be read. For example: -r tables_csv",
    )
    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        required=True,
        help="Directory where the shuffled csv files will be saved. For example: -s shuffled_tables",
    )
    parser.add_argument(
        "-p",
        "--percentage",
        type=float,
        required=True,
        help="Percentage of the rows to shuffle. For example: -p 0.5",
    )
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    table_files = [f for f in os.listdir(args.read_dir) if f.endswith(".csv")]

    # Load tables from CSV files
    for file in table_files:
        table = pd.read_csv(f"{args.read_dir}/{file}", keep_default_na=False)
        # Shuffle rows
        table = row_shuffle(table, args.percentage)
        table.to_csv(f"{args.save_dir}/{file}", index=False, na_rep="")
