"""
analyze dataset statics for a normal table dataset
"""

import argparse
import pandas as pd
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="This script loads csv files from a given directory and analyzes the tables for certain statistics.")
    parser.add_argument('-d', '--directory', type=str, required=True, help="Directory from where the csv files will be read. For example: -d tables_csv")
    args = parser.parse_args()

    table_files = [f for f in os.listdir(args.directory) if f.endswith('.csv')]

    total_rows = 0
    total_cols = 0
    total_empty_cells = 0
    # total_non_empty_cells = 0
    total_number_cells = 0
    # total_nan_cells = 0
    total_cells = 0

    # Load tables from CSV files
    for file in table_files:
        table = pd.read_csv(f"{args.directory}/{file}", keep_default_na=False)
        total_rows += table.shape[0]
        total_cols += table.shape[1]
        total_cells += table.size
        total_empty_cells += (table.values == '').sum()
        # total_non_empty_cells += (table.values != '').sum()
        total_number_cells += table.applymap(np.isreal).sum().sum()
        # total_nan_cells += table.isna().sum().sum()

    # Compute average rows and columns
    avg_rows = total_rows / len(table_files)
    avg_cols = total_cols / len(table_files)

    # Compute proportions
    prop_empty_cells = total_empty_cells / total_cells
    # prop_non_empty_cells = total_non_empty_cells / total_cells
    prop_number_cells = total_number_cells / total_cells
    # prop_nan_cells = total_nan_cells / total_cells

    print(f'Average number of rows: {avg_rows}')
    print(f'Average number of columns: {avg_cols}')
    print(f'Proportion of cells that are empty strings: {prop_empty_cells}')
    # print(f'Proportion of cells that are non-empty strings: {prop_non_empty_cells}')
    print(f'Proportion of cells that are numbers: {prop_number_cells}')
    # print(f'Proportion of cells that are NaNs: {prop_nan_cells}')

if __name__ == "__main__":
    main()
