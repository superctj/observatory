import os
import argparse
import pandas as pd

from typing import List
from observatory.models.huggingface_models import (
    load_transformers_tokenizer,
    load_transformers_tokenizer_and_max_length,
)


def truncate_index(table, tokenizer, max_length, model_name):
    def table2colList(table):
        cols = []
        for column in table.columns:
            string_values = " ".join(table[column].astype(str).tolist())
            col_str = f"{column} {string_values}"
            cols.append(col_str)
        return cols

    def is_fit(cols, tokenizer, max_length, model_name):
        current_tokens = []

        for col in cols:
            col_tokens = tokenizer.tokenize(col)
            if model_name.startswith("t5"):
                col_tokens = ["<s>"] + col_tokens + ["</s>"]
            else:
                col_tokens = ["[CLS]"] + col_tokens + ["[SEP]"]
            if len(current_tokens) + len(col_tokens) > max_length:
                return False
            else:
                if current_tokens:
                    current_tokens = current_tokens[:-1]
                current_tokens += col_tokens
        return True

    def max_rows(table, tokenizer, max_length, model_name):
        low = 0
        high = len(table)

        while low < high:
            mid = (low + high + 1) // 2
            sample_table = table[:mid]
            cols = table2colList(sample_table)
            if is_fit(cols, tokenizer, max_length, model_name):
                low = mid
            else:
                high = mid - 1

        return low

    return max_rows(table, tokenizer, max_length, model_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate tables based on tokenizer and max length."
    )
    parser.add_argument(
        "-r",
        "--read_directory",
        type=str,
        required=True,
        help="Directory to read tables from",
    )
    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Directory to save truncated tables to",
    )
    parser.add_argument(
        "-m", "--model_name", type=str, required=True, help="Model name for tokenizer"
    )
    args = parser.parse_args()

    tokenizer, max_length = load_transformers_tokenizer_and_max_length(args.model_name)

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]
    normal_tables = [
        pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        for file in table_files
    ]

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    for i, table in enumerate(normal_tables):
        max_rows_fit = truncate_index(table, tokenizer, max_length, args.model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_table.to_csv(
            f"{args.save_directory}/table_{i}.csv", index=False, na_rep=""
        )
