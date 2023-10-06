from observatory.models.huggingface_models import (
    load_transformers_tokenizer,
    load_transformers_tokenizer_and_max_length,
)
import os
import argparse
import pandas as pd

from typing import List


def table2colList(table):
    cols = []
    for column in table.columns:
        # Convert column values to strings and join them with spaces
        string_values = " ".join(table[column].astype(str).tolist())
        col_str = f"{column} {string_values}"
        cols.append(col_str)
    return cols


def is_fit(sample_table, tokenizer, max_length, model_name):
    if model_name.startswith("microsoft/tapex"):
        # Initialize result
        result = [tokenizer.cls_token_id]
        
        # Tokenize each column and append to result
        for column in sample_table.columns:
            one_col_table = pd.DataFrame(sample_table[column])
            encoding = tokenizer(one_col_table, return_tensors="pt")
            column_ids = encoding['input_ids'][0].tolist()[1:-1]  # Remove cls and sep tokens
            result.extend(column_ids)
            result.append(tokenizer.cls_token_id)
            if len(result) > max_length:
                return False
    else:
        cols = table2colList(sample_table)
        current_tokens = []

        for col in cols:
            # Tokenize col without special tokens
            col_tokens = tokenizer.tokenize(col)
            # Check model name and use appropriate special tokens
            if model_name.startswith("t5"):
                # For T5, add <s> at the start and </s> at the end
                col_tokens = ["<s>"] + col_tokens + ["</s>"]
            else:
                # For other models (BERT, RoBERTa, TAPAS), add [CLS] at the start and [SEP] at the end
                col_tokens = ["[CLS]"] + col_tokens + ["[SEP]"]
            # Check if adding the new tokens would exceed the max length
            if len(current_tokens) + len(col_tokens) > max_length:
                # If so, stop and return false
                return False
            else:
                # If not, remove the last token (</s> or [SEP]) from the current tokens
                if current_tokens:
                    current_tokens = current_tokens[:-1]
                # Then concatenate the new tokens
                current_tokens += col_tokens
    return True


def column_based_truncate(table, tokenizer, max_length, model_name):

    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)
    low = 0
    high = len(table)

    while low < high:
        mid = (low + high + 1) // 2  # middle point
        sample_table = table[:mid]  # sample table with 'mid' rows
        
        if is_fit(sample_table, tokenizer, max_length, model_name):
            low = mid  # if it fits, try with more rows
        else:
            high = mid - 1  # if it doesn't fit, try with less rows

    # When low == high, we have found the maximum number of rows
    return low


def main(args):
    # Load tokenizer and max_length
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(args.model_name)

    # Read tables
    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    # Save truncated tables
    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)

    for i, table in enumerate(normal_tables):
        max_rows_fit = column_based_truncate(table, tokenizer, max_length, args.model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_table.to_csv(
            f"{args.save_directory}/table_{i}.csv", index=False, na_rep=""
        )


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
    # parser.add_argument('-l', '--max_length', type=int, default=512, help='Max length for tokenizer (default: 512)')
    args = parser.parse_args()

    main(args)
