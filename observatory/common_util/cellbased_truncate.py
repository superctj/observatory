import os
import argparse
import pandas as pd

from typing import List
from observatory.models.huggingface_models import (
    load_transformers_tokenizer,
    load_transformers_tokenizer_and_max_length,
)


def cellbased_truncate(table, tokenizer, max_length, model_name):
    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)
    def table2colList(table):
        cols = []
        for i in range(len(table.columns)):
            col_cells = [table.columns[i]] + table.iloc[:, i].astype(str).tolist()
            cols.append(col_cells)
        return cols

    def is_fit(cols, tokenizer, max_length, model_name):
        current_tokens = []
        # if model_name.startswith("google/tapas"):
        #     current_tokens = ["[CLS]"]  + ["[SEP]"]

        for col in cols:
            if model_name.startswith("t5"):
                current_tokens += ["<s>"]
            else:
                current_tokens += ["[CLS]"]

            for cell in col:
                cell_tokens = tokenizer.tokenize(cell)
                for token in cell_tokens:
                    current_tokens += [token]
                if len(current_tokens) > max_length:
                    return False

        if model_name.startswith("t5"):
            current_tokens += ["</s>"]
        else:
            current_tokens += ["[SEP]"]

        if len(current_tokens) > max_length:
            return False

        return True

    def max_rows(table, tokenizer, max_length, model_name):
        low = 0
        high = len(table)

        while low < high:
            mid = (low + high + 1) // 2
            sample_table = table.iloc[:mid, :]
            cols = table2colList(sample_table)
            if is_fit(cols, tokenizer, max_length, model_name):
                low = mid
            else:
                high = mid - 1

        return low

    return max_rows(table, tokenizer, max_length, model_name)
