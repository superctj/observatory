import argparse
import math
import os
import shutil
import sys
import time
import unittest

from collections import Counter

import pandas as pd
import torch

from observatory.common_util.truncate import truncate_index
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.properties.Column_Order_Insignificance.evaluate_col_shuffle import (  # noqa: E501
    fisher_yates_shuffle,
    get_permutations,
    shuffle_df_columns,
    process_table_wrapper,
)


def test_fisher_yates_shuffle():
    sequence = list(range(10))
    shuffled_seq = fisher_yates_shuffle(sequence)
    assert Counter(sequence) == Counter(shuffled_seq)


def test_get_permutations():
    n = 3
    m = 1
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 5
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 6
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 3
    m = 7
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    n = 10
    m = 10
    permutations = get_permutations(n, m)
    assert len(permutations) == min(m + 1, math.factorial(n))

    uniq_permut = set()
    for permut in permutations:
        permut = tuple(permut)

        if permut not in uniq_permut:
            uniq_permut.add(permut)
        else:
            raise ValueError(f"Permutation {permut} is not unique.")


def test_builtin_and_torch_mean():
    x = list(range(1000))

    start = time.time()
    builtin_mean = sum(x) / len(x)
    end = time.time()
    builtin_time = end - start

    start = time.time()
    torch_mean = torch.mean(torch.FloatTensor(x)).item()
    end = time.time()
    torch_time = end - start

    assert builtin_mean == torch_mean
    print(f"\nBuilt-in average computing time: {builtin_time} s")
    print(f"Torch average computing time: {torch_time} s")
    assert builtin_time < torch_time


class HuggingFaceModels(unittest.TestCase):
    def setUp(self):
        save_directory = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "outputs"
        )
        args_dict = {
            "save_directory": save_directory,
            "num_shuffles": 5,
            "batch_size": 32,
        }
        self.args = argparse.Namespace(**args_dict)

        self.test_data_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "data/web_tables_csv"
        )
        self.table_files = [
            f for f in os.listdir(self.test_data_dir) if f.endswith(".csv")
        ]

    def tearDown(self):
        if os.path.exists(self.args.save_directory):
            try:
                shutil.rmtree(self.args.save_directory)
            except PermissionError:
                print("Permission denied.")
            except Exception as e:
                print(f"An error occurred: {e}")

    def test_shuffle_df_columns(self):
        num_permuts = self.args.num_shuffles

        for f in self.table_files:
            table = pd.read_csv(os.path.join(self.test_data_dir, f))
            columns = table.columns.to_list()

            # get back `num_permuts` + 1 permutations including the original
            # column order
            permutated_dfs, all_permuts = shuffle_df_columns(table, num_permuts)

            # the first df should be in the original column order
            assert columns == permutated_dfs[0].columns.to_list()

            for i in range(1, num_permuts + 1):
                permutated_header = [None] * len(table.columns)

                for k, idx in enumerate(all_permuts[i]):
                    permutated_header[k] = columns[idx]

                assert permutated_header == permutated_dfs[i].columns.to_list()

    def test_process_table_wrapper_bert(self):
        model_name = "bert-base-uncased"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_transformers_model(model_name, device)
        tokenizer, max_length = load_transformers_tokenizer_and_max_length(
            model_name
        )

        for i, f in enumerate(self.table_files):
            table = pd.read_csv(os.path.join(self.test_data_dir, f))

            max_rows_fit = truncate_index(
                table, tokenizer, max_length, model_name
            )
            truncated_table = table.iloc[:max_rows_fit, :]

            process_table_wrapper(
                i,
                truncated_table,
                model_name,
                model,
                tokenizer,
                max_length,
                padding_token=None,
                args=self.args,
                device=device,
            )
