import os
import argparse
import pandas as pd
import torch
from observatory.models.DODUO.doduo.doduo import Doduo
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.column_based_truncate import column_based_truncate

# model_args = argparse.Namespace
# model_args.model = "wikitable"  # two models available "wikitable" and "viznet"
# model = Doduo(model_args, basedir=model_path)
# tokenizer, max_length = load_transformers_tokenizer_and_max_length('bert-base-uncased')


def get_doduo_embeddings(tables, model, tokenizer, max_length):
    device = torch.device("cuda")
    print()
    print(device)
    print()

    model_name = "bert-base-uncased"
    truncated_tables = []
    for table_index, table in enumerate(tables):
        max_rows_fit = column_based_truncate(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)
    try:
        all_embeddings = []
        for table in truncated_tables:
            table = table.reset_index(drop=True)

            table = table.astype(str)
            annot_df = model.annotate_columns(table)
            embeddings = annot_df.colemb
            embeddings = [torch.tensor(embeddings[j]) for j in range(len(embeddings))]

            all_embeddings.append(embeddings)
        return all_embeddings
    except Exception as e:
        print("Error message:", e)
        return []
