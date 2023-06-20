import os
import argparse
import pandas as pd
import torch
from doduo import Doduo
from huggingface_models import  load_transformers_model, load_transformers_tokenizer_and_max_length
from truncate import truncate_index
def get_doduo_embeddings(tables, model_path = "."):
    device = torch.device("cuda")
    print()
    print(device)
    print()

    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"

    model = Doduo(model_args, basedir=model_path)
    model_name = 'bert-base-uncased'
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    truncated_tables =[]
    for table_index, table in enumerate(tables):
        max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)
    try:
        all_embeddings = []        
        for table in truncated_tables:
            table = table.reset_index(drop=True)

            table = table.astype(str)
            annot_df = model.annotate_columns(table)
            embeddings = annot_df.colemb
            embeddings = [torch.tensor(embeddings[j])
                        for j in range(len(embeddings))]

            all_embeddings.append(embeddings)
        return all_embeddings
    except Exception as e:
        print("Error message:", e)
        return [] 
    
    