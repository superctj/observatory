import os
import argparse
import pandas as pd
import torch
from doduo import Doduo

def get_doduo_embeddings(tables, model_path = "."):
    device = torch.device("cuda")
    print()
    print(device)
    print()

    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"

    model = Doduo(model_args, basedir=model_path)
    
    try:
        all_embeddings = []        
        for table in tables:
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
    
    