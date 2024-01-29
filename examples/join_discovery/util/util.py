import os

import pandas as pd
import torch

from web_table_embedding_cl.model import SimTableCLR


def lower_case_filenames(data_dir: str):
    for db in os.listdir(data_dir):
        db_dir = os.path.join(data_dir, db)
        for csv_file in os.listdir(db_dir):
            if csv_file.endswith(".csv"):
                src_path = os.path.join(db_dir, csv_file)
                dst_path = os.path.join(db_dir, csv_file.lower())
                os.rename(src_path, dst_path)


def load_pylon_wte_model(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_path)["state_dict"]
    new_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("model"):
            new_key = ".".join(key.split(".")[1:])
            new_state_dict[new_key] = value
    
    model = SimTableCLR(embedding_dim=150, projection_size=64)
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    # print(next(model.parameters()).device)
    model.eval()
    return model


if __name__ == "__main__":
    data_dir = "/data/spider_artifact/db_csv_extended/"
    lower_case_filenames(data_dir)