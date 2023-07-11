import os
import pandas as pd
import json
import numpy as np
import sys
import argparse
import functools
import torch
from typing import Dict, List
from cellbased_get_hugging_face_embeddings import get_hugging_face_cell_embeddings
from doduo_entity_embeddings import Doduo

import pandas as pd


class SpiderFDDataLoader():
    def __init__(self, dataset_dir: str, fd_metadata_path: str, non_fd_metadata_path: str):
        self.dataset_dir = dataset_dir        
        self.fd_metadata_path = fd_metadata_path
        self.non_fd_metadata_path = non_fd_metadata_path
    
    def read_table(self, table_name: str, drop_nan=True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, table_name)
        table = pd.read_csv(
            table_path, on_bad_lines="skip", lineterminator="\n", **kwargs)
        
        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_fd_metadata(self):
        fd_metadata = pd.read_csv(self.fd_metadata_path, sep=",")
        return fd_metadata
    
    def get_non_fd_metadata(self):
        non_fd_metadata = pd.read_csv(self.non_fd_metadata_path, sep=",")
        return non_fd_metadata

import pandas as pd

def find_groups(df, determinant_col, dependent_col):
    # Create a new DataFrame with only the two columns of interest and the index
    df_temp = df[[determinant_col, dependent_col]].reset_index()
    df_temp['pair'] = list(zip(df_temp[determinant_col], df_temp[dependent_col]))

    # Group by the new pair column and aggregate indices into lists
    groups = df_temp.groupby('pair')['index'].apply(list)

    # Convert the Series to a dictionary
    pairs_dict = groups.to_dict()

    return pairs_dict

if __name__ == "__main__":
    # root_dir = f"/ssd/congtj/observatory/spider_datasets/fd_artifact"


    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str,  required=True, help='Name of the Hugging Face model to use')
    parser.add_argument('-r', '--root_dir', type=str,  required=True, help='Root directory')

    args = parser.parse_args()
    root_dir = args.root_dir 
    dataset_dir = os.path.join(root_dir, "datasets")
    fd_metadata_path = os.path.join(root_dir, "fd_metadata.csv")
    non_fd_metadata_path = os.path.join(root_dir, "non_fd_metadata.csv")
    data_loader = SpiderFDDataLoader(dataset_dir, fd_metadata_path, non_fd_metadata_path)

    model_name = args.model_name
    save_directory  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'FD', model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # save_directory_cell = os.path.join(save_directory, "original_cell_embeddings")
    # save_directory_pairs = os.path.join(save_directory, "pair_embeddings")
    # if not model_name.startswith("doduo"):
    #     if not os.path.exists(save_directory_cell):
    #         os.makedirs(save_directory_cell)
    # if not os.path.exists(save_directory_pairs):
    #     os.makedirs(save_directory_pairs)
    if model_name.startswith("bert") or model_name.startswith("roberta") or model_name.startswith("google/tapas") or model_name.startswith("t5"):
            get_embedding =  functools.partial(get_hugging_face_cell_embeddings, model_name=model_name)
    elif model_name.startswith("doduo"):
        model_args = argparse.Namespace
        model_args.model = "wikitable"
        doduo = Doduo(model_args, basedir="/home/zjsun/DuDuo/doduo")
        get_embedding =  doduo.get_entity_embeddings
    
    fd_metadata = data_loader.get_fd_metadata()
    list_pairs_norms_dict = []
    for _, row in fd_metadata.iterrows():
        table_name = row["table_name"]
        determinant = row["determinant"]
        dependent = row["dependent"]

        table = data_loader.read_table(table_name)
        determinant_index = list(table.columns).index(determinant)
        dependent_index = list(table.columns).index(dependent)
        pairs_dict = find_groups(table, determinant, dependent)
        norms_dict = {}
        if model_name.startswith("doduo"):
            tmp_pairs = []
            for pair, list_row_index in pairs_dict.items():
                for row_index in list_row_index:
                    tmp_pairs.append([[row_index, determinant_index], ((pair, "determinant"), "")])
                    tmp_pairs.append([[row_index, dependent_index], ((pair,"dependent"), "")])          
            try:
                entity_embeddings = get_embedding(table, tmp_pairs)
            except ValueError as e:
                print(e)
                continue
            
            for pair, list_row_index in pairs_dict.items():
                l2_norms = []
                for row_index in list_row_index:
                    try:
                        determinant_embedding, _ = entity_embeddings[(row_index, determinant_index)]
                        dependent_embedding, _ = entity_embeddings[(row_index, dependent_index)]
                        l2_norm = torch.norm(determinant_embedding - dependent_embedding, p=2)
                        l2_norms.append(l2_norm.item())  # Convert the torch tensor to a Python float
                    except (KeyError, AssertionError) as e:
                        print(e)
                        continue
                norms_dict[pair] = l2_norms            
        else:
            cell_embeddings = get_embedding(table)
            for pair, list_row_index in pairs_dict.items():
                l2_norms = []
                for row_index in list_row_index:
                    try:
                        determinant_embedding = cell_embeddings[row_index+1][determinant_index]
                        dependent_embedding = cell_embeddings[row_index+1][dependent_index]
                        l2_norm = torch.norm(determinant_embedding - dependent_embedding, p=2)
                        l2_norms.append(l2_norm.item())  # Convert the torch tensor to a Python float
                    except IndexError:
                        continue
                norms_dict[pair] = l2_norms
                
        list_pairs_norms_dict.append(norms_dict)
        # print(table.head())
        # # infer cell embeddings in determinant column and dependent column
        # break
    torch.save(list_pairs_norms_dict, os.path.join(save_directory,  f"list_pairs_norms_dict.pt"))


    non_fd_metadata = data_loader.get_non_fd_metadata()
    list_non_pairs_norms_dict = []
    for _, row in non_fd_metadata.iterrows():
        table_name = row["table_name"]
        col1 = row["column_1"]
        col2 = row["column_2"]

        table = data_loader.read_table(table_name)
        col1_index = list(table.columns).index(col1)
        col2_index = list(table.columns).index(col2)
        elements_dict = table.reset_index().groupby(col1)['index'].apply(list).to_dict()

        norms_dict = {}

        if model_name.startswith("doduo"):
            tmp_pairs = []
            for element, list_row_index in elements_dict.items():
                for row_index in list_row_index:
                    tmp_pairs.append([[row_index, col1_index], ((element, "col1"), "")])
                    tmp_pairs.append([[row_index, col2_index], ((element, "col2"), "")])          
            try:
                entity_embeddings = get_embedding(table, tmp_pairs)
            except ValueError as e:
                print(e)
                continue

            for element, list_row_index in elements_dict.items():
                l2_norms = []
                for row_index in list_row_index:

                    try:
                        col1_embedding, _ = entity_embeddings[(row_index, col1_index)]
                        col2_embedding, _ = entity_embeddings[(row_index, col2_index)]
                        l2_norm = torch.norm(col1_embedding - col2_embedding, p=2)
                        l2_norms.append(l2_norm.item())  # Convert the torch tensor to a Python float
                    except (KeyError, AssertionError) as e:
                        print(e)
                        continue
                norms_dict[element] = l2_norms  
        else:
            cell_embeddings = get_embedding(table)
            for element, list_row_index in elements_dict.items():
                l2_norms = []
                for row_index in list_row_index:
                    try:
                        col1_embedding = cell_embeddings[row_index+1][col1_index]
                        col2_embedding = cell_embeddings[row_index+1][col2_index]
                        l2_norm = torch.norm(col1_embedding - col2_embedding, p=2)
                        l2_norms.append(l2_norm.item())  # Convert the torch tensor to a Python float
                    except IndexError:
                        continue
                norms_dict[element] = l2_norms
        list_non_pairs_norms_dict.append(norms_dict)

    torch.save(list_non_pairs_norms_dict, os.path.join(save_directory,  f"list_non_pairs_norms_dict.pt"))

    