import os
import argparse
import torch
import functools

from sotab_loader import SOTABDataLoader
from get_doduo_embeddings import get_doduo_embeddings
import pandas as pd
device = torch.device("cpu")

def split_table(table: pd.DataFrame, n: int, m: int):
            total_rows = table.shape[0]
            for i in range(0, total_rows, n*m):
                yield [table.iloc[j:j+n] for j in range(i, min(i+n*m, total_rows), n)]
                
                
                
                
                
def get_average_embedding(table, index, n,  get_embedding):
        m = min(100//len(table.columns.tolist()), 3)
        sum_embeddings = None
        num_embeddings = 0
        chunks_generator = split_table(table, n=n, m=m)
        for tables in chunks_generator:
            embeddings = get_embedding(tables)
            if sum_embeddings is None:
                sum_embeddings = torch.zeros(embeddings[0][index].size())
            for embedding in embeddings:
                sum_embeddings += embedding[index].to(device)
                num_embeddings += 1
        avg_embedding = sum_embeddings / num_embeddings
        return avg_embedding
    
if __name__ == "__main__":
    # root_dir = "/ssd/congtj/observatory/sotab_numerical_data_type_datasets"
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    # parser.add_argument('--r', type=int, required=True)
    parser.add_argument('-m', '--model_name', type=str,  required=True, help='Name of the Hugging Face model to use')
    args = parser.parse_args()
    model_name = args.model_name
    save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'p6', model_name)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    n = args.n
    root_dir = args.root_dir
    dataset_dir = os.path.join(root_dir, "tables")
    metadata_path = os.path.join(root_dir, "metadata.csv")

    data_loader = SOTABDataLoader(dataset_dir, metadata_path)    
    get_embedding =  get_doduo_embeddings

    col_itself = []
    subj_col_as_context = []
    neighbor_col_as_context = []
    entire_table_as_context = []

    for _, row in data_loader.metadata.iterrows():
        table_name = row["table_name"]
        table = data_loader.read_table(table_name)

        # input_tables = []
        # Only consider numerical column alone for representation inference
        numerical_col_idx = row["column_index"]
        numerical_col = table.iloc[:, [numerical_col_idx]]
        # input_tables.append(numerical_col)

        
        # Consider the subject column as context of numerical column for representation inference
        subj_col_idx = row["subject_column_index"]
        two_col_table = table.iloc[:, [subj_col_idx, numerical_col_idx]]
        # input_tables.append(two_col_table)

        # Consider immediate neighboring columns as context of numerical column for representation inference
        num_cols = len(list(table.columns))
   
        if numerical_col_idx > 0 and numerical_col_idx < num_cols - 1:
            three_col_table = table.iloc[:, [numerical_col_idx-1, numerical_col_idx, numerical_col_idx+1]]
        elif numerical_col_idx == num_cols - 1:
            three_col_table = table.iloc[:, [numerical_col_idx-1, numerical_col_idx]]
        
        # input_tables.append(three_col_table)

        # Consider the entire table as context of numerical column for representation inference
        # input_tables.append(table)

        # embeddings = get_hugging_face_embedding(input_tables)
        try:
            col_itself.append((get_average_embedding(numerical_col, 0, n,  get_embedding), row["label"]))
            subj_col_as_context.append((get_average_embedding(two_col_table, 1, n,  get_embedding), row["label"]))
            neighbor_col_as_context.append((get_average_embedding(three_col_table, 1, n,  get_embedding), row["label"]))
            entire_table_as_context.append((get_average_embedding(table, numerical_col_idx, n,  get_embedding), row["label"]))
        except:
            continue
        # Save embeddings
    data ={}
    data["col_itself"] = col_itself
    data["subj_col_as_context"] = subj_col_as_context
    data["neighbor_col_as_context"] = neighbor_col_as_context
    data["entire_table_as_context"] = entire_table_as_context
    torch.save(data, os.path.join(save_directory_results, f"embeddings.pt"))

