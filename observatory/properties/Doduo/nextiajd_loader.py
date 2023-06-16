import os
import argparse
import torch
from get_doduo_embeddings import get_doduo_embeddings
from typing import Dict, List
from torch.nn.functional import cosine_similarity

import pandas as pd


class NextiaJDCSVDataLoader():
    def __init__(self, dataset_dir: str, metadata_path: str, ground_truth_path: str):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{dataset_dir} does not exist.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{metadata_path} does not exist.")
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"{ground_truth_path} does not exist.")

        self.dataset_dir = dataset_dir        
        self.metadata = self._read_metadata(metadata_path)
        self.ground_truth = self._read_ground_truth(ground_truth_path)
    
    def _read_metadata(self, metadata_path: str) -> pd.DataFrame:
        df = pd.read_csv(metadata_path)
        metadata = {}

        for _, row in df.iterrows():
            metadata[row["filename"]] = {
                "delimiter": row["delimiter"],
                "null_val": row["nullVal"],
                "ignore_trailing": row["ignoreTrailing"]
            }

        return metadata
    
    def _read_ground_truth(self, ground_truth_path: str) -> pd.DataFrame:
        return pd.read_csv(ground_truth_path)

    def read_table(self, table_name: str, drop_nan: bool = True, **kwargs) -> pd.DataFrame:
        file_path = os.path.join(self.dataset_dir, table_name)
        try:
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                low_memory=False,
                **kwargs
            )
        except UnicodeDecodeError: # To open CSV files with UnicodeDecodeError
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                encoding="iso-8859-1",
                low_memory=False,
                **kwargs
            )
        except ValueError: # python engine of pandas does not support "low_memory" argument
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                **kwargs
            )

        # Remove hidden characters (e.g., "\r") in DataFrame header and data
        table.columns = table.columns.str.replace("[\r]", "", regex=True)
        table.replace(to_replace=["\r", "\n"], value="", regex=True, inplace=True)

        # Drop empty columns and rows
        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="all", inplace=True)

        return table

    def get_table_names(self) -> List[str]:
        return self.metadata.keys()
    
    def get_queries(self) -> Dict[str, List[str]]:
        queries = []

        for _, row in self.ground_truth.iterrows():
            if row["trueQuality"] > 0:
                query = {"ds_name": row["ds_name"],
                         "att_name": row["att_name"],
                         "ds_name_2": row["ds_name_2"],
                         "att_name_2": row["att_name_2"],

                        }
            
        return queries
    
    def split_table(self, table: pd.DataFrame, n: int, m: int):
            total_rows = table.shape[0]
            for i in range(0, total_rows, n*m):
                yield [table.iloc[j:j+n] for j in range(i, min(i+n*m, total_rows), n)]

if __name__ == "__main__":
    # testbed = "testbedXS"
    # root_dir = f"/ssd/congtj/observatory/nextiajd_datasets/"
    parser = argparse.ArgumentParser()
    parser.add_argument('--testbed', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--r', type=int, required=True)
    parser.add_argument('-m', '--model_name', type=str,  required=True, help='Name of the Hugging Face model to use')
    
    args = parser.parse_args()
    model_name = args.model_name
    n = args.n
    r = args.r
    testbed = args.testbed
    root_dir =  os.path.join(args.root_dir, testbed)
    dataset_dir = os.path.join(root_dir, "datasets")
    metadata_path = os.path.join(root_dir, f"datasetInformation_{testbed}.csv")
    ground_truth_path = os.path.join(root_dir, f"groundTruth_{testbed}.csv")
    data_loader = NextiaJDCSVDataLoader(dataset_dir, metadata_path, ground_truth_path)
    save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'p5', testbed, model_name)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    results = []
    device = torch.device("cpu")
    for i, row in data_loader.ground_truth.iterrows():
        print(f"{i} / {data_loader.ground_truth.shape[0]}")
        if row["trueQuality"] > 0:
            t1_name, t2_name = row["ds_name"], row["ds_name_2"]
            c1_name, c2_name = row["att_name"], row["att_name_2"]
            containment = row["trueContainment"]

            t1 = data_loader.read_table(t1_name)
            t2 = data_loader.read_table(t2_name)

            c1_idx = list(t1.columns).index(c1_name)
            c2_idx = list(t2.columns).index(c2_name)
            
            c1_sum_embeddings = None
            c1_num_embeddings = 0
            c1_chunks_generator = data_loader.split_table(t1, n=n, m=r)
            for tables in c1_chunks_generator:
                embeddings = get_doduo_embeddings(tables)
                if c1_sum_embeddings is None and len(embeddings)>0:
                    c1_sum_embeddings = torch.zeros(embeddings[0][c1_idx].size())
                for embedding in embeddings:
                    c1_sum_embeddings += embedding[c1_idx]
                    c1_num_embeddings += 1
            

            c2_sum_embeddings = None
            c2_num_embeddings = 0
            c2_chunks_generator = data_loader.split_table(t2, n=n, m=r)
            for tables in c2_chunks_generator:
                embeddings = get_doduo_embeddings(tables)
                if c2_sum_embeddings is None and len(embeddings)>0:
                    c2_sum_embeddings = torch.zeros(embeddings[0][c2_idx].size())
                for embedding in embeddings:
                    c2_sum_embeddings += embedding[c2_idx]
                    c2_num_embeddings += 1
            try:
                c2_avg_embedding = c2_sum_embeddings / c2_num_embeddings
                c1_avg_embedding = c1_sum_embeddings / c1_num_embeddings
            except:
                continue
            similarity = cosine_similarity(c1_avg_embedding.unsqueeze(0), c2_avg_embedding.unsqueeze(0))
            print("containment: ", containment)
            print("trueQuality: ", row["trueQuality"])
            print("Cosine Similarity: ", similarity.item())
            results.append((containment, similarity.item()))
    
            # pseudo code
            # c1_embedding = f(t1)[c1_idx]
            # c2_embedding = f(t2)[c2_idx]
            # results.append((<embedding_cosine_similarity>, containment))
    torch.save(results, os.path.join(save_directory_results, f"results.pt"))