import os
import argparse
import torch
from typing import Dict, List
from torch.nn.functional import cosine_similarity
import functools
from observatory.datasets.huggingface_dataset import chunk_neighbor_tables, batch_generator

import pandas as pd
from collections import Counter
import itertools


def jaccard_similarity(df1, df2, col1, col2):
    set1 = set(df1[col1])
    set2 = set(df2[col2])

    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    jaccard_sim = intersection / union if union != 0 else 0
    return jaccard_sim


def multiset_jaccard_similarity(df1, df2, col1, col2):
    multiset1 = Counter(df1[col1])
    multiset2 = Counter(df2[col2])

    minima = sum((multiset1 & multiset2).values())
    maxima = sum((multiset1 | multiset2).values())
    multiset_jaccard_sim = minima / maxima if maxima != 0 else 0
    return multiset_jaccard_sim


class NextiaJDCSVDataLoader:
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
                "ignore_trailing": row["ignoreTrailing"],
            }

        return metadata

    def _read_ground_truth(self, ground_truth_path: str) -> pd.DataFrame:
        return pd.read_csv(ground_truth_path)

    def read_table(
        self, table_name: str, drop_nan: bool = True, nrows=None, **kwargs
    ) -> pd.DataFrame:
        file_path = os.path.join(self.dataset_dir, table_name)
        try:
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar='"',
                on_bad_lines="skip",
                lineterminator="\n",
                low_memory=False,
                nrows=nrows,
                **kwargs,
            )
        except UnicodeDecodeError:  # To open CSV files with UnicodeDecodeError
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar='"',
                on_bad_lines="skip",
                lineterminator="\n",
                encoding="iso-8859-1",
                low_memory=False,
                **kwargs,
            )
        except ValueError:  # python engine of pandas does not support "low_memory" argument
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar='"',
                on_bad_lines="skip",
                lineterminator="\n",
                **kwargs,
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
                query = {
                    "ds_name": row["ds_name"],
                    "att_name": row["att_name"],
                    "ds_name_2": row["ds_name_2"],
                    "att_name_2": row["att_name_2"],
                }

        return queries


def get_average_embedding(table, column_name, get_embedding, model_name, tokenizer, max_length,  n=1, batch_size=10):
    # m = max(min(100 // len(table.columns.tolist()), 3), 1)
    sum_embeddings = None
    num_embeddings = 0
    # chunks_generator = split_table(table, n=n, m=m)
    chunks_generator = chunk_neighbor_tables(tables = [table,], \
        column_name = column_name, n = n , \
            model_name= model_name,  max_length=max_length,
            tokenizer = tokenizer, \
            max_token_per_cell= 20, 
        )
    # Find the index of the column in the chunk table headers
    first_chunk = next(chunks_generator)
    col_index = first_chunk["table"].columns.get_loc(column_name)

    # Use the batch_generator
    for batch_tables in batch_generator(itertools.chain([first_chunk], chunks_generator), batch_size):
        # Extract the actual tables from the dictionaries
        tables_list = [chunk_dict["table"] for chunk_dict in batch_tables]

        # Assuming your get_embedding function can handle a batch of tables
        embeddings = get_embedding(tables_list)
        for embedding in embeddings:
            if sum_embeddings is None:
                sum_embeddings = torch.zeros(embedding[col_index].size()).to(device)
            sum_embeddings += embedding[col_index].to(device)
            num_embeddings += 1
    avg_embedding = sum_embeddings / num_embeddings
    return avg_embedding


if __name__ == "__main__":
    # testbed = "testbedXS"
    # root_dir = f"/ssd/congtj/observatory/nextiajd_datasets/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--testbed", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the Hugging Face model to use",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save the results",
    )
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--num_tables", type=int, required=True)
    parser.add_argument(
        "--value", default=None, type=int, help="An optional max number of rows to read"
    )
    parser.add_argument(
        "--doduo_path",
        type=str,
        default=".",
        help="Path to load the doduo model",
    )
    parser.add_argument(
        "--nearby_column",
        type=int,
        default=1,
        help="The number of nearby columns",
    )
    args = parser.parse_args()
    model_name = args.model_name
    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"
    from observatory.models.DODUO.doduo.doduo import Doduo

    from observatory.models.huggingface_models import (
        load_transformers_model,
        load_transformers_tokenizer_and_max_length,
    )
    import functools
    from observatory.models.doduo_column_embeddings import get_doduo_embeddings

    model_path = args.doduo_path
    model = Doduo(model_args, basedir=model_path)
    temp_model_name = "bert-base-uncased"
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(temp_model_name)
    
    get_embedding = functools.partial(
        get_doduo_embeddings, model=model, tokenizer=tokenizer, max_length=max_length
    )
    n = args.nearby_column
    testbed = args.testbed
    root_dir = os.path.join(args.root_dir, testbed)
    dataset_dir = os.path.join(root_dir, "datasets")
    metadata_path = os.path.join(root_dir, f"datasetInformation_{testbed}.csv")
    ground_truth_path = os.path.join(root_dir, f"groundTruth_{testbed}.csv")
    data_loader = NextiaJDCSVDataLoader(dataset_dir, metadata_path, ground_truth_path)
    save_directory_results = os.path.join(
        args.save_dir, "Join_Relationship", testbed, model_name
    )
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    results = []
    device = torch.device("cpu")
    with open(f'error_{model_name.replace("/", "")}.txt', "w") as f:
        f.write("\n\n")
        f.write(str(model_name))
        f.write("\n")
    for i, row in data_loader.ground_truth.iterrows():
        if i < args.start:
            continue
        if i >= args.start + args.num_tables:
            break
        print(f"{i} / {data_loader.ground_truth.shape[0]}")
        if row["trueQuality"] > 0:
            t1_name, t2_name = row["ds_name"], row["ds_name_2"]
            c1_name, c2_name = row["att_name"], row["att_name_2"]
            containment = row["trueContainment"]
            t1 = data_loader.read_table(
                t1_name, drop_nan=False, nrows=args.value
            ).astype(str)
            t2 = data_loader.read_table(
                t2_name, drop_nan=False, nrows=args.value
            ).astype(str)
            # print("t1_name: ", t1_name)
            # print("c1_name: ", c1_name)
            # print("t2_name: ", t2_name)
            # print("c2_name: ", c2_name)

            # print("t1.columns: ")
            # for column in t1.columns:
            #     print(column)
            # print("t2.columns: ")
            # for column in t2.columns:
            #     print(column)
            try:
                c1_idx = list(t1.columns).index(c1_name)
                c2_idx = list(t2.columns).index(c2_name)
            except Exception as e:
                print("Error message:", e)
                continue
            try:
                c1_avg_embedding = get_average_embedding(table=t1, column_name=c1_name,  \
                    get_embedding=get_embedding, model_name=temp_model_name, tokenizer=tokenizer, max_length=max_length)
            except AssertionError:
                continue
            except Exception as e:
                with open(f'error_{model_name.replace("/", "")}.txt', "a") as f:
                    f.write(f"i: {i}")
                    f.write(
                        "In c1_avg_embedding = get_average_embedding(t1, c1_idx, n,  get_embedding): "
                    )
                    f.write(str(e))
                    f.write("\n")
                print(f"i: {i}")
                print(
                    "In c1_avg_embedding = get_average_embedding(t1, c1_idx, n,  get_embedding): "
                )
                print("Error message:", e)
                # pd.set_option('display.max_columns', None)
                # pd.set_option('display.max_rows', None)
                # print("c1_idx: ", c1_idx)
                # print(t1.columns)
                # print(t1)
                # c1_avg_embedding = get_average_embedding(t1, c1_idx, n,  get_embedding)
                continue
            try:
                c2_avg_embedding = get_average_embedding(table=t2, column_name=c2_name,  \
                    get_embedding=get_embedding, model_name=temp_model_name, tokenizer=tokenizer, max_length=max_length)
            except AssertionError:
                continue
            except Exception as e:
                with open(f'error_{model_name.replace("/", "")}.txt', "a") as f:
                    f.write(f"i: {i}")
                    f.write(
                        "In c2_avg_embedding = get_average_embedding(t2, c2_idx, n,  get_embedding) "
                    )
                    f.write(str(e))
                    f.write("\n")
                print(f"i: {i}")
                print(
                    "In c2_avg_embedding = get_average_embedding(t2, c2_idx, n,  get_embedding): "
                )
                print("Error message:", e)
                # pd.set_option('display.max_columns', None)
                # pd.set_option('display.max_rows', None)
                # print("c2_idx: ", c2_idx)
                # print(t2.columns)
                # print(t2)
                # c2_avg_embedding = get_average_embedding(t2, c2_idx, n,  get_embedding)
                continue
            data_cosine_similarity = cosine_similarity(
                c1_avg_embedding.unsqueeze(0), c2_avg_embedding.unsqueeze(0)
            )
            data_jaccard_similarity = jaccard_similarity(t1, t2, c1_name, c2_name)
            data_multiset_jaccard_similarity = multiset_jaccard_similarity(
                t1, t2, c1_name, c2_name
            )
            print("containment: ", containment)
            print("trueQuality: ", row["trueQuality"])
            print("jaccard_similarity: ", data_jaccard_similarity)
            print("multiset_jaccard_similarity: ", data_multiset_jaccard_similarity)
            print("Cosine Similarity: ", data_cosine_similarity.item())
            result = {}
            result["containment"] = containment
            result["cosine_similarity"] = data_cosine_similarity.item()
            result["jaccard_similarity"] = data_jaccard_similarity
            result["multiset_jaccard_similarity"] = data_multiset_jaccard_similarity
            results.append(result)

            # pseudo code
            # c1_embedding = f(t1)[c1_idx]
            # c2_embedding = f(t2)[c2_idx]
            # results.append((<embedding_cosine_similarity>, containment))
    torch.save(
        results,
        os.path.join(
            save_directory_results,
            f"{args.start}to{min(args.start + args.num_tables, data_loader.ground_truth.shape[0])}_results.pt",
        ),
    )
    print("save complete")
