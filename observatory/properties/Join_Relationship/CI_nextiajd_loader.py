import os
import argparse
import torch
from observatory.models.hugging_face_column_embeddings import (
    get_hugging_face_column_embeddings_batched,
)
from typing import Dict, List
from torch.linalg import inv, norm
import functools
from observatory.datasets.huggingface_dataset import batch_generator, chunk_tables
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.mcv import compute_mcv
import itertools
import pandas as pd
from collections import Counter
import random




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



def get_average_embeddings(table, get_embedding, model_name, tokenizer, max_length, max_row=10,  max_col=10, batch_size=256):

    chunks_generator = chunk_tables(tables = [table,], \
        model_name=model_name, \
            max_length=max_length,
            tokenizer=tokenizer,
            max_col= max_col,
            max_row= max_row,
            max_token_per_cell= 20, 
        )

    all_sum_column_embeddings = [None for col in table.columns]
    all_column_count = [0 for col in table.columns]
    # Use the batch_generator
    for batch_tables in batch_generator(chunks_generator, batch_size):
        # Extract the actual tables from the dictionaries
        tables_list = [chunk_dict["table"] for chunk_dict in batch_tables]
        col_list = [chunk_dict["position"][0] for chunk_dict in batch_tables]

        all_embeddings = get_embedding(tables_list)
        
        for embeddings, col_range in zip(all_embeddings, col_list):
            start_col = col_range[0]
            end_col = col_range[1]
            # start_col = start_col.item()
            # end_col = end_col.item()
            for col_index, embeding in zip(range(start_col, end_col), embeddings):
                if all_sum_column_embeddings[col_index] is None:
                    all_sum_column_embeddings[col_index] = torch.zeros(embeding.size())
                all_sum_column_embeddings[col_index] += embeding
                all_column_count[col_index] += 1
    all_avg_column_embeddings = [sum_embeddings / num_embeddings for sum_embeddings, num_embeddings in zip(all_sum_column_embeddings, all_column_count)]
    return all_avg_column_embeddings

def fisher_yates_shuffle(seq):
    for i in reversed(range(1, len(seq))):
        j = random.randint(0, i)
        seq[i], seq[j] = seq[j], seq[i]
    return seq


def get_permutations(n, m):
    if n < 10:
        # Generate all permutations
        all_perms = list(itertools.permutations(range(n)))
        # Remove the original sequence
        all_perms.remove(tuple(range(n)))
        # Shuffle the permutations
        random.shuffle(all_perms)
        # If m > n! - 1 (because we removed one permutation), return all permutations
        if m > len(all_perms):
            return [list(range(n))] + all_perms
        # Otherwise, return the first m permutations
        return [list(range(n))] + all_perms[:m]
    else:
        original_seq = list(range(n))
        perms = [original_seq.copy()]
        for _ in range(m):  # we already have one permutation
            while True:
                new_perm = fisher_yates_shuffle(original_seq.copy())
                if new_perm not in perms:
                    perms.append(new_perm)
                    break
        return perms


def shuffle_df_columns(df, m):
    # Get the permutations
    perms = get_permutations(len(df.columns), m)

    # Create a new dataframe for each permutation
    dfs = []
    for perm in perms:
        dfs.append(df.iloc[:, list(perm)])

    return dfs, perms

def analyze_embeddings(all_embeddings):
    avg_cosine_similarities = []
    mcvs = []

    for i in range(len(all_embeddings[0])):
        column_cosine_similarities = []
        column_embeddings = []

        for j in range(len(all_embeddings)):
            column_embeddings.append(all_embeddings[j][i])

        for j in range(1, len(all_embeddings)):
            truncated_embedding = all_embeddings[0][i]
            shuffled_embedding = all_embeddings[j][i]

            cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (
                norm(truncated_embedding) * norm(shuffled_embedding)
            )
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = torch.mean(torch.tensor(column_cosine_similarities))
        mcv = compute_mcv(torch.stack(column_embeddings))

        avg_cosine_similarities.append(avg_cosine_similarity.item())
        mcvs.append(mcv)

    table_avg_cosine_similarity = torch.mean(torch.tensor(avg_cosine_similarities))
    table_avg_mcv = torch.mean(torch.tensor(mcvs))

    return (
        avg_cosine_similarities,
        mcvs,
        table_avg_cosine_similarity.item(),
        table_avg_mcv.item(),
    )


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
        "--batch_size",
        type=int,
        default=32,
        help="The btach size for inference",
    )
    parser.add_argument(
        "--max_row",
        type=int,
        default=10,
        help="max_row for chunck generator",
    )
    parser.add_argument(
        "--max_col",
        type=int,
        default=10,
        help="max_col for chunck generator",
    )
    parser.add_argument(
        "-n",
        "--num_shuffles",
        type=int,
        required=True,
        help="Number of times to shuffle and save embeddings",
    )
    args = parser.parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    device = torch.device("cuda")
    print(device)
    model = load_transformers_model(model_name, device)
    model = model.eval()
    get_embedding = functools.partial(
        get_hugging_face_column_embeddings_batched,
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
        batch_size=batch_size,
    )

    testbed = args.testbed
    root_dir = os.path.join(args.root_dir, testbed)
    dataset_dir = os.path.join(root_dir, "datasets")
    metadata_path = os.path.join(root_dir, f"datasetInformation_{testbed}.csv")
    ground_truth_path = os.path.join(root_dir, f"groundTruth_{testbed}.csv")
    data_loader = NextiaJDCSVDataLoader(dataset_dir, metadata_path, ground_truth_path)
    save_directory_results = os.path.join(
        args.save_dir,
        "CI_Join_Relationship",
        testbed,
        model_name,
        "results",
    )
    save_directory_embeddings = os.path.join(
        args.save_dir,
        "CI_Join_Relationship",
        testbed,
        model_name,
        "embeddings",
    )
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    results = []
    filename = "table_names.txt"
    
    # Check if the file already exists
    if os.path.exists(filename):
        # If it exists, read the file and get the table names
        with open(filename, 'r') as f:
            table_names = f.readlines()
        table_names = [name.strip() for name in table_names]
    else:
        # If it doesn't exist, collect the table names
        table_names_set = set()
        for i, row in data_loader.ground_truth.iterrows():
            if row["trueQuality"] > 0:
                t1_name, t2_name = row["ds_name"], row["ds_name_2"]
                table_names_set.add(t1_name)
                table_names_set.add(t2_name)
        
        # Convert the set to a list
        table_names = list(table_names_set)
        
        # Save the table names into a file
        with open(filename, 'w') as f:
            for name in table_names:
                f.write(name + '\n')

    # Iterate over the table names
    for i, table_name in enumerate(table_names):
        if i < args.start:
            continue
        if i >= args.start + args.num_tables:
            break
        print(f"{i} / {len(table_names)}")
        # Read the table
        table = data_loader.read_table(table_name, drop_nan=False, nrows=args.value)
        tables, perms = shuffle_df_columns(table, args.num_shuffles)
        all_embeddings = []
        for table in tables:
            column_embeddings = get_average_embeddings(table, get_embedding=get_embedding, \
                model_name=model_name, tokenizer=tokenizer, max_length=max_length, \
                    max_row= args.max_row,  max_col= args.max_col, batch_size = batch_size)
            
            all_embeddings.append(column_embeddings)
            
        all_ordered_embeddings = []
        for perm ,embeddings in  zip(perms, all_embeddings):
            
            # Create a list of the same length as perm, filled with None
            ordered_embeddings = [None] * len(perm)
            # Assign each embedding to its original position
            for i, p in enumerate(perm):
                ordered_embeddings[p] = embeddings[i]
            all_ordered_embeddings.append(ordered_embeddings)
        all_embeddings = all_ordered_embeddings
        
        torch.save(
            all_embeddings,
            os.path.join(save_directory_embeddings, f"table_{table_name}_{i}_embeddings.pt"),
        )
        (
            avg_cosine_similarities,
            mcvs,
            table_avg_cosine_similarity,
            table_avg_mcv,
        ) = analyze_embeddings(all_embeddings)
        results = {
            "avg_cosine_similarities": avg_cosine_similarities,
            "mcvs": mcvs,
            "table_avg_cosine_similarity": table_avg_cosine_similarity,
            "table_avg_mcv": table_avg_mcv,
        }
        print(f"Table {table_name}, the {i}th table:")
        print("Average Cosine Similarities:", results["avg_cosine_similarities"])
        print("MCVs:", results["mcvs"])
        print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
        print("Table Average MCV:", results["table_avg_mcv"])
        torch.save(
            results, os.path.join(save_directory_results, f"table_{table_name}_{i}_results.pt")
        )   


