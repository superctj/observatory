import os
import argparse
import itertools
import random

import pandas as pd
import torch
import numpy as np
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.table_based_truncate import table_based_truncate
from observatory.common_util.mcv import compute_mcv
from torch.linalg import inv, norm
from observatory.models.hugging_face_table_embeddings import (
    get_hugging_face_table_embeddings_batched,
)



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
    table_embeddings = []
    for j in range(len(all_embeddings)):
            table_embeddings.append(all_embeddings[j])

    cosine_similarities = []
    for j in range(1, len(all_embeddings)):
        truncated_embedding = all_embeddings[0]
        shuffled_embedding = all_embeddings[j]

        cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (
            norm(truncated_embedding) * norm(shuffled_embedding)
        )
        cosine_similarities.append(cosine_similarity.item())
    
    avg_cosine_similarity = torch.mean(torch.tensor(cosine_similarities))
    mcv = compute_mcv(torch.stack(table_embeddings))


    return (
        avg_cosine_similarity,
        mcv,
        avg_cosine_similarity,
        mcv
    )


def process_table_wrapper(
    table_index,
    truncated_table,
    args,
    model_name,
    model,
    tokenizer,
    device,
    max_length,
    padding_token,
):
    save_directory_results = os.path.join(
        args.save_directory,
        "Table_embedding_Column_Order_Insignificance",
        model_name,
        "results",
    )
    save_directory_embeddings = os.path.join(
        args.save_directory,
        "Table_embedding_Column_Order_Insignificance",
        model_name,
        "embeddings",
    )
    # save_directory_results  = os.path.join( args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join( args.save_directory, model_name ,'embeddings')
    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
        
    tables, perms = shuffle_df_columns(truncated_table, args.num_shuffles)

    all_embeddings = get_hugging_face_table_embeddings_batched(
        tables,
        model_name,
        tokenizer,
        max_length,
        model,
        args.batch_size
    )
    if len(all_embeddings)<24:
        print("len(all_embeddings)<24")
        return
    torch.save(
        all_embeddings,
        os.path.join(save_directory_embeddings, f"table_{table_index}_embeddings.pt"),
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
    print(f"Table {table_index}:")
    print("Average Cosine Similarities:", results["avg_cosine_similarities"])
    print("MCVs:", results["mcvs"])
    print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
    print("Table Average MCV:", results["table_avg_mcv"])
    torch.save(
        results, os.path.join(save_directory_results, f"table_{table_index}_results.pt")
    )


def process_and_save_embeddings(model_name, args, tables):
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = load_transformers_model(model_name, device)
    model.eval()
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    for table_index, table in enumerate(tables):
        if table_index < args.start_index:
            continue
        max_rows_fit = table_based_truncate(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        process_table_wrapper(
            table_index,
            truncated_table,
            args,
            model_name,
            model,
            tokenizer,
            device,
            max_length,
            padding_token,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables and save embeddings.")
    parser.add_argument(
        "-r",
        "--read_directory",
        type=str,
        required=True,
        help="Directory to read tables from",
    )
    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Directory to save embeddings to",
    )
    parser.add_argument(
        "-n",
        "--num_shuffles",
        type=int,
        required=True,
        help="Number of times to shuffle and save embeddings",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="",
        help="Name of the Hugging Face model to use",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The batch size for parallel inference",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start table index",
    )
    args = parser.parse_args()

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    if args.model_name == "":
        model_names = [
            "bert-base-uncased",
            "roberta-base",
            "t5-base",
            "google/tapas-base",
        ]
    else:
        model_names = [args.model_name]
    print()
    print("Evaluate row shuffle for: ", model_names)
    print()

    for model_name in model_names:
        process_and_save_embeddings(model_name, args, normal_tables)
