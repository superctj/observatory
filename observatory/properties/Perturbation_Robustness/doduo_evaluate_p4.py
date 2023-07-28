#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import pandas as pd
import torch
from readCompare import compare_directories

# from concurrent.futures import ThreadPoolExecutor
from torch.linalg import inv, norm
import random
import math
import itertools

from observatory.models.DODUO.doduo.doduo import Doduo


def generate_p4_embeddings(model, device, tables):
    all_shuffled_embeddings = []
    # sampled_tables = shuffle_df(table,num_samples, percentage )

    for processed_table in tables:
        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)
        annot_df = model.annotate_columns(processed_table)
        embeddings = annot_df.colemb
        embeddings = [torch.tensor(embeddings[j]) for j in range(len(embeddings))]

        all_shuffled_embeddings.append(embeddings)

    return all_shuffled_embeddings


def analyze_embeddings(all_shuffled_embeddings, changed_column_lists):
    cosine_similarities_dict = {}

    for table_index, changed_columns in enumerate(changed_column_lists):
        for column_index in changed_columns:
            original_embedding = all_shuffled_embeddings[0][column_index]
            shuffled_embedding = all_shuffled_embeddings[table_index + 1][column_index]

            cosine_similarity = torch.dot(original_embedding, shuffled_embedding) / (
                norm(original_embedding) * norm(shuffled_embedding)
            )

            if column_index not in cosine_similarities_dict:
                cosine_similarities_dict[column_index] = []

            cosine_similarities_dict[column_index].append(cosine_similarity.item())

    return cosine_similarities_dict


def process_table_wrapper(
    tables, args, model_name, model, device, key, changed_column_list
):
    save_directory_results = os.path.join(
        "/nfs/turbo/coe-jag/zjsun", "p4", args.save_directory, model_name, "results"
    )
    save_directory_embeddings = os.path.join(
        "/nfs/turbo/coe-jag/zjsun", "p4", args.save_directory, model_name, "embeddings"
    )
    # save_directory_results  = os.path.join(  args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join(args.save_directory, model_name ,'embeddings')
    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)

    all_shuffled_embeddings = generate_p4_embeddings(model, device, tables)
    torch.save(
        all_shuffled_embeddings,
        os.path.join(save_directory_embeddings, f"{key}_embeddings.pt"),
    )
    cosine_similarities_dict = analyze_embeddings(
        all_shuffled_embeddings, changed_column_list
    )
    for column_index, similarities in cosine_similarities_dict.items():
        print(f"Column {column_index}:")
        for i, similarity in enumerate(similarities):
            print(f"\tCosine similarity with modified table {i+1}: {similarity}")
    torch.save(
        cosine_similarities_dict,
        os.path.join(save_directory_results, f"{key}_results.pt"),
    )


def process_and_save_embeddings(model_name, args, result_dict):
    device = torch.device("cuda")
    print()
    print(device)
    print()

    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"

    model = Doduo(model_args, basedir=".")

    for key, value in result_dict.items():
        try:
            process_table_wrapper(
                value[0], args, model_name, model, device, key, value[1]
            )
        except Exception as e:
            print("Error message:", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process tables and save embeddings.")
    parser.add_argument(
        "-o",
        "--original_directory",
        type=str,
        required=True,
        help="Directory of the original tables.",
    )
    parser.add_argument(
        "-c",
        "--changed_directory",
        type=str,
        required=True,
        help="Directory of the modified tables.",
    )
    parser.add_argument(
        "-s",
        "--save_directory",
        type=str,
        required=True,
        help="Directory to save embeddings to",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="",
        help="Name of the Hugging Face model to use",
    )
    args = parser.parse_args()

    result_dict = compare_directories(args.original_directory, args.changed_directory)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    for key, value in result_dict.items():
        print(key)
        for things in value:
            print(things)

    model_name = args.model_name

    print()
    print("Evaluate row shuffle for: ", model_name)
    print()

    process_and_save_embeddings(model_name, args, result_dict)
