#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import itertools
import random

import pandas as pd
import torch

import numpy as np


from observatory.common_util.mcv import compute_mcv

from torch.linalg import norm
from observatory.models.TaBERT.table_bert import Table, Column
from observatory.models.TaBERT.table_bert import TableBertModel


def convert_to_table(df, tokenizer):

    header = []
    data = []

    for col in df.columns:
        try:
            # Remove commas and attempt to convert to float
            val = float(str(df[col].iloc[0]).replace(",", ""))
            # If conversion is successful, it's a real column
            col_type = "real"
            sample_value = df[col][0]
        except (ValueError, AttributeError):
            # If conversion fails, it's a text column
            col_type = "text"
            sample_value = df[col][0]

        # Create a Column object
        header.append(Column(col, col_type, sample_value=sample_value))

        # Add the column data to 'data' list
    for row_index in range(len(df)):
        data.append(list(df.iloc[row_index]))
        # print()
        # print(col_type)
        # print(sample_value)
    # Create the Table
    table = Table(id="", header=header, data=data)

    # Tokenize
    table.tokenize(tokenizer)

    return table


def fisher_yates_shuffle(seq):

    for i in reversed(range(1, len(seq))):

        j = random.randint(0, i)

        seq[i], seq[j] = seq[j], seq[i]

    return seq


def get_permutations(n, m):

    if n < 9:

        # Generate all permutations
        all_perms = list(itertools.permutations(range(n)))

        # Remove the original sequence

        all_perms.remove(tuple(range(n)))

        # Shuffle the permutations
        random.shuffle(all_perms)

        # If m > n! - 1 (because we removed one permutation), return all permutations

        if m > len(all_perms):
            return all_perms

        # Otherwise, return the first m permutations

        return all_perms[:m]

    else:

        original_seq = list(range(n))

        perms = [original_seq.copy()]

        for _ in range(m):  # we already have one permutation

            while True:

                new_perm = fisher_yates_shuffle(original_seq.copy())

                if new_perm not in perms:

                    perms.append(new_perm)

                    break

        perms.remove(list(range(n)))
        return perms


# Define the function to shuffle a dataframe and create new dataframes


def shuffle_df(df, m):

    # Get the permutations
    perms = get_permutations(len(df), m)

    # Create a new dataframe for each permutation

    dfs = [df]

    for perm in perms:

        dfs.append(df.iloc[list(perm)])

    return dfs


def generate_row_shuffle_embeddings(model, device, table, num_shuffles):

    all_shuffled_embeddings = []
    tables = shuffle_df(table, num_shuffles)

    for j in range(len(tables)):

        processed_table = tables[j]

        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)
        processed_table = convert_to_table(processed_table, model.tokenizer)
        context = ""
        with torch.no_grad():
            context_encoding, column_encoding, info_dict = model.encode(
                contexts=[model.tokenizer.tokenize(context)], tables=[processed_table]
            )
        embeddings = column_encoding[0]
        all_shuffled_embeddings.append(embeddings)
        # Free up some memory by deleting column_encoding and info_dict variables
        del column_encoding
        del info_dict
        del context_encoding
        del embeddings
        # Empty the cache
        torch.cuda.empty_cache()

    return all_shuffled_embeddings


def analyze_embeddings(all_shuffled_embeddings):

    avg_cosine_similarities = []

    mcvs = []

    for i in range(len(all_shuffled_embeddings[0])):

        column_cosine_similarities = []

        column_embeddings = []

        for j in range(len(all_shuffled_embeddings)):

            column_embeddings.append(all_shuffled_embeddings[j][i])

        for j in range(1, len(all_shuffled_embeddings)):

            truncated_embedding = all_shuffled_embeddings[0][i]

            shuffled_embedding = all_shuffled_embeddings[j][i]

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


def process_table_wrapper(table_index, table, args, model_name, model, device):

    save_directory_results = os.path.join(
        args.save_directory,
        "Row_Order_Insignificance",
        model_name,
        "results",
    )

    save_directory_embeddings = os.path.join(
        args.save_directory,
        "Row_Order_Insignificance",
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

    all_shuffled_embeddings = generate_row_shuffle_embeddings(
        model, device, table, args.num_shuffles
    )

    torch.save(
        all_shuffled_embeddings,
        os.path.join(save_directory_embeddings, f"table_{table_index}_embeddings.pt"),
    )

    (
        avg_cosine_similarities,
        mcvs,
        table_avg_cosine_similarity,
        table_avg_mcv,
    ) = analyze_embeddings(all_shuffled_embeddings)

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

    device = torch.device("cuda")
    print()
    print(device)
    print()

    model = TableBertModel.from_pretrained(
         args.tabert_bin,
    )
    model = model.to(device)
    model.eval()

    for table_index, table in enumerate(tables):

        if table_index < args.table_num:
            continue
        # try:
        process_table_wrapper(table_index, table, args, model_name, model, device)
        # except Exception as e:
        #     print("Error message:", e)
        #     pd.set_option('display.max_columns', None)
        #     pd.set_option('display.max_rows', None)
        #     print(table.columns)
        #     print(table)


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
        "-t", "--table_num", type=int, default=0, help="num of start table"
    )
    parser.add_argument(
        "--tabert_bin",
        type=str,
        default=".",
        help="Path to load the tabert model",
    )
    args = parser.parse_args()

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]

    normal_tables = []

    for file in table_files:

        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    model_name = args.model_name
    print()

    print("Evaluate  for: ", model_name)
    print()

    process_and_save_embeddings(model_name, args, normal_tables)
