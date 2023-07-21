#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import itertools
import random

import pandas as pd
import torch

import numpy as np


from torch.linalg import inv, norm

from torch.serialization import save

from scipy.spatial.distance import cosine

from observatory.models.DODUO.doduo.doduo import Doduo


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


def generate_col_shuffle_embeddings(model, table, num_shuffles):

    all_shuffled_embeddings = []
    tables, perms = shuffle_df_columns(table, num_shuffles)

    for j in range(len(tables)):

        processed_table = tables[j]

        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)

        annot_df = model.annotate_columns(processed_table)
        embeddings = annot_df.colemb
        embeddings = [torch.tensor(embeddings[j])
                      for j in range(len(embeddings))]

        perm = perms[j]

        # Create a list of the same length as perm, filled with None

        ordered_embeddings = [None] * len(perm)

        # Assign each embedding to its original position

        for i, p in enumerate(perm):

            ordered_embeddings[p] = embeddings[i]         
        all_shuffled_embeddings.append(ordered_embeddings)

        # Clear memory
        del embeddings

        torch.cuda.empty_cache()

    return all_shuffled_embeddings, tables


# from multiprocessing import Pool, cpu_count

# def process_table(args):
#     processed_table, model, perm = args
#     processed_table = processed_table.reset_index(drop=True)
#     processed_table = processed_table.astype(str)
#     annot_df = model.annotate_columns(processed_table)
#     embeddings = annot_df.colemb
#     embeddings = [torch.tensor(embeddings[j]) for j in range(len(embeddings))]
#     ordered_embeddings = [None] * len(perm)
#     for i, p in enumerate(perm):
#         ordered_embeddings[p] = embeddings[i]
#     return ordered_embeddings

# def generate_col_shuffle_embeddings(model, table, num_shuffles):
#     tables, perms = shuffle_df_columns(table, num_shuffles)
#     with Pool(cpu_count()) as p:
#         all_shuffled_embeddings = p.map(process_table, [(tables[j], model, perms[j]) for j in range(len(tables))])
#     return all_shuffled_embeddings


def analyze_embeddings(all_shuffled_embeddings,tables):

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

                norm(truncated_embedding) * norm(shuffled_embedding))
            if cosine_similarity.item() < 0.7:
                with open('output.txt', 'a') as f:
                    f.write(f"In cloumn {i}: \n")
                    f.write(f"cosine_similarity: {cosine_similarity.item()}\n")
                    f.write(f"\n\norginal table: \n\n")

                    f.write(tables[0].to_string(index=False))
                    f.write("\n\n The other table:\n\n\n")
                    f.write(tables[j].to_string(index=False))
                print(f"\n\nIn cloumn {i}: \n")
                print(f"cosine_similarity: {cosine_similarity.item()}\n")
                print(f"\n\norginal table: \n\n")
                print(tables[0])
                print("\n\n The other table:\n\n\n")
                print(tables[j])
                # assert False, f"cosine_similarity: {cosine_similarity.item()}\n"
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = torch.mean(
            torch.tensor(column_cosine_similarities))

        mcv = compute_mcv(torch.stack(column_embeddings))

        avg_cosine_similarities.append(avg_cosine_similarity.item())

        mcvs.append(mcv)

    table_avg_cosine_similarity = torch.mean(

        torch.tensor(avg_cosine_similarities))

    table_avg_mcv = torch.mean(torch.tensor(mcvs))

    return avg_cosine_similarities, mcvs, table_avg_cosine_similarity.item(), table_avg_mcv.item()


def process_table_wrapper(table_index, table, args, model_name, model, device):

    save_directory_results = os.path.join(

        '/nfs/turbo/coe-jag/zjsun', 'col_insig', args.save_directory, model_name, 'results')

    save_directory_embeddings = os.path.join(

        '/nfs/turbo/coe-jag/zjsun', 'col_insig', args.save_directory, model_name, 'embeddings')

    # save_directory_results  = os.path.join( args.save_directory, model_name ,'results')

    # save_directory_embeddings  = os.path.join( args.save_directory, model_name ,'embeddings')

    # Create the directories if they don't exist

    if not os.path.exists(save_directory_embeddings):

        os.makedirs(save_directory_embeddings)

    if not os.path.exists(save_directory_results):

        os.makedirs(save_directory_results)

    all_shuffled_embeddings, tables = generate_col_shuffle_embeddings(
        model, table, args.num_shuffles)

    torch.save(all_shuffled_embeddings, os.path.join(

        save_directory_embeddings, f"table_{table_index}_embeddings.pt"))

    avg_cosine_similarities, mcvs, table_avg_cosine_similarity, table_avg_mcv = analyze_embeddings(
        all_shuffled_embeddings,tables)

    results = {

        "avg_cosine_similarities": avg_cosine_similarities,

        "mcvs": mcvs,

        "table_avg_cosine_similarity": table_avg_cosine_similarity,

        "table_avg_mcv": table_avg_mcv

    }

    print(f"Table {table_index}:")

    print("Average Cosine Similarities:", results["avg_cosine_similarities"])

    print("MCVs:", results["mcvs"])

    print("Table Average Cosine Similarity:",

          results["table_avg_cosine_similarity"])

    print("Table Average MCV:", results["table_avg_mcv"])

    torch.save(results, os.path.join(

        save_directory_results, f"table_{table_index}_results.pt"))


def process_and_save_embeddings(model_name, args, tables):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"

    model = Doduo(model_args, basedir=".")

    for table_index, table in enumerate(tables):
        if table_index < args.table_num:
            continue

        try:
            process_table_wrapper(table_index, table, args,
                              model_name, model, device)
        except Exception as e:
            print("Error message:", e)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            print(table.columns)
            print(table)
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description='Process tables and save embeddings.')

    parser.add_argument('-r', '--read_directory', type=str,

                        required=True, help='Directory to read tables from')

    parser.add_argument('-s', '--save_directory', type=str,

                        required=True, help='Directory to save embeddings to')

    parser.add_argument('-n', '--num_shuffles', type=int, required=True,

                        help='Number of times to shuffle and save embeddings')

    parser.add_argument('-m', '--model_name', type=str,

                        default="", help='Name of the Hugging Face model to use')
    parser.add_argument('-t', '--table_num', type=int,

                        default=0, help='num of start table')
    args = parser.parse_args()
    pd.set_option('display.max_columns', None)  
    pd.set_option('display.max_rows', None)
    table_files = [f for f in os.listdir(

        args.read_directory) if f.endswith('.csv')]
    with open('table_list', 'w') as file:
        for index, filename in enumerate(table_files, start=0):
            file.write(f'Table {index}: {filename}\n')
    normal_tables = []

    for file in table_files:

        table = pd.read_csv(

            f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    model_name = args.model_name
    model_name = args.model_name
    print()
    with open('output.txt', 'w') as f:
            f.write(f"Evaluate  for: {model_name}\n")

    print("Evaluate  for: ", model_name)
    print()

    process_and_save_embeddings(model_name, args, normal_tables)
