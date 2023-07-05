import argparse
import random
random.seed(12345)
import time
from itertools import permutations
# import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch.linalg import norm
import itertools

from doduo import Doduo

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
# def generate_permutations(n, cut_off=1000):
#     visited = set([])
#     permutations = []

#     def dfs_helper(path):
#         if len(path) == n+1:
#             permutations.append(list(path))
#         else:
#             for i in range(n+1):
#                 if i not in visited:
#                     visited.add(i)
#                     path.append(i)
#                     dfs_helper(path)
#                     path.pop()
#                     visited.remove(i)

#     dfs_helper([])
#     assert(len(permutations) == math.factorial(n+1))
#     return permutations[:cut_off]


def numpy_version(doduo, input_table):
    num_rows = input_table.shape[0]
    num_cols = input_table.shape[1]
    all_embeddings = []

    num_valid_tables = 0
    for i, perm in enumerate(permutations(range(num_rows))):
        # print("Processing permutation: ", i)
        shuffled_table = input_table.iloc[list(perm)].reset_index(drop=True)
        try:
            annot_table = doduo.annotate_columns(shuffled_table)
        except RuntimeError as e:
            # print("Skip permutation: ", i)
            # print(e)
            continue

        num_valid_tables += 1
        all_embeddings.append(annot_table.colemb)
        # if num_valid_tables >= 3:
        #     break

    all_embeddings = np.array(all_embeddings)
    print("Number of rows: ", num_rows)
    print("Number of columns: ", num_cols)
    print("Number of valid permutations: ", num_valid_tables)
    print(all_embeddings.shape)

    for i in range(num_cols):
        i_similarities = cosine_similarity(all_embeddings[0, i, :].reshape(1, -1), all_embeddings[1:, i, :])

        print(i_similarities.shape)
        print(np.sum(i_similarities) / i_similarities.shape[1])


def zhenjie_torch_version(doduo, input_table):
    num_rows = input_table.shape[0]
    num_cols = input_table.shape[1]
    all_embeddings = []

    num_valid_tables = 0
    # for i, perm in enumerate(permutations(range(num_rows))):
    for shuffled_table in shuffle_df(input_table, 1000):
        # print("Processing permutation: ", i)
        # shuffled_table = input_table.iloc[list(perm)].reset_index(drop=True)
        shuffled_table = shuffled_table.reset_index(drop=True)
        shuffled_table = shuffled_table.astype(str)
        try:
            annot_table = doduo.annotate_columns(shuffled_table)
        except RuntimeError as e:
            # print("Skip permutation: ", i)
            # print(e)
            continue

        num_valid_tables += 1
        col_embeddings = annot_table.colemb
        col_embeddings = [torch.tensor(col_embeddings[j]) for j in range(len(col_embeddings))]
        all_embeddings.append(col_embeddings)

    print("Number of rows: ", num_rows)
    print("Number of columns: ", num_cols)
    print("Number of valid permutations: ", num_valid_tables)

    avg_cosine_similarities = []
    for i in range(len(all_embeddings[0])):
        column_cosine_similarities = []
        for j in range(1, len(all_embeddings)):
            truncated_embedding = all_embeddings[0][i]
            shuffled_embedding = all_embeddings[j][i]

            cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (norm(truncated_embedding) * norm(shuffled_embedding))

            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = torch.mean(torch.tensor(column_cosine_similarities))
        avg_cosine_similarities.append(avg_cosine_similarity.item())

    print(avg_cosine_similarities)

from truncate import truncate_index
from huggingface_models import  load_transformers_model, load_transformers_tokenizer_and_max_length

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="wikitable",
        type=str,
        choices=["wikitable", "viznet"],
        help="Pretrained model"
    )
    parser.add_argument(
        "--input",
        default="/home/zjsun/data/normal_TD/table_4209.csv",
        type=str,
        help="Input csv file path."
    )
    args = parser.parse_args()

    doduo = Doduo(args, basedir=".")
    
    model_name = 'bert-base-uncased'
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    original_table = pd.read_csv(args.input)
    max_rows_fit = truncate_index(original_table, tokenizer, max_length, model_name)
    print("Max_row_fit: ", max_rows_fit)    

    
    input_table = pd.read_csv(args.input, nrows=max_rows_fit)
    start = time.time()
    zhenjie_torch_version(doduo, input_table)
    end = time.time()
    # print(end - start)
    # start = time.time()
    # numpy_version(doduo, input_table)
    # end = time.time()
    # print(end - start)