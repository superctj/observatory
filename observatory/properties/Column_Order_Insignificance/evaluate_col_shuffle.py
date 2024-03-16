import argparse
import itertools
import os
import random

import pandas as pd
import torch
import torch.nn as nn

from observatory.common_util.mcv import compute_mcv
from observatory.common_util.truncate import truncate_index
from observatory.models.hugging_face_column_embeddings import (
    get_hugging_face_column_embeddings_batched,
)
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)


def fisher_yates_shuffle(seq: list) -> tuple:
    """Shuffles a sequence using the Fisher-Yates algorithm.

    Args:
        seq: A sequence to shuffle.

    Returns:
        seq: A shuffled sequence in place.
    """

    for i in reversed(range(1, len(seq))):
        j = random.randint(0, i)
        seq[i], seq[j] = seq[j], seq[i]

    return tuple(seq)


def get_permutations(n: int, m: int) -> list[list]:
    """Generates m unique permutations of the sequence [0, 1, ..., n-1].

    The original sequence is not included in the m permutations.

    Args:
        n: The length of the sequence.
        m: The number of unique permutations to generate. If m > n! - 1, all
            possible permutations are returned.

    Returns:
        uniq_permuts: A list of m+1 or n! (whichever is smaller) unique
            sequences with the original sequence at the start.
    """

    if n < 10:
        # Generate all permutations
        all_permuts = list(itertools.permutations(range(n)))

        # Remove the original sequence
        all_permuts.remove(tuple(range(n)))

        # Shuffle the permutations
        random.shuffle(all_permuts)

        # If m > n! - 1 (we removed the original sequence), return all
        # permutations
        if m > len(all_permuts):
            return [list(range(n))] + all_permuts
        else:
            return [list(range(n))] + all_permuts[:m]
    else:
        original_seq = tuple(range(n))
        uniq_permuts = set([original_seq])

        for _ in range(m):
            while True:
                new_permut = fisher_yates_shuffle(list(original_seq))

                if new_permut not in uniq_permuts:
                    uniq_permuts.add(new_permut)
                    break

        return uniq_permuts


def shuffle_df_columns(
    df: pd.DataFrame, m: int
) -> tuple[list[pd.DataFrame], list[list[int]]]:
    """Shuffles the columns of a dataframe by at most m+1 permutations.

    Args:
        df: A dataframe to shuffle.
        m: The number of unique permutations to generate excluding the original
            sequence.

    Returns:
        dfs: A list of column-wise shuffled dataframes.
        permuts: A list of permutations used to shuffle the columns.
    """

    # Get m+1 permutations (+1 because of the original sequence)
    uniq_permuts = get_permutations(len(df.columns), m)

    # Create a new dataframe for each permutation
    dfs = []
    for permut in uniq_permuts:
        dfs.append(df.iloc[:, list(permut)])

    return dfs, uniq_permuts


def analyze_embeddings(
    all_embeddings: list[list[torch.FloatTensor]],
) -> tuple[list[float], list[float], float, float]:
    """Analyzes column embedding populations induced by permutations.

    Computes the average of pairwise cosine similarities and multivariate
    coefficient of variation (MCV) for each column embedding population.

    Args:
        all_embeddings: A list of lists of column embeddings where each list
            contains column embeddings of a table induced by a permutation.

    Returns:
        avg_cosine_similarities: A list of the average pairwise cosine
            similarities. E.g., avg_cosine_similarities[0] is the average of
            pairwise cosine similarities of the embedding population induced by
            the first column in the original table.
        mcvs: A list of MCV values. E.g., mcvs[0] is the MCV of the embedding
            population induced by the first column in the original table.
        table_avg_cosine_similarity: The cosine similarity averaged over
            columns, i.e., the average of `avg_cosine_similarities`.
        table_avg_mcv: The MCV value averaged over columns, i.e., the average
            of `mcvs`.
    """

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

            cosine_similarity = nn.functional.cosine_similarity(
                truncated_embedding, shuffled_embedding, dim=0
            )
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = sum(column_cosine_similarities) / len(
            column_cosine_similarities
        )
        mcv = compute_mcv(torch.stack(column_embeddings))

        avg_cosine_similarities.append(avg_cosine_similarity)
        mcvs.append(mcv)

    table_avg_cosine_similarity = sum(avg_cosine_similarities) / len(
        avg_cosine_similarities
    )
    table_avg_mcv = sum(mcvs) / len(mcvs)

    return (
        avg_cosine_similarities,
        mcvs,
        table_avg_cosine_similarity,
        table_avg_mcv,
    )


def process_table_wrapper(
    table_index: int,
    truncated_table: pd.DataFrame,
    model_name: str,
    model,
    tokenizer,
    max_length: int,
    padding_token: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Processes a single table and saves the embeddings and results.

    Args:
        table_index: The index of the table to process.
        truncated_table: The table to process.
        model_name: The name of a Hugging Face model.
        model: A Hugging Face model for embedding inference.
        tokenizer: A Hugging Face tokenizer.
        max_length: The maximum length of input tokens.
        padding_token: The padding token.
        args: The command-line arguments.
        device: The torch device.

    Returns:
        None (saves the embeddings and results to the specified directories).
    """

    # Create the directories if they do not exist
    save_directory_results = os.path.join(
        args.save_directory,
        "Column_Order_Insignificance",
        model_name,
        "results",
    )
    save_directory_embeddings = os.path.join(
        args.save_directory,
        "Column_Order_Insignificance",
        model_name,
        "embeddings",
    )

    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)

    tables, perms = shuffle_df_columns(truncated_table, args.num_shuffles)
    all_embeddings = get_hugging_face_column_embeddings_batched(
        tables=tables,
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
        batch_size=args.batch_size,
    )

    all_ordered_embeddings = []

    for perm, embeddings in zip(perms, all_embeddings):
        # Create a list of the same length as perm, filled with None
        ordered_embeddings = [None] * len(perm)
        # Assign each embedding to its original position
        for i, p in enumerate(perm):
            ordered_embeddings[p] = embeddings[i]

        all_ordered_embeddings.append(ordered_embeddings)

    all_embeddings = all_ordered_embeddings
    torch.save(
        all_embeddings,
        os.path.join(
            save_directory_embeddings, f"table_{table_index}_embeddings.pt"
        ),
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
    print(
        "Table Average Cosine Similarity:",
        results["table_avg_cosine_similarity"],
    )
    print("Table Average MCV:", results["table_avg_mcv"])

    torch.save(
        results,
        os.path.join(save_directory_results, f"table_{table_index}_results.pt"),
    )


def process_and_save_embeddings(
    model_name: str, tables: list[pd.DataFrame], args: argparse.Namespace
) -> None:
    """Processes the tables and saves the embeddings and results.

    Args:
        model_name: The name of a Hugging Face model for embedding inference.
        tables: A list of tables in dataframes to process.
        args: Command-line arguments.

    Returns:
        None (saves the embeddings and results to the specified directories).
    """

    tokenizer, max_length = load_transformers_tokenizer_and_max_length(
        model_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformers_model(model_name, device)
    model.eval()

    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    for table_index, table in enumerate(tables):
        if table_index < args.start_index:
            continue

        max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]

        process_table_wrapper(
            table_index,
            truncated_table,
            model_name,
            model,
            tokenizer,
            max_length,
            padding_token,
            args,
            device,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process tables and save embeddings."
    )
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

    table_files = [
        f for f in os.listdir(args.read_directory) if f.endswith(".csv")
    ]
    normal_tables = []

    for file in table_files:
        table = pd.read_csv(
            f"{args.read_directory}/{file}", keep_default_na=False
        )
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

    print(f"\nEvaluate row shuffle for: {model_names}\n")

    for model_name in model_names:
        process_and_save_embeddings(model_name, normal_tables, args)
