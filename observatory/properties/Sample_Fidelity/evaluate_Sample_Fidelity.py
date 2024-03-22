import argparse
import itertools
import math
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


def get_subsets(n: int, m: int, frac: float) -> list[list[int]]:
    """Get up to m distinct subsets of size n * frac from the set {0, 1, ...,
    n - 1}.

    If the total number of unique subsets is small (smaller than 10 * m),
    generate all combinations and randomly select from them; else, use random
    sampling to generate distinct subsets.

    Args:
        n: The size of the whole set.
        m: The number of subsets to return.
        frac: Sampling fraction.

    Returns:
        A list of up to m distinct subsets of size n * frac.
    """

    subset_size = int(n * frac)
    num_uniq_subsets = math.comb(n, subset_size)

    if num_uniq_subsets <= 10 * m:
        all_subsets = list(itertools.combinations(range(n), subset_size))
        random.shuffle(all_subsets)
        return [list(subset) for subset in all_subsets[:m]]
    else:
        subsets = set()

        while len(subsets) < min(m, num_uniq_subsets):
            new_subset = tuple(sorted(random.sample(range(n), subset_size)))
            subsets.add(new_subset)

        return [list(subset) for subset in subsets]


def sample_df(
    df: pd.DataFrame, m: int, frac: float
) -> tuple[list[pd.DataFrame], list[list[int]]]:
    """Get up to m unqiue samples of a table at a sampling fraction.

    Args:
        df: A dataframe to sample.
        m: The number of unique samples to generate.
        frac: The sampling fraction.

    Returns:
        dfs: A list of at most m row-wise sampled dataframes plus the orginal
          dataframes at the beginning, i.e., dfs[0].
        uniq_permuts: A list of row identifiers of each sample plus the
          original row order at the begining, i.e., uniq_permuts[0]).
    """

    subsets = get_subsets(len(df), m, frac)
    uniq_permuts = [list(range(len(df)))] + subsets
    dfs = [df]

    for subset in subsets:
        dfs.append(df.iloc[subset])

    return dfs, uniq_permuts


# TODO: This is a duplicate function that can be extracted out
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
        colwise_avg_cosine_similarities: Average pairwise cosine similarities
            per column. E.g., colwise_avg_cosine_similarities[0] is the
            average of pairwise cosine similarities of the embedding population
            induced by the first column in the original table.
        colwise_mcvs: MCV values per column. E.g., colwise_mcvs[0] is the MCV
            value of the embedding population induced by the first column in the
            original table.
        table_avg_cosine_similarity: The cosine similarity averaged over
            columns, i.e., the average of `colwise_avg_cosine_similarities`.
        table_avg_mcv: The MCV value averaged over columns, i.e., the average
            of `colwise_mcvs`.
    """

    colwise_avg_cosine_similarities = []
    colwise_mcvs = []

    for i in range(len(all_embeddings[0])):
        column_cosine_similarities = []

        # iterate over permutations
        for j in range(1, len(all_embeddings)):
            # embedding of a column inferred from the original table
            col_embedding = all_embeddings[0][i]
            # embedding of the same column inferred from a shuffled table
            shuffled_embedding = all_embeddings[j][i]

            cosine_similarity = nn.functional.cosine_similarity(
                col_embedding, shuffled_embedding, dim=0
            )
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = sum(column_cosine_similarities) / len(
            column_cosine_similarities
        )
        colwise_avg_cosine_similarities.append(avg_cosine_similarity)

        # compute MCV
        column_embeddings = []

        for j in range(len(all_embeddings)):
            column_embeddings.append(all_embeddings[j][i])

        mcv = compute_mcv(torch.stack(column_embeddings))
        colwise_mcvs.append(mcv)

    table_avg_cosine_similarity = sum(colwise_avg_cosine_similarities) / len(
        colwise_avg_cosine_similarities
    )
    table_avg_mcv = sum(colwise_mcvs) / len(colwise_mcvs)

    return (
        colwise_avg_cosine_similarities,
        colwise_mcvs,
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
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Processes a single table and saves the embeddings and results.

    Args:
        table_index: The index of the table to process.
        truncated_table: The table to process.
        args: The command-line arguments.
        model_name: The name of a Hugging Face model.
        model: A Hugging Face model for embedding inference.
        tokenizer: A Hugging Face tokenizer.
        device: The torch device.
        max_length: The maximum length of input tokens.
        padding_token: The padding token.

    Returns:
        None (saves the embeddings and results to the specified directories).
    """

    save_directory_results = os.path.join(
        args.save_directory,
        "Sample_Fidelity",
        str(args.sample_portion),
        model_name,
        "results",
    )
    save_directory_embeddings = os.path.join(
        args.save_directory,
        "Sample_Fidelity",
        str(args.sample_portion),
        model_name,
        "embeddings",
    )

    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)

    sampled_tables, _ = sample_df(
        truncated_table, args.num_samples, args.sample_portion
    )
    all_embeddings = get_hugging_face_column_embeddings_batched(
        tables=sampled_tables,
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        model=model,
        batch_size=args.batch_size,
        device=device,
    )

    save_file_path = os.path.join(
        save_directory_embeddings, f"table_{table_index}_embeddings.pt"
    )

    # If the file exists, load it and substitute the elements.
    if os.path.exists(save_file_path):
        existing_embeddings = torch.load(save_file_path)

        # Ensure that existing_embeddings is long enough
        if len(existing_embeddings) < len(all_embeddings):
            existing_embeddings = all_embeddings
        else:
            # Substitute the elements
            existing_embeddings[: len(all_embeddings)] = all_embeddings

        # Save the modified embeddings
        torch.save(existing_embeddings, save_file_path)
    else:
        # If the file doesn't exist, just save all_shuffled_embeddings
        torch.save(all_embeddings, save_file_path)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformers_model(model_name, device)

    tokenizer, max_length = load_transformers_tokenizer_and_max_length(
        model_name
    )

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
        "--num_samples",
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
        "-p",
        "--sample_portion",
        type=float,
        default=0.25,
        help="Portion of sample to use",
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
    all_tables = []

    for f in table_files:
        table = pd.read_csv(
            os.path.join(args.read_directory, f), keep_default_na=False
        )
        all_tables.append(table)

    if args.model_name == "":
        all_model_names = ["bert-base-uncased", "roberta-base", "t5-base"]
    else:
        all_model_names = [args.model_name]

    for model_name in all_model_names:
        print(f"\nEvaluate sample fidelity for: {model_name}\n")
        process_and_save_embeddings(model_name, all_tables, args)
