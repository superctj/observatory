#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import pandas as pd
import torch
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.truncate import truncate_index
from torch.linalg import inv, norm
from observatory.common_util.mcv import compute_mcv
import random
import math
import itertools
from observatory.models.hugging_face_column_embeddings import get_hugging_face_column_embeddings_batched


def get_subsets(n, m, portion):
    portion_size = int(n * portion)
    max_possible_tables = math.comb(n, portion_size)

    if max_possible_tables <= 10 * m:
        # If the number of combinations is small, generate all combinations and randomly select from them
        all_subsets = list(itertools.combinations(range(n), portion_size))
        random.shuffle(all_subsets)
        return [list(subset) for subset in all_subsets[:m]]
    else:
        # If the number of combinations is large, use random sampling to generate distinct subsets
        subsets = set()
        while len(subsets) < min(m, max_possible_tables):
            new_subset = tuple(sorted(random.sample(range(n), portion_size)))
            subsets.add(new_subset)
        return [list(subset) for subset in subsets]


def shuffle_df(df, m, portion):
    subsets = get_subsets(len(df), m, portion)
    dfs = [df]
    for subset in subsets:
        dfs.append(df.iloc[subset])
    return dfs



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
    # save_directory_results  = os.path.join( str(args.sample_portion), args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join( str(args.sample_portion), args.save_directory, model_name ,'embeddings')
    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    sampled_tables = shuffle_df(table, args.num_samples, args.sample_portion)

    all_embeddings = get_hugging_face_column_embeddings_batched(
        tables=sampled_tables, model_name=model_name, tokenizer=tokenizer, max_length=max_length, model=model, batch_size=args.batch_size
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
            existing_embeddings[
                : len(all_embeddings)
            ] = all_embeddings

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
        max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
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
    args = parser.parse_args()

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    if args.model_name == "":
        model_names = ["bert-base-uncased", "roberta-base", "t5-base"]
    else:
        model_names = [args.model_name]
    print()
    print("Evaluate row shuffle for: ", model_names)
    print()

    for model_name in model_names:
        process_and_save_embeddings(model_name, args, normal_tables)
