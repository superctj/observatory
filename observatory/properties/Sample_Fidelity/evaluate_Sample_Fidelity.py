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


def table2colList(table):
    cols = []
    for column in table.columns:
        # Convert column values to strings and join them with spaces
        string_values = " ".join(table[column].astype(str).tolist())
        col_str = f"{column} {string_values}"
        cols.append(col_str)
    return cols


def process_table(tokenizer, cols, max_length, model_name):
    current_tokens = []
    cls_positions = []

    for idx, col in enumerate(cols):
        col_tokens = tokenizer.tokenize(col)
        # Check model name and use appropriate special tokens
        if model_name.startswith("t5"):
            # For T5, add <s> at the start and </s> at the end
            col_tokens = ["<s>"] + col_tokens + ["</s>"]
        else:
            # For other models (BERT, RoBERTa, TAPAS), add [CLS] at the start and [SEP] at the end
            col_tokens = ["[CLS]"] + col_tokens + ["[SEP]"]

        if len(current_tokens) + len(col_tokens) > max_length:
            assert (
                False
            ), "The length of the tokens exceeds the max length. Please run the truncate.py first."
            break
        else:
            if current_tokens:
                current_tokens = current_tokens[:-1]
            current_tokens += col_tokens
            cls_positions.append(
                len(current_tokens) - len(col_tokens)
            )  # Store the position of [CLS]

    if len(current_tokens) < max_length:
        padding_length = max_length - len(current_tokens)
        # Use appropriate padding token based on the model
        padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
        current_tokens += [padding_token] * padding_length

    return current_tokens, cls_positions


def tapas_column_embeddings(inputs, last_hidden_states):
    # find the maximum column id
    max_column_id = inputs["token_type_ids"][0][:, 1].max()

    column_embeddings = []

    # loop over all column ids
    for column_id in range(1, max_column_id + 1):
        # find all indices where the token_type_ids is equal to the column id
        indices = torch.where(inputs["token_type_ids"][0][:, 1] == column_id)[0]

        # get the embeddings at these indices
        embeddings = last_hidden_states[0][indices]

        # compute the average embedding
        column_embedding = embeddings.mean(dim=0)

        column_embeddings.append(column_embedding)

    return column_embeddings


def generate_row_sample_embeddings(
    tokenizer, model, device, max_length, padding_token, table, num_samples, percentage
):
    all_shuffled_embeddings = []
    sampled_tables = shuffle_df(table, num_samples, percentage)

    for processed_table in sampled_tables:
        if model_name.startswith("google/tapas"):
            processed_table = processed_table.reset_index(drop=True)
            processed_table = processed_table.astype(str)

            inputs = tokenizer(
                table=processed_table, padding="max_length", return_tensors="pt"
            )
            inputs = inputs.to(device)
            with torch.no_grad():  # Turn off gradients to save memory
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            embeddings = tapas_column_embeddings(inputs, last_hidden_states)
        else:
            col_list = table2colList(processed_table)
            processed_tokens = process_table(
                tokenizer, col_list, max_length, model.name_or_path
            )
            input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
            attention_mask = [
                1 if token != padding_token else 0 for token in processed_tokens[0]
            ]
            cls_positions = processed_tokens[1]

            input_ids_tensor = torch.tensor([input_ids], device=device)
            attention_mask_tensor = torch.tensor([attention_mask], device=device)

            if model.name_or_path.startswith("t5"):
                outputs = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    decoder_input_ids=input_ids_tensor,
                )
            else:
                outputs = model(
                    input_ids=input_ids_tensor, attention_mask=attention_mask_tensor
                )
            last_hidden_state = outputs.last_hidden_state

            embeddings = []
            for position in cls_positions:
                cls_embedding = last_hidden_state[0, position, :].detach().cpu()
                embeddings.append(cls_embedding)

        all_shuffled_embeddings.append(embeddings)

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

    all_shuffled_embeddings = generate_row_sample_embeddings(
        tokenizer,
        model,
        device,
        max_length,
        padding_token,
        truncated_table,
        args.num_samples,
        args.sample_portion,
    )

    save_file_path = os.path.join(
        save_directory_embeddings, f"table_{table_index}_embeddings.pt"
    )
    # If the file exists, load it and substitute the elements.
    if os.path.exists(save_file_path):
        existing_embeddings = torch.load(save_file_path)

        # Ensure that existing_embeddings is long enough
        if len(existing_embeddings) < len(all_shuffled_embeddings):
            existing_embeddings = all_shuffled_embeddings
        else:
            # Substitute the elements
            existing_embeddings[
                : len(all_shuffled_embeddings)
            ] = all_shuffled_embeddings

        # Save the modified embeddings
        torch.save(existing_embeddings, save_file_path)
    else:
        # If the file doesn't exist, just save all_shuffled_embeddings
        torch.save(all_shuffled_embeddings, save_file_path)

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