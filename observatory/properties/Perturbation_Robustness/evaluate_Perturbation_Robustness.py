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
from readCompare import compare_directories

# from concurrent.futures import ThreadPoolExecutor
from torch.linalg import inv, norm


# def get_subsets(n, m, portion):
#     portion_size = int(n * portion)
#     max_possible_tables = math.comb(n, portion_size)

#     if max_possible_tables <= 10 * m:
#         # If the number of combinations is small, generate all combinations and randomly select from them
#         all_subsets = list(itertools.combinations(range(n), portion_size))
#         random.shuffle(all_subsets)
#         return [list(subset) for subset in all_subsets[:m]]
#     else:
#         # If the number of combinations is large, use random sampling to generate distinct subsets
#         subsets = set()
#         while len(subsets) < min(m, max_possible_tables):
#             new_subset = tuple(sorted(random.sample(range(n), portion_size)))
#             subsets.add(new_subset)
#         return [list(subset) for subset in subsets]

# def shuffle_df(df, m, portion):
#     subsets = get_subsets(len(df), m, portion)
#     dfs = []
#     for subset in subsets:
#         dfs.append(df.iloc[subset])
#     return dfs


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


def generate_p4_embeddings(tokenizer, model, device, max_length, padding_token, tables):
    all_embeddings = []
    # sampled_tables = shuffle_df(table,num_samples, percentage )

    for processed_table in tables:
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

        all_embeddings.append(embeddings)

    return all_embeddings


def analyze_embeddings(all_embeddings, changed_column_lists):
    cosine_similarities_dict = {}

    for table_index, changed_columns in enumerate(changed_column_lists):
        for column_index in changed_columns:
            original_embedding = all_embeddings[0][column_index]
            shuffled_embedding = all_embeddings[table_index + 1][column_index]

            cosine_similarity = torch.dot(original_embedding, shuffled_embedding) / (
                norm(original_embedding) * norm(shuffled_embedding)
            )

            if column_index not in cosine_similarities_dict:
                cosine_similarities_dict[column_index] = []

            cosine_similarities_dict[column_index].append(cosine_similarity.item())

    return cosine_similarities_dict


def process_table_wrapper(
    truncated_tables,
    args,
    model_name,
    model,
    tokenizer,
    device,
    max_length,
    padding_token,
    key,
    changed_column_list,
):
    save_directory_results = os.path.join(
        args.save_directory, "Perturbation_Robustness", model_name, "results"
    )
    save_directory_embeddings = os.path.join(
        args.save_directory, "Perturbation_Robustness", model_name, "embeddings"
    )

    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)

    all_shuffled_embeddings = generate_p4_embeddings(
        tokenizer, model, device, max_length, padding_token, truncated_tables
    )
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
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = load_transformers_model(model_name, device)
    model.eval()
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    for key, value in result_dict.items():
        truncated_tables = []
        list_max_rows_fit = []
        for table in value[0]:
            max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
            list_max_rows_fit.append(max_rows_fit)
        min_max_rows_fit = min(list_max_rows_fit)

        for table in value[0]:
            truncated_table = table.iloc[:min_max_rows_fit, :]
            truncated_tables.append(truncated_table)
        process_table_wrapper(
            truncated_tables,
            args,
            model_name,
            model,
            tokenizer,
            device,
            max_length,
            padding_token,
            key,
            value[1],
        )


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

    if args.model_name == "":
        model_names = ["bert-base-uncased", "roberta-base", "t5-base"]
    else:
        model_names = [args.model_name]
    print()
    print("Evaluate row shuffle for: ", model_names)
    print()

    for model_name in model_names:
        process_and_save_embeddings(model_name, args, result_dict)

# def process_and_save_embeddings(model_name, args, tables):
#     tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     model = load_transformers_model(model_name, device)
#     model.eval()
#     padding_token = '<pad>' if model_name.startswith("t5") else '[PAD]'

#     # Use ThreadPoolExecutor for parallel processing
#     with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
#         futures = []
#         for table_index, table in enumerate(tables):
#             max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
#             truncated_table = table.iloc[:max_rows_fit, :]
#             futures.append(executor.submit(process_table_wrapper, table_index, truncated_table, args, model_name, model, tokenizer, device, max_length, padding_token))

#         # Wait for all tasks to complete
#         for future in futures:
#             future.result()
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
#     parser.add_argument('-r', '--read_directory', type=str, required=True, help='Directory to read tables from')
#     parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
#     parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of times to shuffle and save embeddings')
#     parser.add_argument('-w', '--num_workers', type=int, default=4, help="Number of worker threads for parallel processing. For example: -n 4")
#     parser.add_argument('-m', '--model_name', type=str, default="", help='Name of the Hugging Face model to use')
#     parser.add_argument('-p', '--sample_portion', type=float, default=0.25, help='Portion of sample to use')

#     args = parser.parse_args()

#     table_files = [f for f in os.listdir(args.read_directory) if f.endswith('.csv')]
#     normal_tables = []
#     for file in table_files:
#         table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
#         normal_tables.append(table)

#     if args.model_name == "":
#         model_names = [ 'bert-base-uncased', 'roberta-base',  't5-base', 'google/tapas-base']
#     else:
#         model_names =[args.model_name]
#     print()
#     print("Evaluate row shuffle for: ",model_names)
#     print()

#     for model_name in model_names:
#             process_and_save_embeddings(model_name, args, normal_tables)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
#     parser.add_argument('-r', '--read_directory', type=str, required=True, help='Directory to read tables from')
#     parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
#     parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the Hugging Face model to use')
#     parser.add_argument('-n', '--num_samples', type=int, required=True, help='Number of times to shuffle and save embeddings')
#     args = parser.parse_args()

#     tokenizer, max_length = load_transformers_tokenizer_and_max_length(args.model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_transformers_model(args.model_name, device)

#     table_files = [f for f in os.listdir(args.read_directory) if f.endswith('.csv')]
#     normal_tables = []
#     for file in table_files:
#         table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
#         normal_tables.append(table)

#     if not os.path.exists(args.save_directory):
#         os.makedirs(args.save_directory)
#     padding_token = '<pad>' if args.model_name.startswith("t5") else '[PAD]'

#     for i, table in enumerate(normal_tables):
#         max_rows_fit = truncate_index(table, tokenizer, max_length, args.model_name)
#         truncated_table = table.iloc[:max_rows_fit, :]

#         table_dir = os.path.join(args.save_directory, f"table_{i + 1}")
#         if not os.path.exists(table_dir):
#             os.makedirs(table_dir)

#         for j in range(args.num_samples):
#             shuffled_table = row_shuffle(truncated_table)
#             col_list = table2colList(shuffled_table)
#             processed_table = process_table(tokenizer, col_list, max_length, args.model_name)
#             input_ids = tokenizer.convert_tokens_to_ids(processed_table[0])
#             attention_mask = [1 if token != padding_token else 0 for token in processed_table[0]]
#             cls_positions = processed_table[1]

#             input_ids_tensor = torch.tensor([input_ids], device=device)
#             attention_mask_tensor = torch.tensor([attention_mask], device=device)

#             outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
#             last_hidden_state = outputs.last_hidden_state

#             embeddings = []
#             for position in cls_positions:
#                 cls_embedding = last_hidden_state[0, position, :].detach().cpu().numpy()
#                 embeddings.append(cls_embedding)

#             np.save(os.path.join(table_dir, f"shuffle_{j + 1}_embeddings.npy"), embeddings)
