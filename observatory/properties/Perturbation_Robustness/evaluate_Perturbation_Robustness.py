#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import pandas as pd
import torch

from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.models.hugging_face_column_embeddings import get_hugging_face_column_embeddings_batched
from observatory.common_util.truncate import truncate_index
from readCompare import compare_directories

# from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn 


def analyze_embeddings(
    all_embeddings: list[list[torch.FloatTensor]], 
    changed_column_lists: list[list[int]]
) -> dict[int, list[float]]:
    """ Analyze the embeddings and return a dictionary of cosine similarities for each column.
    
    Args:
        all_embeddings: A list of lists of embeddings for each table.
        changed_column_lists: A list of lists of column indices that were changed.
        
    Returns:
        cosine_similarities_dict: A dictionary of cosine similarities for each column.
    """
    cosine_similarities_dict = {}

    for table_index, changed_columns in enumerate(changed_column_lists):
        for column_index in changed_columns:
            original_embedding = all_embeddings[0][column_index]
            shuffled_embedding = all_embeddings[table_index + 1][column_index]

            cosine_similarity = nn.functional.cosine_similarity(
                original_embedding, shuffled_embedding, dim=0
            )

            if column_index not in cosine_similarities_dict:
                cosine_similarities_dict[column_index] = []

            cosine_similarities_dict[column_index].append(cosine_similarity.item())

    return cosine_similarities_dict


def process_table_wrapper(
    truncated_tables: list[pd.DataFrame],
    args: argparse.Namespace,
    model_name: str,
    model,
    tokenizer,
    device: torch.device,
    max_length: int,
    padding_token: str,
    key: str,
    changed_column_list: list[int],
)  -> None:
    """Processes a single table and saves the embeddings and results.
    
    Args:
        truncated_tables: A list of truncated DataFrames.
        args: The command line arguments.
        model_name: The name of the Hugging Face model to use.
        model: The Hugging Face model.
        tokenizer: The Hugging Face tokenizer.
        device: The device to use for processing.
        max_length: The maximum length of the input sequence.
        padding_token: The padding token for the tokenizer.
        key: The key for the table in the result dictionary.
        changed_column_list: A list of column indices that were changed.
    
    Returns:
        None (saves the embeddings and results to the specified directories).
    """
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

    all_embeddings = get_hugging_face_column_embeddings_batched(
        truncated_tables, model_name=model_name, tokenizer=tokenizer, max_length=max_length, model=model, batch_size=args.batch_size
    )
    torch.save(
        all_embeddings,
        os.path.join(save_directory_embeddings, f"{key}_embeddings.pt"),
    )
    cosine_similarities_dict = analyze_embeddings(
        all_embeddings, changed_column_list
    )
    for column_index, similarities in cosine_similarities_dict.items():
        print(f"Column {column_index}:")
        for i, similarity in enumerate(similarities):
            print(f"\tCosine similarity with modified table {i+1}: {similarity}")
    torch.save(
        cosine_similarities_dict,
        os.path.join(save_directory_results, f"{key}_results.pt"),
    )


def process_and_save_embeddings(
    model_name: str, 
    args: argparse.Namespace, 
    result_dict: dict[str, tuple[pd.DataFrame, list[int]]]
) -> None:
    """Processes and saves the embeddings for the tables in the result dictionary.
    
    Args:
        model_name: The name of the Hugging Face model to use.
        args: The command line arguments.
        result_dict: A dictionary of table keys and values.
    
    Returns:
        None (saves the embeddings and results to the specified directories).
    """
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
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The batch size for parallel inference",
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
