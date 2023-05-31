#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import itertools
import random

import pandas as pd
import torch
import numpy as np
from huggingface_models import load_transformers_tokenizer, load_transformers_model, load_transformers_tokenizer_and_max_length
from truncate import truncate_index
from concurrent.futures import ThreadPoolExecutor
from torch.linalg import inv, norm
from mcv import compute_mcv
from torch.serialization import save
from scipy.spatial.distance import cosine


def table2colList(table):
    cols = []
    for column in table.columns:
        # Convert column values to strings and join them with spaces
        string_values = ' '.join(table[column].astype(str).tolist())
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
            col_tokens = ['<s>'] + col_tokens + ['</s>']
        else:
            # For other models (BERT, RoBERTa, TAPAS), add [CLS] at the start and [SEP] at the end
            col_tokens = ['[CLS]'] + col_tokens + ['[SEP]']

        if len(current_tokens) + len(col_tokens) > max_length:
            assert False, "The length of the tokens exceeds the max length. Please run the truncate.py first."
            break
        else:
            if current_tokens:
                current_tokens = current_tokens[:-1]
            current_tokens += col_tokens
            # Store the position of [CLS]
            cls_positions.append(len(current_tokens) - len(col_tokens))

    if len(current_tokens) < max_length:
        padding_length = max_length - len(current_tokens)
        # Use appropriate padding token based on the model
        padding_token = '<pad>' if model_name.startswith("t5") else '[PAD]'
        current_tokens += [padding_token] * padding_length

    return current_tokens, cls_positions


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


def get_column_embeddings(inputs, last_hidden_states):
    # find the maximum column id
    max_column_id = inputs['token_type_ids'][0][:, 1].max()

    column_embeddings = []

    # loop over all column ids
    for column_id in range(1, max_column_id + 1):
        # find all indices where the token_type_ids is equal to the column id
        indices = torch.where(inputs['token_type_ids'][0][:, 1] == column_id)[0]


        # get the embeddings at these indices
        embeddings = last_hidden_states[0][indices]

        # compute the average embedding
        column_embedding = embeddings.mean(dim=0)

        column_embeddings.append(column_embedding)

    return column_embeddings

def generate_col_shuffle_embeddings(tokenizer, model, device, max_length, padding_token, table, num_shuffles):
    all_shuffled_embeddings = []
    tables, perms = shuffle_df_columns(table, num_shuffles)

    for j in range(len(tables)):
        #     if j == 0:
        #         processed_table = table
        #     else:
        #         processed_table = row_shuffle(table)
        processed_table = tables[j]

        processed_table = processed_table.reset_index(drop=True)
        processed_table = processed_table.astype(str)
        inputs = tokenizer(table=processed_table,
                           padding="max_length", return_tensors="pt")
        inputs = inputs.to(device)
        with torch.no_grad():  # Turn off gradients to save memory
            outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        embeddings = get_column_embeddings(inputs, last_hidden_states)


        # Get the permutation used for this table
        perm = perms[j]
        # Create a list of the same length as perm, filled with None
        ordered_embeddings = [None] * len(perm)
        # Assign each embedding to its original position
        for i, p in enumerate(perm):
            ordered_embeddings[p] = embeddings[i]
        all_shuffled_embeddings.append(ordered_embeddings)
        
        # Clear memory
        del inputs
        del outputs
        del last_hidden_states
        del embeddings
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
                norm(truncated_embedding) * norm(shuffled_embedding))
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


def process_table_wrapper(table_index, truncated_table, args, model_name, model, tokenizer, device, max_length, padding_token):
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

    all_shuffled_embeddings = generate_col_shuffle_embeddings(
        tokenizer, model, device, max_length, padding_token, truncated_table, args.num_shuffles)
    torch.save(all_shuffled_embeddings, os.path.join(
        save_directory_embeddings, f"table_{table_index}_embeddings.pt"))
    avg_cosine_similarities, mcvs, table_avg_cosine_similarity, table_avg_mcv = analyze_embeddings(
        all_shuffled_embeddings)
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
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(
        model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = load_transformers_model(model_name, device)
    model.eval()
    padding_token = '<pad>' if model_name.startswith("t5") else '[PAD]'

    for table_index, table in enumerate(tables):
        max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        process_table_wrapper(table_index, truncated_table, args, model_name,
                              model, tokenizer, device, max_length, padding_token)


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
    args = parser.parse_args()

    table_files = [f for f in os.listdir(
        args.read_directory) if f.endswith('.csv')]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(
            f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    if args.model_name == "":
        model_names = ['bert-base-uncased', 'roberta-base',
                       't5-base', 'google/tapas-base']
    else:
        model_names = [args.model_name]
    print()
    print("Evaluate row shuffle for: ", model_names)
    print()

    for model_name in model_names:
        process_and_save_embeddings(model_name, args, normal_tables)
#     torch.save(results, os.path.join(save_directory, f"table_{table_index}_results.pt"))
# def process_and_save_embeddings(model_name, args, tables):

#     tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)
#     model = load_transformers_model(model_name, device)
#     model.eval()
#     padding_token = '<pad>' if model_name.startswith("t5") else '[PAD]'
#     for table_index, table in enumerate(tables):
#         max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
#         truncated_table = table.iloc[:max_rows_fit, :]

#         table_dir = os.path.join(args.save_directory, model_name, f"table_{table_index}")
#         if not os.path.exists(table_dir):
#             os.makedirs(table_dir)

#         for j in range(args.num_shuffles + 1):
#             if j == 0:
#                 processed_table = truncated_table
#             else:
#                 processed_table = row_shuffle(truncated_table)

#             col_list = table2colList(processed_table)
#             processed_tokens = process_table(tokenizer, col_list, max_length, model_name)
#             input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
#             attention_mask = [1 if token != padding_token else 0 for token in processed_tokens[0]]
#             cls_positions = processed_tokens[1]

#             input_ids_tensor = torch.tensor([input_ids], device=device)
#             attention_mask_tensor = torch.tensor([attention_mask], device=device)

#             if model_name.startswith("t5"):
#                 outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, decoder_input_ids=input_ids_tensor)
#             else:
#                 outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
#             last_hidden_state = outputs.last_hidden_state

#             embeddings = []
#             for position in cls_positions:
#                 cls_embedding = last_hidden_state[0, position, :].detach().cpu().numpy()
#                 embeddings.append(cls_embedding)

#             np.save(os.path.join(table_dir, f"shuffle_{j}_embeddings.npy"), embeddings)
# def process_table_wrapper(table_index, truncated_table, args, model_name, model, tokenizer, device, max_length, padding_token):
#     save_directory = os.path.join(args.save_directory, model_name)
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)

#     all_shuffled_embeddings = []

#     for j in range(args.num_shuffles + 1):
#         if j == 0:
#             processed_table = truncated_table
#         else:
#             processed_table = row_shuffle(truncated_table)

#         col_list = table2colList(processed_table)
#         processed_tokens = process_table(tokenizer, col_list, max_length, model_name)
#         input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
#         attention_mask = [1 if token != padding_token else 0 for token in processed_tokens[0]]
#         cls_positions = processed_tokens[1]

#         input_ids_tensor = torch.tensor([input_ids], device=device)
#         attention_mask_tensor = torch.tensor([attention_mask], device=device)

#         if model_name.startswith("t5"):
#             outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, decoder_input_ids=input_ids_tensor)
#         else:
#             outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
#         last_hidden_state = outputs.last_hidden_state

#         embeddings = []
#         for position in cls_positions:
#             cls_embedding = last_hidden_state[0, position, :].detach().cpu()
#             embeddings.append(cls_embedding)

#         all_shuffled_embeddings.append(embeddings)

#     avg_cosine_similarities = []
#     mcvs = []

#     for i in range(len(all_shuffled_embeddings[0])):
#         column_cosine_similarities = []
#         column_embeddings = []

#         for j in range(args.num_shuffles + 1):
#             column_embeddings.append(all_shuffled_embeddings[j][i])

#         for j in range(1, args.num_shuffles + 1):
#             truncated_embedding = all_shuffled_embeddings[0][i]
#             shuffled_embedding = all_shuffled_embeddings[j][i]

#             cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (norm(truncated_embedding) * norm(shuffled_embedding))
#             column_cosine_similarities.append(cosine_similarity.item())

#         avg_cosine_similarity = torch.mean(torch.tensor(column_cosine_similarities))
#         mcv = compute_mcv(torch.stack(column_embeddings))

#         avg_cosine_similarities.append(avg_cosine_similarity.item())
#         mcvs.append(mcv)

#     table_avg_cosine_similarity = torch.mean(torch.tensor(avg_cosine_similarities))
#     table_avg_mcv = torch.mean(torch.tensor(mcvs))

#     # print(f"Table {table_index}: Average Cosine Similarity = {table_avg_cosine_similarity.item()}, Average MCV = {table_avg_mcv.item()}")

#     results = {
#         "avg_cosine_similarities": avg_cosine_similarities,
#         "mcvs": mcvs,
#         "table_avg_cosine_similarity": table_avg_cosine_similarity.item(),
#         "table_avg_mcv": table_avg_mcv.item()
#     }
#     print(f"Table {table_index}:")
#     print("Average Cosine Similarities:", results["avg_cosine_similarities"])
#     print("MCVs:", results["mcvs"])
#     print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
#     print("Table Average MCV:", results["table_avg_mcv"])

#     torch.save(results, os.path.join(save_directory, f"table_{table_index}_results.pt"))


# def process_table_wrapper(table_index, truncated_table, args, model_name, model, tokenizer, device, max_length, padding_token):

#     save_directory = os.path.join(args.save_directory, model_name)
#     if not os.path.exists(save_directory):
#         os.makedirs(save_directory)

#     all_shuffled_embeddings = []

#     for j in range(args.num_shuffles + 1):
#         if j == 0:
#             processed_table = truncated_table
#         else:
#             processed_table = row_shuffle(truncated_table)

#         col_list = table2colList(processed_table)
#         processed_tokens = process_table(tokenizer, col_list, max_length, model_name)
#         input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
#         attention_mask = [1 if token != padding_token else 0 for token in processed_tokens[0]]
#         cls_positions = processed_tokens[1]

#         input_ids_tensor = torch.tensor([input_ids], device=device)
#         attention_mask_tensor = torch.tensor([attention_mask], device=device)

#         if model_name.startswith("t5"):
#             outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, decoder_input_ids=input_ids_tensor)
#         else:
#             outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
#         last_hidden_state = outputs.last_hidden_state

#         embeddings = []
#         for position in cls_positions:
#             cls_embedding = last_hidden_state[0, position, :].detach().cpu().numpy()
#             embeddings.append(cls_embedding)

#         all_shuffled_embeddings.append(embeddings)

#     np.save(os.path.join(save_directory, f"table_{table_index}.npy"), all_shuffled_embeddings)

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
#     parser.add_argument('-n', '--num_shuffles', type=int, required=True, help='Number of times to shuffle and save embeddings')
#     parser.add_argument('-w', '--num_workers', type=int, default=4, help="Number of worker threads for parallel processing. For example: -n 4")
#     parser.add_argument('-m', '--model_name', type=str, default="", help='Name of the Hugging Face model to use')
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
#     parser.add_argument('-n', '--num_shuffles', type=int, required=True, help='Number of times to shuffle and save embeddings')
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

#         for j in range(args.num_shuffles):
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
