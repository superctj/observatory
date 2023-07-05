#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import itertools
import random

import pandas as pd
import torch

import numpy as np
import multiprocessing
from multiprocessing import Pool
from huggingface_models import  load_transformers_model, load_transformers_tokenizer_and_max_length
from truncate import truncate_index
from torch.linalg import inv, norm

from mcv import compute_mcv
from scipy.special import comb


from torch.linalg import inv, norm

from torch.serialization import save

from scipy.spatial.distance import cosine

from doduo import Doduo


def get_subsets(n, m, portion):

    portion_size = int(n * portion)

    max_possible_tables = comb(n, portion_size)

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

        dfs.append(df.iloc[subset].copy())
    return dfs


def generate_row_sample_embeddings(model, device, table, num_shuffles, portion):
    all_shuffled_embeddings = []
    tables = shuffle_df(table, num_shuffles, portion)
    for processed_table in tables:

        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)
        annot_df = model.annotate_columns(processed_table)
        embeddings = annot_df.colemb
        embeddings = [torch.tensor(embeddings[j])
                      for j in range(len(embeddings))]

        all_shuffled_embeddings.append(embeddings)

    return all_shuffled_embeddings

# from multiprocessing import Pool, cpu_count

# def process_table(args):
#     processed_table, model = args
#     processed_table = processed_table.reset_index(drop=True)
#     processed_table = processed_table.astype(str)
#     annot_df = model.annotate_columns(processed_table)
#     embeddings = annot_df.colemb
#     embeddings = [torch.tensor(embeddings[j]) for j in range(len(embeddings))]
#     return embeddings

# def generate_row_sample_embeddings(model, device, table, num_shuffles, portion):
#     all_shuffled_embeddings = []
#     tables = shuffle_df(table, num_shuffles, portion)
#     with Pool(cpu_count()) as p:
#         all_shuffled_embeddings = p.map(process_table, [(table, model) for table in tables])
#     return all_shuffled_embeddings


def analyze_embeddings(all_shuffled_embeddings):

    avg_cosine_similarities = []

    mcvs = []
    
    if len(all_shuffled_embeddings)<24:
        return [], [], None, None

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


def process_table_wrapper(table_index, table, args, model_name, model, device):

    save_directory_results = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(

        args.sample_portion), args.save_directory, model_name, 'results')

    save_directory_embeddings = os.path.join('/nfs/turbo/coe-jag/zjsun', 'sample_portion', str(

        args.sample_portion), args.save_directory, model_name, 'embeddings')

    # Create the directories if they don't exist

    if not os.path.exists(save_directory_embeddings):

        os.makedirs(save_directory_embeddings)

    if not os.path.exists(save_directory_results):

        os.makedirs(save_directory_results)

    all_shuffled_embeddings = generate_row_sample_embeddings(

        model, device,  table, args.num_samples, args.sample_portion)

    save_file_path = os.path.join(save_directory_embeddings, f"table_{table_index}_embeddings.pt")
    # If the file exists, load it and substitute the elements.
    if os.path.exists(save_file_path):
        existing_embeddings = torch.load(save_file_path)

        # Ensure that existing_embeddings is long enough
        if len(existing_embeddings) < len(all_shuffled_embeddings):
            existing_embeddings = all_shuffled_embeddings
        else:
            # Substitute the elements
            existing_embeddings[:len(all_shuffled_embeddings)] = all_shuffled_embeddings

        # Save the modified embeddings
        torch.save(existing_embeddings, save_file_path)
    else:
        # If the file doesn't exist, just save all_shuffled_embeddings
        torch.save(all_shuffled_embeddings, save_file_path)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = argparse.Namespace

    model_args.model = "wikitable"  # two models available "wikitable" and "viznet"

    model = Doduo(model_args, basedir=".")

    for table_index, table in enumerate(tables):
        if table_index < args.table_num:
            continue
        try:
            process_table_wrapper(table_index, table, args,
                                model_name, model,  device)
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

    parser.add_argument('-n', '--num_samples', type=int, required=True,

                        help='Number of times to shuffle and save embeddings')

    parser.add_argument('-m', '--model_name', type=str,

                        default="", help='Name of the Hugging Face model to use')

    parser.add_argument('-p', '--sample_portion', type=float,

                        default=0.25, help='Portion of sample to use')
    parser.add_argument('-t', '--table_num', type=int,

                        default=0, help='num of start table')
    args = parser.parse_args()

    table_files = [f for f in os.listdir(

        args.read_directory) if f.endswith('.csv')]

    normal_tables = []

    for file in table_files:

        table = pd.read_csv(

            f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)
    model_name = 'bert-base-uncased'
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    truncated_tables =[]
    for table_index, table in enumerate(normal_tables):
        max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)
    model_name = args.model_name
    print()

    print("Evaluate  for: ", model_name)
    print()

    process_and_save_embeddings(model_name, args, truncated_tables)


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
