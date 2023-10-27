import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import argparse
import itertools
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import torch
import numpy as np
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.mcv import compute_mcv
from torch.linalg import inv, norm

# align with the processing way in taptap_dataset.py
def table_to_row_strings(df):
    numerical_features = df.select_dtypes(include=np.number).columns.to_list()
    row_strings = []

    for _, row in df.iterrows():
        formatted_values = []
        
        for i in range(len(df.columns)):
            feature = df.columns[i]
            value = str(row[i]).strip()
            
            if feature not in numerical_features or value == 'None':
                formatted_value = "%s is %s" % (feature, value)
            else:
                if '.' not in value:
                    formatted_value = "%s is %s" % (feature, value)
                else:
                    v = float(value)
                    i = 0
                    if abs(v) < 1e-10:
                        v = 0
                    else:
                        while abs(v * (10 ** i)) < 1:
                            i += 1
                        v = round(v, max(3, i + 2))
                    formatted_value = "%s is %s" % (feature, str(v))
            
            formatted_values.append(formatted_value)
        
        row_string = ", ".join(formatted_values)
        row_strings.append(row_string)
    
    return row_strings


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



def analyze_embeddings(all_embeddings):
    avg_cosine_similarities = []
    mcvs = []

    for i in range(min([len(embeddings) for embeddings in all_embeddings])):
        cosine_similarities = []
        row_embeddings = []

        for j in range(len(all_embeddings)):
            row_embeddings.append(all_embeddings[j][i])

        for j in range(1, len(all_embeddings)):
            truncated_embedding = all_embeddings[0][i]
            shuffled_embedding = all_embeddings[j][i]

            cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (
                norm(truncated_embedding) * norm(shuffled_embedding)
            )
            cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = torch.mean(torch.tensor(cosine_similarities))
        mcv = compute_mcv(torch.stack(row_embeddings))

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
    


class TableEmbedder:
    def __init__(self, model, tokenizer, device):
        self.model = model.to(device).eval()  # Ensure the model is in eval mode
        self.tokenizer = tokenizer
        self.device = device

    def get_last_hidden_state(self, input_ids, attention_mask=None):
        with torch.no_grad():  # Disable gradient computation
            outputs = self.model.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def compute_embeddings(self, tables, batch_size):
        all_row_strings = []
        table_row_indices = []

        for table_idx, table in enumerate(tables):
            row_strings = table_to_row_strings(table)
            all_row_strings.extend(row_strings)
            table_row_indices.extend([(table_idx, row_idx) for row_idx in range(len(row_strings))])

        all_embeddings = [[None] * len(table_to_row_strings(table)) for table in tables]

        for i in range(0, len(all_row_strings), batch_size):
            batch_row_strings = all_row_strings[i:i + batch_size]

            # Batch tokenization
            tokenized_texts = self.tokenizer(batch_row_strings, padding='longest', return_tensors='pt', truncation=True)
            
            input_ids = tokenized_texts['input_ids'].to(self.device)
            attention_masks = tokenized_texts['attention_mask'].to(self.device)

            hidden_states = self.get_last_hidden_state(input_ids, attention_mask=attention_masks)

            for idx, mask in enumerate(attention_masks):
                avg_embedding = hidden_states[idx][mask.bool()].mean(dim=0)
                table_idx, row_idx = table_row_indices[i + idx]
                all_embeddings[table_idx][row_idx] = avg_embedding

            # Release unnecessary tensors and clear GPU cache
            del input_ids, attention_masks, hidden_states
            torch.cuda.empty_cache()

        return all_embeddings






def process_table_wrapper(
    table_index,
    truncated_table,
    args,
    model_name,
    model,
    tokenizer,
    device,
    max_length,
):
    save_directory_results = os.path.join(
        args.save_directory,
        "Row_embedding_Column_Order_Insignificance",
        model_name,
        "results",
    )
    save_directory_embeddings = os.path.join(
        args.save_directory,
        "Row_embedding_Column_Order_Insignificance",
        model_name,
        "embeddings",
    )
    # save_directory_results  = os.path.join( args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join( args.save_directory, model_name ,'embeddings')
    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
        
    tables, perms = shuffle_df_columns(truncated_table, args.num_shuffles)
    embedder = TableEmbedder(model, tokenizer, device)
    all_embeddings = embedder.compute_embeddings(tables, args.batch_size)

    if len(all_embeddings)<24:
        print("len(all_embeddings)<24")
        return
    torch.save(
        all_embeddings,
        os.path.join(save_directory_embeddings, f"table_{table_index}_embeddings.pt"),
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
    print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
    print("Table Average MCV:", results["table_avg_mcv"])
    torch.save(
        results, os.path.join(save_directory_results, f"table_{table_index}_results.pt")
    )


def process_and_save_embeddings(model_name, args, tables):
    model_name = "ztphs980/taptap-distill"
    tokenizer = AutoTokenizer.from_pretrained("ztphs980/taptap-distill")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = AutoModelForSequenceClassification.from_pretrained("ztphs980/taptap-distill")
    model.eval()
    # padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    for table_index, table in enumerate(tables):
        if table_index < args.start_index:
            continue
        if table_index >= args.start_index + args.num_tables:
            break
        process_table_wrapper(
            table_index,
            table,
            args,
            model_name,
            model,
            tokenizer,
            device,
            tokenizer.model_max_length,
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
        "--num_shuffles",
        type=int,
        required=True,
        help="Number of times to shuffle and save embeddings",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="ztphs980/taptap-distill",
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
    parser.add_argument(
        "--num_tables",
        type=int,
        default=1000,
        help="Number of tables to process",
    )
    args = parser.parse_args()

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith(".csv")]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
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
    print()
    print("Evaluate row shuffle for: ", model_names)
    print()

    for model_name in model_names:
        process_and_save_embeddings(model_name, args, normal_tables)
