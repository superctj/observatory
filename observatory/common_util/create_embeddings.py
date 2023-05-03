import os
import argparse
import pandas as pd
import torch
import numpy as np
from observatory.observatory.models.huggingface_models import load_transformers_tokenizer, load_transformers_model, load_transformers_tokenizer_and_max_length

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
            cls_positions.append(len(current_tokens) - len(col_tokens))  # Store the position of [CLS]

    if len(current_tokens) < max_length:
        padding_length = max_length - len(current_tokens)
        # Use appropriate padding token based on the model
        padding_token = '<pad>' if  model_name.startswith("t5") else '[PAD]'
        current_tokens += [padding_token] * padding_length

    return current_tokens, cls_positions

def main(args):
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(args.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_transformers_model(args.model_name, device)

    table_files = [f for f in os.listdir(args.read_directory) if f.endswith('.csv')]
    normal_tables = []
    for file in table_files:
        table = pd.read_csv(f"{args.read_directory}/{file}", keep_default_na=False)
        normal_tables.append(table)

    if not os.path.exists(args.save_directory):
        os.makedirs(args.save_directory)
    padding_token = '<pad>' if  args.model_name.startswith("t5") else '[PAD]'
    for i, table in enumerate(normal_tables):
        col_list = table2colList(table)
        processed_table = process_table(tokenizer, col_list, max_length, args.model_name)
        input_ids = tokenizer.convert_tokens_to_ids(processed_table[0])
        attention_mask = [1 if token != padding_token else 0 for token in processed_table[0]]
        cls_positions = processed_table[1]

        input_ids_tensor = torch.tensor([input_ids], device=device)
        attention_mask_tensor = torch.tensor([attention_mask], device=device)

        outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor)
        last_hidden_state = outputs.last_hidden_state

        embeddings = []
        for position in cls_positions:
            cls_embedding = last_hidden_state[0, position, :].detach().cpu().numpy()
            embeddings.append(cls_embedding)

        np.save(os.path.join(args.save_directory, f"table_{i + 1}_embeddings.npy"), embeddings)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
    parser.add_argument('-r', '--read_directory', type=str, required=True, help='Directory to read tables from')
    parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the Hugging Face model to use')
    # parser.add_argument('--max_length', type=int, default=512, help='Maximum length for the processed tables')

    args = parser.parse_args()
    main(args)
