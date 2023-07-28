import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import pandas as pd
import torch
from huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from cellbased_truncate import truncate_index


def table2colList(table):
    cols = []
    for i in range(len(table.columns)):
        col_cells = [table.columns[i]] + table.iloc[:, i].astype(str).tolist()
        cols.append(col_cells)
    return cols


def process_table(tokenizer, cols, max_length, model_name):
    current_tokens = []
    token_positions = []

    for col_idx, col in enumerate(cols):
        for cell_idx, cell in enumerate(col):
            cell_tokens = tokenizer.tokenize(cell)

            if cell_idx == 0:  # Start of a new column
                if model_name.startswith("t5"):
                    current_tokens += ["<s>"]
                    token_positions.append((0, col_idx, 1))  # (row, column, flag)
                else:
                    current_tokens += ["[CLS]"]
                    token_positions.append((0, col_idx, 1))  # (row, column, flag)

            for token in cell_tokens:
                current_tokens += [token]
                token_positions.append((cell_idx, col_idx, 0))  # (row, column, flag)

    if model_name.startswith("t5"):
        current_tokens += ["</s>"]
        token_positions.append((0, 0, 2))  # (row, column, flag)
    else:
        current_tokens += ["[SEP]"]
        token_positions.append((0, 0, 2))  # (row, column, flag)

    if len(current_tokens) > max_length:
        assert (
            False
        ), "The length of the tokens exceeds the max length. Please run the truncate.py first."

    if len(current_tokens) < max_length:
        padding_length = max_length - len(current_tokens)
        padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
        current_tokens += [padding_token] * padding_length
        token_positions += [(0, 0, 3)] * padding_length  # (row, column, flag)

    return current_tokens, token_positions


def get_tapas_cell_embeddings(inputs, last_hidden_states):
    # Extract the column and row ids from the token type ids
    column_ids = inputs["token_type_ids"][0][:, 1]
    row_ids = inputs["token_type_ids"][0][:, 2]

    # Find the maximum column and row ids
    max_column_id = column_ids.max().item()
    max_row_id = row_ids.max().item()

    # Initialize a tensor to hold all cell embeddings
    cell_embeddings = torch.zeros(
        max_row_id + 1,
        max_column_id,
        last_hidden_states.shape[-1],
        device=last_hidden_states.device,
    )

    # Loop over all row and column id pairs (which correspond to cells)
    for row_id in range(max_row_id + 1):
        for column_id in range(1, max_column_id + 1):
            # Find all indices where the column and row ids match the current cell
            indices = torch.where((column_ids == column_id) & (row_ids == row_id))[0]

            # If there are no tokens for this cell, continue
            if len(indices) == 0:
                continue

            # Get the embeddings at these indices
            embeddings = last_hidden_states[0][indices]

            # Compute the average embedding and assign it to the corresponding cell
            cell_embeddings[row_id, column_id - 1] = embeddings.mean(dim=0)

    return cell_embeddings


def get_hugging_face_cell_embeddings(table, model_name):
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    model = load_transformers_model(model_name, device)
    model.eval()
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
    # truncated_tables =[]
    # for table_index, table in enumerate(tables):
    max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
    if max_rows_fit < 1:
        assert False, "Headers too long!"
    truncated_table = table.iloc[:max_rows_fit, :]
    # all_embeddings = []
    processed_table = truncated_table
    if model_name.startswith("google/tapas"):

        # for processed_table in truncated_tables:
        processed_table.columns = processed_table.columns.astype(str)
        processed_table = processed_table.reset_index(drop=True)
        processed_table = processed_table.astype(str)
        inputs = tokenizer(
            table=processed_table,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        inputs = inputs.to(device)
        try:
            with torch.no_grad():  # Turn off gradients to save memory
                outputs = model(**inputs)
        except Exception as e:
            print("Error message:", e)
            print(inputs)
            print()
            for row in inputs["token_type_ids"]:
                for cell in row:
                    print(cell)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            print()
            print(processed_table)
            assert False, "error in outputs = model(**inputs)"
        last_hidden_states = outputs.last_hidden_state
        cell_embeddings = get_tapas_cell_embeddings(inputs, last_hidden_states)
        # all_embeddings.append(cell_embeddings)

        # Clear memory
        del inputs
        del outputs
        del last_hidden_states
        torch.cuda.empty_cache()
    else:

        # for processed_table in truncated_tables:

        col_list = table2colList(processed_table)
        processed_tokens = process_table(
            tokenizer, col_list, max_length, model.name_or_path
        )
        input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
        attention_mask = [
            1 if token != padding_token else 0 for token in processed_tokens[0]
        ]
        token_positions = processed_tokens[1]

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

        # Convert the 2D array of token positions to a 3D tensor
        max_rows = len(table)
        max_cols = len(table.columns)
        cell_embeddings = torch.zeros(
            max_rows + 1, max_cols, last_hidden_state.shape[2], device=device
        )
        sum_embeddings = torch.zeros(
            last_hidden_state.shape[2], device=last_hidden_state.device
        )
        token_count = 0
        current_cell = token_positions[0][:2]

        for idx, (row, col, flag) in enumerate(token_positions):
            # Check if we've moved to a new cell
            if (row, col) != current_cell or idx == len(token_positions) - 1:
                # Compute average for the current cell and store it
                avg_embedding = (
                    sum_embeddings / token_count if token_count else sum_embeddings
                )
                cell_embeddings[
                    current_cell[0], current_cell[1]
                ] = avg_embedding.detach().cpu()

                # Reset counters for the next cell
                current_cell = (row, col)
                sum_embeddings.zero_()
                token_count = 0

            # Exclude special tokens from the cell embedding calculation
            if flag == 0:  # not a special token
                sum_embeddings += last_hidden_state[0, idx]
                token_count += 1

        # all_embeddings.append(cell_embeddings)
    return cell_embeddings
