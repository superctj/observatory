import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import argparse
import pandas as pd
import torch

from observatory.common_util.column_based_truncate import column_based_truncate


def table2colList(table):
    cols = []
    for i in range(len(table.columns)):
        string_values = " ".join(table.iloc[:, i].astype(str).tolist())
        col_str = f"{table.columns[i]} {string_values}"
        cols.append(col_str)
    return cols


def column_based_process_table(tokenizer, cols, max_length, model_name):
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


def get_tapas_column_embeddings(inputs, last_hidden_states):
    # find the maximum column id
    max_column_id = inputs["token_type_ids"][0][:, 1].max()

    column_embeddings = []

    # loop over all column ids
    # try:
    for column_id in range(1, max_column_id + 1):
        # find all indices where the token_type_ids is equal to the column id
        indices = torch.where(inputs["token_type_ids"][0][:, 1] == column_id)[0]

        # get the embeddings at these indices
        embeddings = last_hidden_states[0][indices]

        # compute the average embedding
        column_embedding = embeddings.mean(dim=0)

        column_embeddings.append(column_embedding)
    # except Exception as e:
    #         print("Error message:", e)
    #         print(max_column_id)
    #         assert False, "Stop as expected"

    return column_embeddings


def get_hugging_face_column_embeddings(tables, model_name, tokenizer, max_length, model):
    # tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print(device)
    # model = load_transformers_model(model_name, device)
    # model.eval()
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
    truncated_tables = []
    for table_index, table in enumerate(tables):
        max_rows_fit = column_based_truncate(table, tokenizer, max_length, model_name)
        if max_rows_fit < 1:
            ##################
            ## for other properties, do something here
            ################
            continue
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)
    all_embeddings = []
    if model_name.startswith("google/tapas"):

        for processed_table in truncated_tables:
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
            embeddings = get_tapas_column_embeddings(inputs, last_hidden_states)
            all_embeddings.append(embeddings)

            # Clear memory
            del inputs
            del outputs
            del last_hidden_states
            del embeddings
            torch.cuda.empty_cache()
    else:

        for processed_table in truncated_tables:

            col_list = table2colList(processed_table)
            processed_tokens = column_based_process_table(
                tokenizer, col_list, max_length, model.name_or_path
            )
            input_ids = tokenizer.convert_tokens_to_ids(processed_tokens[0])
            attention_mask = [
                1 if token != padding_token else 0 for token in processed_tokens[0]
            ]
            cls_positions = processed_tokens[1]

            input_ids_tensor = torch.tensor([input_ids], device=device)
            attention_mask_tensor = torch.tensor([attention_mask], device=device)
            with torch.no_grad():
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


def get_hugging_face_column_embeddings_batched(tables, model_name, tokenizer, max_length, model, batch_size=32):
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    truncated_tables = []
    for table_index, table in enumerate(tables):
        max_rows_fit = column_based_truncate(table, tokenizer, max_length, model_name)
        if max_rows_fit < 1:
            ##################
            ## for other properties, do something here
            ################
            continue
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)
    num_all_tables = len(truncated_tables)
    all_embeddings = []

    batch_input_ids = []
    batch_attention_masks = []
    batch_token_type_ids = []
    batch_cls_positions = []

    for table_num, processed_table in enumerate(truncated_tables):
        if model_name.startswith("google/tapas"):
            processed_table.columns = processed_table.columns.astype(str)
            processed_table = processed_table.reset_index(drop=True)
            processed_table = processed_table.astype(str)

            inputs = tokenizer(
                table=processed_table,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )

            batch_input_ids.append(inputs["input_ids"][0])
            batch_token_type_ids.append(inputs["token_type_ids"][0])
            batch_attention_masks.append(inputs["attention_mask"][0])

            # If batch size is reached or it's the last table, then process the batch.
            if len(batch_input_ids) == batch_size or num_all_tables == table_num + 1:
                batched_inputs = {
                    'input_ids': torch.stack(batch_input_ids, dim=0).to(device),
                    'token_type_ids': torch.stack(batch_token_type_ids, dim=0).to(device),
                    'attention_mask': torch.stack(batch_attention_masks, dim=0).to(device)
                }

                with torch.no_grad():
                    outputs = model(**batched_inputs)

                last_hidden_states = outputs.last_hidden_state

                # Extracting embeddings for rows
                for batch_idx in range(last_hidden_states.shape[0]):
                    column_embeddings = []
                    for column_id in range(1, max(batched_inputs["token_type_ids"][batch_idx][:, 1]) + 1):
                        indices = torch.where(batched_inputs["token_type_ids"][batch_idx][:, 1] == column_id)[0]
                        embeddings = last_hidden_states[batch_idx][indices]
                        column_embedding = embeddings.mean(dim=0)
                        column_embeddings.append(column_embedding)
                    all_embeddings.append(column_embeddings)

                # Clear the batch lists
                batch_input_ids, batch_token_type_ids, batch_attention_masks = [], [], []

        else:
            processed_tokens, cls_positions = column_based_process_table(
                tokenizer, processed_table, max_length, model.name_or_path
            )
            input_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
            attention_mask = [
                1 if token != padding_token else 0 for token in processed_tokens
            ]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))
            batch_cls_positions.append(cls_positions)

            # If batch size is reached or it's the last table, then process the batch.
            if len(batch_input_ids) == batch_size or num_all_tables == table_num + 1:
                input_ids_tensor = torch.stack(batch_input_ids, dim=0).to(device)
                attention_mask_tensor = torch.stack(batch_attention_masks, dim=0).to(device)
                
                with torch.no_grad():
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

                for i, last_hidden_state in enumerate(outputs.last_hidden_state):
                    embeddings = []
                    for position in batch_cls_positions[i]:
                        cls_embedding = last_hidden_state[position, :].detach().cpu()
                        embeddings.append(cls_embedding)
                    all_embeddings.append(embeddings)

                # Clear the batch lists
                batch_input_ids, batch_attention_masks, batch_cls_positions = [], [], []

    return all_embeddings