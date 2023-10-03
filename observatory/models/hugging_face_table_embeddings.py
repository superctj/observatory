

import pandas as pd
import torch

from observatory.common_util.table_based_truncate import table_based_truncate, table2colList

def table_based_process_table(tokenizer, cols, max_length, model_name):
    current_tokens = []

    # Check model name and use appropriate special tokens
    if model_name.startswith("t5"):
        # For T5, add <s> at the start only once for the whole table
        current_tokens = ["<s>"]
    else:
        # For other models, add [CLS] at the start only once for the whole table
        current_tokens = ["[CLS]"]

    for idx, col in enumerate(cols):
        col_tokens = tokenizer.tokenize(col)
        # Do not add [CLS] or <s> in between columns
        if model_name.startswith("t5"):
            col_tokens += ["</s>"]  # only add </s> at the end for T5
        else:
            col_tokens += ["[SEP]"]  # only add [SEP] at the end for others

        if len(current_tokens) + len(col_tokens) > max_length:
            assert False, "The length of the tokens exceeds the max length. Please run the truncate.py first."
            break
        else:
            if current_tokens and model_name.startswith("t5"):
                current_tokens = current_tokens[:-1]  # Remove previous </s> for T5
            current_tokens += col_tokens

    if len(current_tokens) < max_length:
        padding_length = max_length - len(current_tokens)
        padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
        current_tokens += [padding_token] * padding_length

    return current_tokens, [0]  # [0] since there's only one [CLS] at the start


def get_hugging_face_table_embeddings(tables, model_name, tokenizer, max_length, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    truncated_tables = []
    for table_index, table in enumerate(tables):
        max_rows_fit = table_based_truncate(table, tokenizer, max_length, model_name)
        if max_rows_fit < 1:
            ##################
            ## for other properties, do something here
            ################
            continue
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)


    all_embeddings = []

    for processed_table in truncated_tables:
        if model_name.startswith("google/tapas"):
            processed_table.columns = processed_table.columns.astype(str)
            processed_table = processed_table.reset_index(drop=True)
            processed_table = processed_table.astype(str)
            # TAPAS-specific logic for table
            inputs = tokenizer(
                table=processed_table,
                padding="max_length",
                return_tensors="pt",
                truncation=True,
            )
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state

            # Get the indices where column_ids > 0 (excluding the headers)
            indices = torch.where(inputs["token_type_ids"][0][:, 1] > 0)[0]
            embeddings = last_hidden_states[0][indices]
            table_embedding = embeddings.mean(dim=0)
            all_embeddings.append(table_embedding)
        
        else:
            # Logic for models other than TAPAS
            # Convert your processed_table to a single string representing the entire table and tokenize it
            cols = table2colList(processed_table)
            processed_tokens, cls_position = table_based_process_table(
                tokenizer, cols, max_length, model.name_or_path
            )
            input_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
            attention_mask = [1 if token != padding_token else 0 for token in processed_tokens]

            # Convert to tensors and move to device
            input_ids_tensor = torch.tensor([input_ids], device=device)
            attention_mask_tensor = torch.tensor([attention_mask], device=device)

            # Get model outputs
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
            
            table_embedding = last_hidden_state[0, cls_position, :].detach().cpu()
            all_embeddings.append(table_embedding)

    return all_embeddings


def get_hugging_face_table_embeddings_batched(tables, model_name, tokenizer, max_length, model, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    truncated_tables = []
    for table_index, table in enumerate(tables):
        max_rows_fit = table_based_truncate(table, tokenizer, max_length, model_name)
        if max_rows_fit < 1:
            ##################
            ## for other properties, do something here
            ################
            continue
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)

    all_embeddings = []

    batch_input_ids = []
    batch_token_type_ids = []
    batch_attention_masks = []
    batch_cls_positions = []

    for processed_table in truncated_tables:
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
            if len(batch_input_ids) == batch_size or processed_table is truncated_tables[-1]:
                batched_inputs = {
                    "input_ids": torch.stack(batch_input_ids, dim=0).to(device),
                    "token_type_ids": torch.stack(batch_token_type_ids, dim=0).to(device),
                    "attention_mask": torch.stack(batch_attention_masks, dim=0).to(device)
                }

                with torch.no_grad():
                    outputs = model(**batched_inputs)

                for last_hidden_state in outputs.last_hidden_state:
                    indices = torch.where(batched_inputs["token_type_ids"][:, 1] > 0)[0]
                    embeddings = last_hidden_state[indices]
                    table_embedding = embeddings.mean(dim=0)
                    all_embeddings.append(table_embedding)

                # Clear the batch lists
                batch_input_ids, batch_token_type_ids, batch_attention_masks = [], [], []

        else:
            # Logic for models other than TAPAS
            cols = table2colList(processed_table)
            processed_tokens, cls_position = table_based_process_table(
                tokenizer, cols, max_length, model.name_or_path
            )
            input_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
            attention_mask = [1 if token != padding_token else 0 for token in processed_tokens]
            
            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_cls_positions.append(cls_position)
            
            if len(batch_input_ids) == batch_size or processed_table is truncated_tables[-1]:
                input_ids_tensor = torch.tensor(batch_input_ids, device=device)
                attention_mask_tensor = torch.tensor(batch_attention_masks, device=device)

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
                    cls_position = batch_cls_positions[i]
                    table_embedding = last_hidden_state[0, cls_position, :].detach().cpu()
                    all_embeddings.append(table_embedding)

                # Clear the batch lists
                batch_input_ids, batch_attention_masks, batch_cls_positions = [], [], []

    return all_embeddings
