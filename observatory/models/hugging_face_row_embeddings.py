import os

import pandas as pd
import torch

from observatory.common_util.row_based_truncate import row_based_truncate
from observatory.models.tapex import tapex_inference

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def row2strList(
    table: pd.DataFrame
) -> list[str]:
    """Convert a table to a list of rows, where each row is a string.
    
    Args:
        table: A pandas DataFrame representing a table.
        
    Returns:
        A list of rows, where each row is a string.
    """
    rows = []

    for index, row in table.iterrows():
        row_str = " ".join(
            [f"{col} {str(val)}" for col, val in zip(table.columns, row)]
        )
        rows.append(row_str)

    return rows


def row_based_process_table(
    tokenizer, table: pd.DataFrame, max_length: int, model_name: str
) -> tuple[list[str], list[int]]:
    """Process a table by tokenizing its rows and adding special tokens.
    
    Args:
        tokenizer: The tokenizer to use.
        table: A pandas DataFrame representing a table.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
        
    Returns:
        current_tokens: A list of tokens representing the table.
        cls_positions: A list of positions of the [CLS] tokens.
    """
    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)

    if model_name.startswith("microsoft/tapex"):
        # Initialize result
        result = [tokenizer.cls_token_id]
        cls_positions = [0]  # The first token is always cls_token_id

        # Tokenize each row and append to result
        for _, row in table.iterrows():
            one_row_table = pd.DataFrame([row])
            encoding = tokenizer(one_row_table, return_tensors="pt")
            row_ids = encoding["input_ids"][0].tolist()[
                1:-1
            ]  # Remove cls and sep tokens
            result.extend(row_ids)
            result.append(tokenizer.cls_token_id)
            cls_positions.append(len(result) - 1)

        # Remove the last added cls_token_id
        result = result[:-1]
        cls_positions = cls_positions[:-1]

        # Append sep_token_id to result
        result.append(tokenizer.sep_token_id)

        # Pad to maxlength
        result = result + [tokenizer.pad_token_id] * (max_length - len(result))

        return result, cls_positions
    else:
        rows = row2strList(table)
        current_tokens = []
        cls_positions = []

        for idx, row in enumerate(rows):
            row_tokens = tokenizer.tokenize(row)

            # Check model name and use appropriate special tokens
            if model_name.startswith("t5"):
                # For T5, add <s> at the start and </s> at the end
                row_tokens = ["<s>"] + row_tokens + ["</s>"]
            else:
                # For other models, add [CLS] at the start and [SEP] at the end
                row_tokens = ["[CLS]"] + row_tokens + ["[SEP]"]

            if len(current_tokens) + len(row_tokens) > max_length:
                assert False, (
                    "The length of the tokens exceeds the max length. Please "
                    "run the truncate.py first."
                )
            else:
                if current_tokens:
                    current_tokens = current_tokens[
                        :-1
                    ]  # Remove previous [SEP]

                current_tokens += row_tokens
                cls_positions.append(
                    len(current_tokens) - len(row_tokens)
                )  # Store the position of [CLS]

        if len(current_tokens) < max_length:
            padding_length = max_length - len(current_tokens)
            padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
            current_tokens += [padding_token] * padding_length

        return current_tokens, cls_positions


def get_hugging_face_row_embeddings_batched(
    tables: list[pd.DataFrame], 
    model_name: str,
    tokenizer,
    max_length: int,
    model,
    batch_size: int = 32
) -> list[list[torch.Tensor]]:
    """Get row embeddings for a list of tables using a model.
    
    Args:
        tables: A list of pandas DataFrames representing tables.
        model_name: The name of the model.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model: The model to use.
        batch_size: The batch size to use.
        
    Returns:
        A list of lists of row embeddings for each table.
    """
    model = model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"

    truncated_tables = []

    for table_index, table in enumerate(tables):
        max_rows_fit = row_based_truncate(
            table, tokenizer, max_length, model_name
        )
        if max_rows_fit < 1:
            # for other properties, do something here
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

            # If batch size is reached or it's the last table, then process
            # the batch.
            if (
                len(batch_input_ids) == batch_size
                or num_all_tables == table_num + 1
            ):
                batched_inputs = {
                    "input_ids": torch.stack(batch_input_ids, dim=0).to(device),
                    "token_type_ids": torch.stack(
                        batch_token_type_ids, dim=0
                    ).to(device),
                    "attention_mask": torch.stack(
                        batch_attention_masks, dim=0
                    ).to(device),
                }

                with torch.no_grad():
                    outputs = model(**batched_inputs)

                last_hidden_states = outputs.last_hidden_state

                # Extracting embeddings for rows
                for batch_idx in range(last_hidden_states.shape[0]):
                    row_embeddings = []
                    for row_id in range(
                        1,
                        max(batched_inputs["token_type_ids"][batch_idx][:, 2])
                        + 1,
                    ):
                        indices = torch.where(
                            batched_inputs["token_type_ids"][batch_idx][:, 2]
                            == row_id
                        )[0]
                        embeddings = last_hidden_states[batch_idx][indices]
                        row_embedding = embeddings.mean(dim=0)
                        row_embeddings.append(row_embedding)
                    all_embeddings.append(row_embeddings)

                # Clear the batch lists
                batch_input_ids, batch_token_type_ids, batch_attention_masks = (
                    [],
                    [],
                    [],
                )
        else:
            if model_name.startswith("microsoft/tapex"):
                input_ids, cls_positions = row_based_process_table(
                    tokenizer, processed_table, max_length, model.name_or_path
                )
                attention_mask = [
                    1 if id != tokenizer.pad_token_id else 0 for id in input_ids
                ]
            else:
                processed_tokens, cls_positions = row_based_process_table(
                    tokenizer, processed_table, max_length, model.name_or_path
                )
                input_ids = tokenizer.convert_tokens_to_ids(processed_tokens)
                attention_mask = [
                    1 if token != padding_token else 0
                    for token in processed_tokens
                ]

            batch_input_ids.append(torch.tensor(input_ids))
            batch_attention_masks.append(torch.tensor(attention_mask))
            batch_cls_positions.append(cls_positions)

            # If batch size is reached or it's the last table, then process
            # the batch.
            if (
                len(batch_input_ids) == batch_size
                or num_all_tables == table_num + 1
            ):
                input_ids_tensor = torch.stack(batch_input_ids, dim=0).to(
                    device
                )
                attention_mask_tensor = torch.stack(
                    batch_attention_masks, dim=0
                ).to(device)

                with torch.no_grad():
                    if model.name_or_path.startswith("t5"):
                        outputs = model(
                            input_ids=input_ids_tensor,
                            attention_mask=attention_mask_tensor,
                            decoder_input_ids=input_ids_tensor,
                        )
                        last_hidden_states = outputs.last_hidden_state
                    elif model_name.startswith("microsoft/tapex"):
                        last_hidden_states = tapex_inference(
                            model, input_ids_tensor, attention_mask_tensor
                        )
                    else:
                        outputs = model(
                            input_ids=input_ids_tensor,
                            attention_mask=attention_mask_tensor,
                        )
                        last_hidden_states = outputs.last_hidden_state

                for i, last_hidden_state in enumerate(last_hidden_states):
                    embeddings = []

                    for position in batch_cls_positions[i]:
                        cls_embedding = (
                            last_hidden_state[position, :].detach().cpu()
                        )
                        embeddings.append(cls_embedding)
                    all_embeddings.append(embeddings)

                # Clear the batch lists
                batch_input_ids, batch_attention_masks, batch_cls_positions = (
                    [],
                    [],
                    [],
                )

    return all_embeddings
