import pandas as pd


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


def is_fit(
    table: pd.DataFrame,
    tokenizer,
    max_length: int,
    model_name: str
) -> bool:
    """Check if the table fits within the maximum token length.
    
    Args:
        table: A pandas DataFrame representing a table.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
        
    Returns:
        A boolean indicating if the table fits within the maximum token length.
    """
    if model_name.startswith("microsoft/tapex"):
        result = [tokenizer.cls_token_id]

        # Tokenize each row and append to result
        for _, row in table.iterrows():
            one_row_table = pd.DataFrame([row])
            encoding = tokenizer(one_row_table, return_tensors="pt")
            row_ids = encoding["input_ids"][0].tolist()[
                1:-1
            ]  # Remove cls and sep tokens
            result.extend(row_ids)
            result.append(tokenizer.cls_token_id)
            if len(result) > max_length:
                return False

    else:
        current_tokens = []

        rows = row2strList(table)

        for row in rows:
            # Tokenize row without special tokens
            row_tokens = tokenizer.tokenize(row)
            # Check model name and use appropriate special tokens
            if model_name.startswith("t5"):
                # For T5, add <s> at the start and </s> at the end
                row_tokens = ["<s>"] + row_tokens + ["</s>"]
            else:
                # For other models (BERT, RoBERTa, TAPAS), add [CLS] at the
                #  start and [SEP] at the end
                row_tokens = ["[CLS]"] + row_tokens + ["[SEP]"]

            # Check if adding the new tokens would exceed the max length
            if len(current_tokens) + len(row_tokens) > max_length:
                # If so, stop and return false
                return False
            else:
                # If not, remove the last token (</s> or [SEP]) from the
                # current tokens
                if current_tokens:
                    current_tokens = current_tokens[:-1]
                # Then concatenate the new tokens
                current_tokens += row_tokens

    return True


def row_based_truncate(
    # table, tokenizer, max_length, model_name):
    table: pd.DataFrame,
    tokenizer,
    max_length: int,
    model_name: str
) -> int:
    """Truncate a table based on the maximum token length and row-based linearalization.
    
    Args:
        table: A pandas DataFrame representing a table.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
    
    Returns:
        low: The maximum number of rows that fit within the maximum token length.
    """
    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)
    low = 0
    high = len(table)

    while low < high:
        mid = (low + high + 1) // 2
        sample_table = table[:mid]
        if is_fit(sample_table, tokenizer, max_length, model_name):
            low = mid
        else:
            high = mid - 1

    return low
