import pandas as pd
def table2str_using_columns(
    table: pd.DataFrame
) -> str:
    """Convert a table to a string using columns.
    
    Args:
        table: A pandas DataFrame representing a table.
    
    Returns:
        A string representation of the table.
    """
    cols = table2colList(table)
    return " ".join(cols)


def table_based_is_fit(
    sample_table: pd.DataFrame,
    tokenizer,
    max_length: int,
    model_name: str
) -> bool:
    """Check if the table fits within the maximum token length.
    
    Args:
        sample_table: A pandas DataFrame representing a sample table.
        tokenizer: The tokenizer to use.    
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
    
    Returns:
        A boolean indicating if the table fits within the maximum token length.
    """
    if model_name.startswith("microsoft/tapex"):
        encoding = tokenizer(sample_table, return_tensors="pt")
        input_ids = encoding["input_ids"][0].tolist()

        return len(input_ids) <= max_length
    else:
        table_str = table2str_using_columns(sample_table)
        tokens = tokenizer.tokenize(table_str)

        if model_name.startswith("t5"):
            tokens = ["<s>"] + tokens + ["</s>"]
        else:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        return len(tokens) <= max_length


def table_based_truncate(
    table: pd.DataFrame,
    tokenizer,
    max_length: int,
    model_name: str
) -> int:
    """Truncate a table based on the maximum token length and column-based linearalization. 
    
    Args:
        table: A pandas DataFrame representing a table.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
    
    Returns:
        The number of rows to keep in the table.
    """
    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)
    low = 0
    high = len(table)

    while low < high:
        mid = (low + high + 1) // 2
        sample_table = table[:mid]

        if table_based_is_fit(sample_table, tokenizer, max_length, model_name):
            low = mid
        else:
            high = mid - 1

    return low


def table2colList(
    table: pd.DataFrame
) -> list[str]:
    """Convert a table to a list of columns, where each column is a string.
    
    Args:
        table: A pandas DataFrame representing a table.
    
    Returns:
        A list of columns, where each column is a string.
    """
    cols = []

    for i in range(len(table.columns)):
        string_values = " ".join(table.iloc[:, i].astype(str).tolist())
        col_str = f"{table.columns[i]} {string_values}"
        cols.append(col_str)

    return cols
