import pandas as pd
def table2colList(
    # table):
    table: pd.DataFrame
) -> list[list[str]]:
    """Convert a table to a list of columns, where each column is a list of cells.
    
    Args:
        table: A pandas DataFrame representing a table.
    
    Returns:
        A list of columns, where each column is a list of cells.
    """
    cols = []

    for i in range(len(table.columns)):
        col_cells = [table.columns[i]] + table.iloc[:, i].astype(str).tolist()
        cols.append(col_cells)

    return cols

def is_fit(
    # cols, tokenizer, max_length, model_name):
    cols: list[list[str]], 
    tokenizer, 
    max_length: int,
    model_name: str
) -> bool:
    """Check if the table fits within the maximum token length.
    
    Args:
        cols: A list of columns, where each column is a list of cells.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
    
    Returns:
        A boolean indicating if the table fits within the maximum token length.
    """
    current_tokens = []
    # if model_name.startswith("google/tapas"):
    #     current_tokens = ["[CLS]"]  + ["[SEP]"]

    for col in cols:
        if model_name.startswith("t5"):
            current_tokens += ["<s>"]
        else:
            current_tokens += ["[CLS]"]

        for cell in col:
            cell_tokens = tokenizer.tokenize(cell)
            for token in cell_tokens:
                current_tokens += [token]
            if len(current_tokens) > max_length:
                return False

    if model_name.startswith("t5"):
        current_tokens += ["</s>"]
    else:
        current_tokens += ["[SEP]"]

    if len(current_tokens) > max_length:
        return False

    return True

def max_rows(
    table: pd.DataFrame,
    tokenizer,
    max_length: int,
    model_name: str
) -> int:
    """Find the maximum number of rows that fit within the maximum token length.
    
    Args:
        table: A pandas DataFrame representing a table.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of the tokens.
        model_name: The name of the model.
    
    Returns:
        low: The maximum number of rows that fit within the maximum token length.
    """
    low = 0
    high = len(table)

    while low < high:
        mid = (low + high + 1) // 2
        sample_table = table.iloc[:mid, :]
        cols = table2colList(sample_table)
        if is_fit(cols, tokenizer, max_length, model_name):
            low = mid
        else:
            high = mid - 1

    return low

def cellbased_truncate(
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
        The maximum number of rows that fit within the maximum token length.
    """
    table.columns = table.columns.astype(str)
    table = table.reset_index(drop=True)
    table = table.astype(str)
    return max_rows(table, tokenizer, max_length, model_name)
