def table2str_using_columns(table):
    cols = table2colList(table)
    return " ".join(cols)


def table_based_is_fit(sample_table, tokenizer, max_length, model_name):
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


def table_based_truncate(table, tokenizer, max_length, model_name):
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


def table2colList(table):
    cols = []

    for column in table.columns:
        # Convert column values to strings and join them with spaces
        string_values = " ".join(table[column].astype(str).tolist())
        col_str = f"{column} {string_values}"
        cols.append(col_str)

    return cols
