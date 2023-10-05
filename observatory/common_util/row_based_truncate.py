def table2rowList(table):
    rows = []
    for index, row in table.iterrows():
        row_str = " ".join([f"{col} {cell}" for col, cell in zip(table.columns, row.astype(str).tolist())])
        rows.append(row_str)
    return rows

def is_fit(table, tokenizer, max_length, model_name):
    
    if model_name.startswith("microsoft/tapex"):
        result = [tokenizer.cls_token_id]
        
        # Tokenize each row and append to result
        for _, row in table.iterrows():
            one_row_table = table.DataFrame([row])
            encoding = tokenizer(one_row_table, return_tensors="pt")
            row_ids = encoding['input_ids'][0].tolist()[1:-1]  # Remove cls and sep tokens
            result.extend(row_ids)
            result.append(tokenizer.cls_token_id)
            if len(result) > max_length:
                return False
            
    else: 
        current_tokens = []
        
        rows = table2rowList(table)

        for row in rows:
            # Tokenize row without special tokens
            row_tokens = tokenizer.tokenize(row)
            # Check model name and use appropriate special tokens
            if model_name.startswith("t5"):
                # For T5, add <s> at the start and </s> at the end
                row_tokens = ["<s>"] + row_tokens + ["</s>"]
            else:
                # For other models (BERT, RoBERTa, TAPAS), add [CLS] at the start and [SEP] at the end
                row_tokens = ["[CLS]"] + row_tokens + ["[SEP]"]
            # Check if adding the new tokens would exceed the max length
            if len(current_tokens) + len(row_tokens) > max_length:
                # If so, stop and return false
                return False
            else:
                # If not, remove the last token (</s> or [SEP]) from the current tokens
                if current_tokens:
                    current_tokens = current_tokens[:-1]
                # Then concatenate the new tokens
                current_tokens += row_tokens
    return True

def row_based_truncate(table, tokenizer, max_length, model_name):
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
