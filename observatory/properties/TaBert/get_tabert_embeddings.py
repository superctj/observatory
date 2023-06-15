import os
import argparse
import pandas as pd
import torch
from table_bert import Table, Column
from table_bert import TableBertModel

def convert_to_table(df, tokenizer):

    header = []
    data = []

    for col in df.columns:
        try:
            # Remove commas and attempt to convert to float
            val = float(str(df[col].iloc[0]).replace(',', ''))
            # If conversion is successful, it's a real column
            col_type = 'real'
            sample_value = df[col][0]
        except (ValueError, AttributeError):
            # If conversion fails, it's a text column
            col_type = 'text'
            sample_value = df[col][0]

        # Create a Column object
        header.append(Column(col, col_type, sample_value=sample_value))
        
        # Add the column data to 'data' list
    for row_index in range(len(df)):
        data.append(list(df.iloc[row_index]))
        # print()
        # print(col_type)
        # print(sample_value)
    # Create the Table
    table = Table(id='', header=header, data=data)

    # Tokenize
    table.tokenize(tokenizer)

    return table

def get_tabert_embeddings(tables, model_path = '/home/zjsun/TaBert/TaBERT/tabert_base_k3/model.bin'):
    device = torch.device("cuda")
    print()
    print(device)
    print()
    model = TableBertModel.from_pretrained(
        model_path,
    )
    model = model.to(device)
    all_embeddings = []
    
    for table in tables:
        table = table.reset_index(drop=True)
        table = table.astype(str)
        table = convert_to_table(table, model.tokenizer)
        context = ''
        with torch.no_grad():
            context_encoding, column_encoding, info_dict = model.encode(
                contexts=[model.tokenizer.tokenize(context)],
                tables=[table]
            )
        embeddings = column_encoding[0]
        all_embeddings.append(embeddings)
        # Free up some memory by deleting column_encoding and info_dict variables
        del column_encoding
        del info_dict
        del context_encoding
        del embeddings
        # Empty the cache
        torch.cuda.empty_cache()
    
    return all_embeddings
 
    