from observatory.common_util.column_based_truncate import (
    truncate_index,
    table2colList,
)
import pandas as pd
from observatory.models.hugging_face_column_embeddings import column_based_process_table
from observatory.models.hugging_face_cell_embeddings import cell_based_process_table
import random
from torch.utils.data import Dataset, DataLoader
import torch
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer






def chunk_tables(tables, model_name, max_length, tokenizer, max_col=None, max_row=None, max_token_per_cell=None):

    # Iterate over each table
    for table_index, table in enumerate(tables):
        # Convert table to pandas DataFrame for easier manipulation
        df = DataFrame(table)
        if not max_col:
            max_col = df.shape[1]

        # Split each table into chunks based on max_col
        start_col = 0
        while start_col < df.shape[1]:
            end_col = start_col + max_col
            chunk = df.iloc[:, start_col:end_col]

            # Split each chunk into sub-chunks based on max_row
            start_row = 0
            while start_row < chunk.shape[0]:
                optimal_rows = 0
                approxiamte_optimal_rows = max_length // chunk.shape[1]
                if max_token_per_cell:
                    approxiamte_optimal_rows = max_length // (chunk.shape[1] * max_token_per_cell)
                    if max_row:
                        approxiamte_optimal_rows = min(max_row, approxiamte_optimal_rows)
                end_row = min(start_row + approxiamte_optimal_rows, chunk.shape[0])
                sub_chunk = chunk.iloc[start_row:end_row, :]
                cols = table2colList(sub_chunk)  # Assuming table2colList is some external function
                optimal_rows = truncate_index(cols, tokenizer, max_length, model_name)  # Assuming max_rows is some external function

                # Ensure optimal_rows is at least 1 to prevent infinite loops
                if optimal_rows == 0:
                    print("Column headers too many, please set a valid max_col or reduce the current max_col!")
                    print("You may also try to set a valid max_token_per_cell or reduce the current max_token_per_cell")
                    start_row = start_row + 1
                    continue

                truncated_chunk = sub_chunk.iloc[:optimal_rows, :]

                # Yield the chunk with its start and end row indices and other relevant information
                yield {
                    "table": truncated_chunk,
                    "position": ((start_col, end_col), (start_row, start_row + optimal_rows)),
                    "index": table_index
                }

                start_row = start_row + optimal_rows

            start_col = end_col





class TableColumnDataset(Dataset):
    def __init__(self, tables, tokenizer, max_length, padding_token, model_name):
        self.tables = tables
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name
        self.padding_token = padding_token

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        table_info = self.tables[idx]
        table = table_info["table"]
        table_position = table_info["position"]
        table_index = table_info["index"]

        if self.model_name.startswith("google/tapas"):
            table.columns = table.columns.astype(str)
            table = table.reset_index(drop=True)
            table = table.astype(str)
            inputs = self.tokenizer(
                table=table, padding="max_length", return_tensors="pt", truncation=True
            )
            return {"inputs": inputs, "position": table_position, "index": table_index}
        else:
            cols = table2colList(table)
            processed_tokens, cls_positions = column_based_process_table(
                self.tokenizer, cols, self.max_length, self.model_name
            )
            input_ids = self.tokenizer.convert_tokens_to_ids(processed_tokens[0])
            attention_mask = [
                1 if token != self.padding_token else 0 for token in processed_tokens[0]
            ]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "cls_positions": cls_positions,
                "position": table_position,
                "index": table_index,
            }
            
class TableCellDataset(Dataset):
    def __init__(self, tables, tokenizer, max_length, padding_token, model_name):
        self.tables = tables
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_name = model_name
        self.padding_token = padding_token

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):
        table_info = self.tables[idx]
        table = table_info["table"]
        table_position = table_info["position"]
        table_index = table_info["index"]

        if self.model_name.startswith("google/tapas"):
            table.columns = table.columns.astype(str)
            table = table.reset_index(drop=True)
            table = table.astype(str)
            inputs = self.tokenizer(
                table=table, padding="max_length", return_tensors="pt", truncation=True
            )
            return {"inputs": inputs, "position": table_position, "index": table_index}
        else:
            cols = [list(table[column]) for column in table.columns]
            processed_tokens, token_positions = cell_based_process_table(
                self.tokenizer, cols, self.max_length, self.model_name
            )
            input_ids = self.tokenizer.convert_tokens_to_ids(processed_tokens)
            attention_mask = [
                1 if token != self.padding_token else 0 for token in processed_tokens
            ]
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "token_positions": token_positions,
                "position": table_position,
                "index": table_index,
            }


