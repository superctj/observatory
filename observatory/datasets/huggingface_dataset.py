from observatory.common_util.column_based_truncate import (
    column_based_truncate,
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


def truncate_cell(text, tokenizer, max_token_per_cell):
    """
    Truncate the text in a cell to the max_token_per_cell limit.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_token_per_cell:
        tokens = tokens[:max_token_per_cell]
    return tokenizer.convert_tokens_to_string(tokens)

def chunk_neighbor_tables_template(tables, column_name, n, max_length, max_row=None, max_token_per_cell=None):
    """
    Chunk tables based on a central column and its neighbors.
    """

    for table_index, df in enumerate(tables):
        
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in table {table_index}. Skipping...")
            continue
        
        # Find the index of the specified column
        col_index = df.columns.get_loc(column_name)
        
        # Determine the range of columns to select based on n
        start_col_idx = max(0, col_index - n)
        end_col_idx = min(df.shape[1], col_index + n + 1)
        
        # Extract the central and neighboring columns
        chunk = df.iloc[:, start_col_idx:end_col_idx]
        
        # Integrate the chunking mechanism from the previous function
        start_row = 0
        while start_row < chunk.shape[0]:
            optimal_rows = max_length // chunk.shape[1]
            if max_token_per_cell:
                optimal_rows = max_length // (chunk.shape[1] * max_token_per_cell)
            if max_row:
                optimal_rows = min(max_row, optimal_rows)
            end_row = min(start_row + optimal_rows, chunk.shape[0])
            truncated_chunk = chunk.iloc[start_row:end_row, :]
            
            # Yield the chunk with its start and end row indices and other relevant information
            yield {
                "table": truncated_chunk,
                "position": ((start_col_idx, end_col_idx), (start_row, start_row + optimal_rows)),
                "index": table_index
            }

            start_row = start_row + optimal_rows

def chunk_neighbor_tables(tables, column_name, n, model_name, max_length, tokenizer, max_row=None, max_token_per_cell=None):
    """
    Chunk tables based on a central column and its neighbors.
    """

    for table_index, df in enumerate(tables):
        
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in table {table_index}. Skipping...")
            continue
        
        # Find the index of the specified column
        col_index = df.columns.get_loc(column_name)
        
        # Determine the range of columns to select based on n
        start_col_idx = max(0, col_index - n)
        end_col_idx = min(df.shape[1], col_index + n + 1)
        
        # Extract the central and neighboring columns
        chunk = df.iloc[:, start_col_idx:end_col_idx]
        
        # Integrate the chunking mechanism from the previous function
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
            
            # Ensure each cell adheres to the max_token_per_cell limit
            if max_token_per_cell:
                for col in sub_chunk.columns:
                    sub_chunk[col] = sub_chunk[col].apply(lambda x: truncate_cell(str(x), tokenizer, max_token_per_cell))
            
            # Assuming table2colList and truncate_index are some external functions
            optimal_rows = column_based_truncate(df, tokenizer, max_length, model_name)

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
                "position": ((start_col_idx, end_col_idx), (start_row, start_row + optimal_rows)),
                "index": table_index
            }

            start_row = start_row + optimal_rows


def chunk_tables(tables, model_name, max_length, tokenizer, max_col=None, max_row=None, max_token_per_cell=None):

    # Iterate over each table
    for table_index, df in enumerate(tables):
        print(table_index)
        # Convert table to pandas DataFrame for easier manipulation
        # df = DataFrame(table)
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
                if max_token_per_cell:
                    for col in sub_chunk.columns:
                        sub_chunk[col] = sub_chunk[col].apply(lambda x: truncate_cell(str(x), tokenizer, max_token_per_cell))
                optimal_rows = column_based_truncate(df, tokenizer, max_length, model_name)  # Assuming max_rows is some external function

                # Ensure optimal_rows is at least 1 to prevent infinite loops
                if optimal_rows == 0:
                    print("Column headers too many, please set a valid max_col or reduce the current max_col!")
                    print("You may also try to set a valid max_token_per_cell or reduce the current max_token_per_cell")
                    start_row = start_row + 1
                    continue

                truncated_chunk = sub_chunk.iloc[:optimal_rows, :]
                # print(truncated_chunk)
                # Yield the chunk with its start and end row indices and other relevant information
                yield {
                    "table": truncated_chunk,
                    "position": ((start_col, end_col), (start_row, start_row + optimal_rows)),
                    "index": table_index
                }

                start_row = start_row + optimal_rows

            start_col = end_col


def batch_generator(generator, batch_size):
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch



class TableColumnDataset(Dataset):
    def __init__(self, chunk_generator, tokenizer, max_length, padding_token, model_name):
        self.tables = list(chunk_generator)
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
            return (inputs, table_position, table_index)
        else:
            cols = table2colList(table)
            # print(cols)
            processed_tokens, cls_positions = column_based_process_table(
                self.tokenizer, cols, self.max_length, self.model_name
            )
            # print(processed_tokens)
            input_ids = self.tokenizer.convert_tokens_to_ids(processed_tokens)
            # print(input_ids)
            attention_mask = [
                1 if token != self.padding_token else 0 for token in processed_tokens
            ]
            return (torch.tensor(input_ids, dtype=torch.long), 
                    torch.tensor(attention_mask, dtype=torch.long), 
                    cls_positions, table_position, table_index)
     
       
class TableCellDataset(Dataset):
    def __init__(self, chunk_generator, tokenizer, max_length, padding_token, model_name):
        self.tables = list(chunk_generator)
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
            return (inputs, table_position, table_index)
        else:
            cols = [list(table[column]) for column in table.columns]
            processed_tokens, token_positions = cell_based_process_table(
                self.tokenizer, cols, self.max_length, self.model_name
            )
            input_ids = self.tokenizer.convert_tokens_to_ids(processed_tokens)
            attention_mask = [
                1 if token != self.padding_token else 0 for token in processed_tokens
            ]
            return (torch.tensor(input_ids, dtype=torch.long), 
                    torch.tensor(attention_mask, dtype=torch.long), 
                    token_positions, table_position, table_index)



