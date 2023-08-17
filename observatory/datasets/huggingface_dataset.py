from observatory.common_util.column_based_truncate import (
    max_rows,
    table2colList,
    max_rows,
)
import pandas as pd
from observatory.models.hugging_face_column_embeddings import column_based_process_table
from observatory.models.hugging_face_cell_embeddings import cell_based_process_table

from pandas import DataFrame


def chunk_tables(tables, max_col, model_name, max_length, tokenizer, max_row=None):
    chunked_tables = {}

    # Iterate over each table
    for table_index, table in enumerate(tables):
        # Convert table to pandas DataFrame for easier manipulation
        df = DataFrame(table)

        chunked_tables[table_index] = {}
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
                if max_row:
                    approxiamte_optimal_rows = max_row
                end_row = min(start_row + approxiamte_optimal_rows, chunk.shape[0])
                sub_chunk = chunk.iloc[start_row:end_row, :]
                cols = table2colList(sub_chunk)
                optimal_rows = max_rows(cols, tokenizer, max_length, model_name)

                # Ensure optimal_rows is at least 1 to prevent infinite loops
                if optimal_rows == 0:
                    start_row = start_row + 1
                    continue

                truncated_chunk = sub_chunk.iloc[:optimal_rows, :]

                # Store the chunk with its start and end row indices
                if (start_col, end_col) not in chunked_tables[table_index]:
                    chunked_tables[table_index][(start_col, end_col)] = {}
                chunked_tables[table_index][(start_col, end_col)][
                    (start_row, start_row + optimal_rows)
                ] = truncated_chunk

                start_row = start_row + optimal_rows

            start_col = end_col
    
    
    chunked_list = []
    for table_index, column_chunks in chunked_tables.items():
        for position, chunk in column_chunks.items():
            chunked_list.append(
                {"table": chunk, "position": position, "index": table_index}
            )
    return chunked_list


from torch.utils.data import Dataset, DataLoader
from pandas import DataFrame
import torch


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


