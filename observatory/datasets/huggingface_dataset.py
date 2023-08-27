from observatory.common_util.column_based_truncate import (
    max_rows,
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

def n_gram_overlap(text, row_text, n=2):
    vectorizer = CountVectorizer(ngram_range=(n, n))
    counts = vectorizer.fit_transform([text, row_text])
    overlap = sum((counts.toarray()[0] & counts.toarray()[1]) > 0)
    return overlap

def get_most_similar_rows(df, text_to_compare, n=2, k=2):
    overlaps = []
    for _, row in df.iterrows():
        row_text = ' '.join(row.astype(str))
        overlap = n_gram_overlap(text_to_compare, row_text, n=n)
        overlaps.append((overlap, row))
    
    # Sort by overlap and take the top k rows
    sorted_rows = sorted(overlaps, key=lambda x: x[0], reverse=True)[:k]
    most_similar_rows = pd.DataFrame([row for _, row in sorted_rows])
    
    return most_similar_rows

def get_most_similar_cols(df, text_to_compare, n=2, k=2):
    overlaps = {}
    # Compute overlap for each column
    for col in df.columns:
        overlaps[col] = n_gram_overlap(text_to_compare, df[col], n=n)
        
    # Sort columns by overlap
    sorted_columns = sorted(overlaps, key=overlaps.get, reverse=True)[:k]
    
    # Keep the most similar k columns
    most_similar_cols_df = df[sorted_columns]
    
    return most_similar_cols_df

def random_sample_rows(df, k):
    return df.sample(n=k)

def random_sample_cols(df, k):
    random_cols = random.sample(list(df.columns), k)
    return df[random_cols]

def sample_top_k_rows(df, k):
    return df.head(k)

def sample_top_k_cols(df, k):
    return df.iloc[:, :k]



def filter_rows_by_tfidf(df, text_column, top_n=None, threshold=None):
    """
    Filter rows in a DataFrame based on high TF-IDF scores of a text column.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - text_column (str): Name of the column containing text data.
    - top_n (int, optional): Number of top rows to keep. If None, all rows above the threshold are kept.
    - threshold (float, optional): Minimum sum of TF-IDF scores to keep a row. If None, no threshold is applied.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    # Generate TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(df[text_column])
    
    # Calculate the sum of TF-IDF scores for each row
    tfidf_sum = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Create a new DataFrame with TF-IDF sum
    df_tfidf = df.copy()
    df_tfidf['tfidf_sum'] = tfidf_sum
    
    # Sort DataFrame by TF-IDF sum
    df_tfidf = df_tfidf.sort_values(by='tfidf_sum', ascending=False)
    
    # Apply top_n and threshold filters
    if top_n is not None:
        df_tfidf = df_tfidf.head(top_n)
    if threshold is not None:
        df_tfidf = df_tfidf[df_tfidf['tfidf_sum'] >= threshold]
        
    # Drop the TF-IDF sum column before returning
    df_tfidf.drop(columns=['tfidf_sum'], inplace=True)
    
    return df_tfidf

def filter_rows_by_avg_tfidf(df, top_n=None, threshold=None):
    """
    Filter rows in a DataFrame based on high average TF-IDF scores across all text columns.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - top_n (int, optional): Number of top rows to keep. If None, all rows above the threshold are kept.
    - threshold (float, optional): Minimum average TF-IDF scores to keep a row. If None, no threshold is applied.
    
    Returns:
    - pd.DataFrame: Filtered DataFrame.
    """
    
    # Initialize a list to store TF-IDF sums for each row
    tfidf_sums = []
    
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    
    for col in df.columns:
        # Check if the column contains text data
        if df[col].dtype == 'object':
            # Generate TF-IDF matrix for the column
            tfidf_matrix = vectorizer.fit_transform(df[col])
            
            # Calculate the sum of TF-IDF scores for each row
            tfidf_sum = np.sum(tfidf_matrix.toarray(), axis=1)
            
            # Append to the list of TF-IDF sums
            tfidf_sums.append(tfidf_sum)
    
    # Calculate the average TF-IDF score for each row
    avg_tfidf = np.mean(np.array(tfidf_sums), axis=0)
    
    # Create a new DataFrame with the average TF-IDF score
    df_tfidf = df.copy()
    df_tfidf['avg_tfidf'] = avg_tfidf
    
    # Sort DataFrame by average TF-IDF score
    df_tfidf = df_tfidf.sort_values(by='avg_tfidf', ascending=False)
    
    # Apply top_n and threshold filters
    if top_n is not None:
        df_tfidf = df_tfidf.head(top_n)
    if threshold is not None:
        df_tfidf = df_tfidf[df_tfidf['avg_tfidf'] >= threshold]
        
    # Drop the average TF-IDF score column before returning
    df_tfidf.drop(columns=['avg_tfidf'], inplace=True)
    
    return df_tfidf





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
                optimal_rows = max_rows(cols, tokenizer, max_length, model_name)  # Assuming max_rows is some external function

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


