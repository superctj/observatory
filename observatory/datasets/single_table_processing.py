import random

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def n_gram_overlap(text: str, row_text: str, n: int):
    """
    Calculate the n-gram overlap between two texts.

    Parameters:
    - text (str): First text string.
    - row_text (str): Second text string.
    - n (int): N in n-gram. Default is 2.

    Returns:
    - overlap (int): Overlap score between the two texts.
    """

    vectorizer = CountVectorizer(ngram_range=(n, n))
    vectorizer.fit([text, row_text])

    counts = vectorizer.transform([text, row_text])
    overlap = sum(counts.toarray()[0] * counts.toarray()[1])

    return overlap


def get_most_similar_rows(df, text_to_compare, n=1, k=None):
    """
    Get the most similar rows based on n-gram overlap with a given text.

    Parameters:
    - df (pd.DataFrame): DataFrame to compare.
    - text_to_compare (str): Text to compare with each row.
    - n (int): N in n-gram. Default is 1.
    - k (int): Number of top rows to return. If None, return all rows. Default
               is None.

    Returns:
    - pd.DataFrame: DataFrame with the most similar rows.
    """

    df_str = df.apply(lambda row: " ".join(row.astype(str)), axis=1)
    overlaps = df_str.apply(
        lambda row_text: n_gram_overlap(text_to_compare, row_text, n)
    )
    # print(overlaps)
    sorted_indices = overlaps.sort_values(ascending=False).index
    if k:
        sorted_indices = sorted_indices[:k]

    return df.loc[sorted_indices]


def get_most_similar_cols(df, text_to_compare, n=1, k=None):
    """
    Get the most similar columns based on n-gram overlap with a given text.

    Parameters:
    - df (pd.DataFrame): DataFrame to compare.
    - text_to_compare (str): Text to compare with each column.
    - n (int): N in n-gram. Default is 1.
    - k (int): Number of top columns to return. If None, return all columns.
               Default is None.

    Returns:
    - pd.DataFrame: DataFrame with the most similar columns.
    """

    overlaps = {
        col: n_gram_overlap(text_to_compare, " ".join(df[col].astype(str)), n)
        for col in df.columns
    }

    sorted_columns = sorted(overlaps, key=overlaps.get, reverse=True)
    if k:
        sorted_columns = sorted_columns[:k]

    return df[sorted_columns]


def random_sample_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns a random sample of k rows from a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - k (int): The number of rows to sample.

    Returns:
    - pd.DataFrame: A DataFrame with k random rows.
    """
    if k > len(df):
        raise ValueError(
            f"DataFrame only contains {len(df)} rows. Cannot sample {k} rows."
        )
    return df.sample(n=k)


def random_sample_cols(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns a random sample of k columns from a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - k (int): The number of columns to sample.

    Returns:
    - pd.DataFrame: A DataFrame with k random columns.
    """

    if k > len(df.columns):
        raise ValueError(
            f"DataFrame only contains {len(df.columns)} columns. Cannot sample "
            f"{k} columns."
        )
    random_cols = random.sample(list(df.columns), k)

    return df[random_cols]


def sample_top_k_rows(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns the top k rows from a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - k (int): The number of rows to retrieve.

    Returns:
    - pd.DataFrame: A DataFrame with the top k rows.
    """

    return df.head(k)


def sample_top_k_cols(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """
    Returns the first k columns from a given DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - k (int): The number of columns to retrieve.

    Returns:
    - pd.DataFrame: A DataFrame with the first k columns.
    """
    if k > len(df.columns):
        raise ValueError(
            f"DataFrame only contains {len(df.columns)} columns. Cannot "
            f"retrieve {k} columns."
        )
    return df.iloc[:, :k]


def rank_rows_by_tfidf(
    source_df: pd.DataFrame, top_n: int = None, drop_score=True
) -> pd.DataFrame:
    """
    Rank rows of a DataFrame based on the summed TF-IDF scores after
    concatenating all columns in each row.

    Parameters:
    - source_df: The input DataFrame.
    - top_n: Number of top rows to return. If None, returns all rows.

    Returns:
    - A new DataFrame containing rows sorted by importance based on TF-IDF,
      without the tfidf_sum column.
    """

    df = source_df.copy()
    # Concatenate all columns for each row and treat entire row as a single
    # document
    df["combined"] = df.apply(lambda row: " ".join(map(str, row)), axis=1)

    # Calculate TF-IDF scores for each term in the combined rows
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["combined"])

    # Sum up the scores for each row
    df["tfidf_sum"] = tfidf_matrix.sum(axis=1)

    # Sort rows based on their summed scores
    df_sorted = df.sort_values(by="tfidf_sum", ascending=False)

    # Drop the 'combined' and 'tfidf_sum' columns as they're no longer needed
    df_sorted.drop(
        columns=[
            "combined",
        ],
        inplace=True,
    )
    if drop_score:
        df_sorted.drop(columns=["tfidf_sum"], inplace=True)

    # Return top rows based on the summed TF-IDF scores
    return df_sorted.head(top_n) if top_n else df_sorted
