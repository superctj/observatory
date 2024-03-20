import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from typing import Callable, Generator
import argparse
import torch
import functools

from observatory.datasets.sotab_loader import SOTABDataLoader
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length

)

from observatory.models.hugging_face_column_embeddings import (
    get_hugging_face_column_embeddings_batched,

)
import pandas as pd

device = torch.device("cpu")


def split_table(
    table: pd.DataFrame, 
    m: int, 
    n: int
) -> Generator:
    """ Split a table into chunks with at most m rows and n columns.
    
    Args:
        table: The table to split.
        m: The maximum number of rows in each chunk.
        n: The maximum number of columns in each chunk.
    
    Returns:
        Iterator over the chunks of the table.
    """
    total_rows = table.shape[0]
    for i in range(0, total_rows, m * n):
        yield [table.iloc[j : j + m] for j in range(i, min(i + m * n, total_rows), m)]


def get_average_embedding(
    # table, index, n, get_embedding):
    table: pd.DataFrame,
    index: int,
    n: int,
    get_embedding: Callable,
) -> torch.Tensor:
    """ Get the average embedding of a table.
    
    Args:
        table: The table to get the average embedding of.
        index: The index of the column to get the average embedding of.
        n: The maximum number of columns in each chunk that the table is split into.
        get_embedding: A callable that takes a pandas DataFrame and returns a torch.Tensor of embeddings.
        
    Returns:
        The average column embeddings of the table.
    """
    m = min(100 // len(list(table.columns)), 3)
    sum_embeddings = None
    num_embeddings = 0
    chunks_generator = split_table(table, m=m, n=n)
    for tables in chunks_generator:
        embeddings = get_embedding(tables)
        if sum_embeddings is None:
            sum_embeddings = torch.zeros(embeddings[0][index].size())
        for embedding in embeddings:
            sum_embeddings += embedding[index].to(device)
            num_embeddings += 1
    avg_embedding = sum_embeddings / num_embeddings
    return avg_embedding


if __name__ == "__main__":
    # root_dir = "/ssd/congtj/observatory/sotab_numerical_data_type_datasets"
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--save_folder", type=str, default="p6")
    parser.add_argument("--metadata_path", type=str, default="metadata.csv")

    # parser.add_argument('--r', type=int, required=True)
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the Hugging Face model to use",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The batch size for parallel inference",
    )
    args = parser.parse_args()
    model_name = args.model_name
    save_directory_results = os.path.join(args.save_folder, model_name)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)
    n = args.n
    root_dir = args.root_dir
    dataset_dir = os.path.join(root_dir, "tables")
    metadata_path = os.path.join(root_dir, args.metadata_path)

    data_loader = SOTABDataLoader(dataset_dir, metadata_path)
    tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
    
    model = load_transformers_model(model_name, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    get_embedding = functools.partial(
        get_hugging_face_column_embeddings_batched, model_name=model_name, \
            tokenizer=tokenizer, max_length=max_length, model=model, batch_size=args.batch_size
    )

    col_itself = []
    subj_col_as_context = []
    neighbor_col_as_context = []
    entire_table_as_context = []
    # with open(f'output_{model_name.replace("/", "")}.txt', "w") as f:
    #     f.write(f"Error message for {model_name}\n\n")
    for _, row in data_loader.metadata.iterrows():
        table_name = row["table_name"]
        table = data_loader.read_table(table_name)
        table.columns = [" "] * len(table.columns)

        # input_tables = []
        # Only consider numerical column alone for representation inference
        numerical_col_idx = row["column_index"]
        numerical_col = table.iloc[:, [numerical_col_idx]]
        # input_tables.append(numerical_col)

        # Consider the subject column as context of numerical column for representation inference
        subj_col_idx = row["subject_column_index"]
        two_col_table = table.iloc[:, [subj_col_idx, numerical_col_idx]]
        # input_tables.append(two_col_table)

        # Consider immediate neighboring columns as context of numerical column for representation inference
        num_cols = len(list(table.columns))

        if numerical_col_idx > 0 and numerical_col_idx < num_cols - 1:
            three_col_table = table.iloc[
                :, [numerical_col_idx - 1, numerical_col_idx, numerical_col_idx + 1]
            ]
        elif numerical_col_idx == num_cols - 1:
            three_col_table = table.iloc[:, [numerical_col_idx - 1, numerical_col_idx]]

        # input_tables.append(three_col_table)

        # Consider the entire table as context of numerical column for representation inference
        # input_tables.append(table)

        try:
            col_itself_embedding = get_average_embedding(
                numerical_col, 0, n, get_embedding
            )
        except Exception as e:
            print(f"In table: {table_name}\n")
            print(
                "In col_itself_embedding = get_average_embedding(numerical_col, 0, n,  get_embedding): "
            )
            print("Error message:", e)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            # print(numerical_col.columns)
            print(numerical_col)
            # with open(f"output_{model_name}.txt", "a") as f:
            #     f.write(f"In table: {table_name}\n")
            #     f.write(
            #         "In col_itself_embedding = get_average_embedding(numerical_col, 0, n,  get_embedding): "
            #     )
            #     f.write(f"Error message: {e}\n\n")
            #     f.write(f"\n\n")
            continue

        try:
            subj_col_as_context_embedding = get_average_embedding(
                two_col_table, 1, n, get_embedding
            )
        except Exception as e:
            print(f"In table: {table_name}\n")
            print(
                "In subj_col_as_context_embedding = get_average_embedding(two_col_table, 1, n,  get_embedding) "
            )
            print("Error message:", e)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            # print(numerical_col.columns)
            print(numerical_col)
            # with open(f"output_{model_name}.txt", "a") as f:
            #     f.write(f"In table: {table_name}\n")
            #     f.write(
            #         "In subj_col_as_context_embedding = get_average_embedding(two_col_table, 1, n,  get_embedding) "
            #     )
            #     f.write(f"Error message: {e}\n\n")
            #     f.write(f"\n\n")
            continue
        # subj_col_as_context_embedding = get_average_embedding(two_col_table, 1, n,  get_embedding)

        try:
            neighbor_col_as_context_embedding = get_average_embedding(
                three_col_table, 1, n, get_embedding
            )
        except Exception as e:
            print(f"In table: {table_name}\n")
            print(
                "In neighbor_col_as_context_embedding = get_average_embedding(three_col_table, 1, n,  get_embedding) "
            )
            print("Error message:", e)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            # print(numerical_col.columns)
            print(numerical_col)
            # with open(f"output_{model_name}.txt", "a") as f:
            #     f.write(f"In table: {table_name}\n")
            #     f.write(
            #         "In neighbor_col_as_context_embedding = get_average_embedding(three_col_table, 1, n,  get_embedding) "
            #     )
            #     f.write(f"Error message: {e}\n\n")
            #     f.write(f"\n\n")
            continue

        try:
            entire_table_as_context_embedding = get_average_embedding(
                table, numerical_col_idx, n, get_embedding
            )
        except Exception as e:
            print(f"In table: {table_name}\n")
            print(
                "In entire_table_as_context_embedding = get_average_embedding(table, numerical_col_idx, n,  get_embedding) "
            )
            print("Error message:", e)
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            # print(numerical_col.columns)
            print(numerical_col)
            # with open(f"output_{model_name}.txt", "a") as f:
            #     f.write(f"In table: {table_name}\n")
            #     f.write(
            #         "In entire_table_as_context_embedding = get_average_embedding(table, numerical_col_idx, n,  get_embedding) "
            #     )
            #     f.write(f"Error message: {e}\n\n")
            #     f.write(f"\n\n")
            continue

        col_itself.append((col_itself_embedding, row["label"]))
        subj_col_as_context.append((subj_col_as_context_embedding, row["label"]))
        neighbor_col_as_context.append(
            (neighbor_col_as_context_embedding, row["label"])
        )
        entire_table_as_context.append(
            (entire_table_as_context_embedding, row["label"])
        )

    data = {}
    data["col_itself"] = col_itself
    data["subj_col_as_context"] = subj_col_as_context
    data["neighbor_col_as_context"] = neighbor_col_as_context
    data["entire_table_as_context"] = entire_table_as_context
    torch.save(data, os.path.join(save_directory_results, f"embeddings.pt"))
