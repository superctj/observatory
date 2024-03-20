import os
import pandas as pd
import numpy as np
import argparse
import functools
import torch
from typing import Dict, List
from observatory.models.hugging_face_cell_embeddings import (
    get_hugging_face_cell_embeddings,
)
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)

from observatory.models.doduo_entity_embeddings import Doduo

import pandas as pd


class SpiderFDDataLoader:
    def __init__(
        self, dataset_dir: str, fd_metadata_path: str, non_fd_metadata_path: str
    ):
        self.dataset_dir = dataset_dir
        self.fd_metadata_path = fd_metadata_path
        self.non_fd_metadata_path = non_fd_metadata_path

    def read_table(self, table_name: str, drop_nan=True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, table_name)
        table = pd.read_csv(
            table_path, on_bad_lines="skip", lineterminator="\n", **kwargs
        )

        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_fd_metadata(self):
        fd_metadata = pd.read_csv(self.fd_metadata_path, sep=",")
        return fd_metadata

    def get_non_fd_metadata(self):
        non_fd_metadata = pd.read_csv(self.non_fd_metadata_path, sep=",")
        return non_fd_metadata


import pandas as pd


def find_groups(
    df: pd.DataFrame,
    determinant_col: str,
    dependent_col: str
) -> Dict:
    """Find groups of rows with the same pair of values in the determinant and dependent columns.
    
    Args:
        df: The DataFrame to find groups in.
        determinant_col: The name of the column that determines the group.
        dependent_col: The name of the column that is dependent on the determinant column. 
    
    Returns:
        A dictionary with the pair of values as the key and a list of indices as the value.
    """
    # Create a new DataFrame with only the two columns of interest and the index
    df_temp = df[[determinant_col, dependent_col]].reset_index()
    df_temp["pair"] = list(zip(df_temp[determinant_col], df_temp[dependent_col]))

    # Group by the new pair column and aggregate indices into lists
    groups = df_temp.groupby("pair")["index"].apply(list)

    # Convert the Series to a dictionary
    pairs_dict = groups.to_dict()

    return pairs_dict


if __name__ == "__main__":
    # root_dir = f"/ssd/congtj/observatory/spider_datasets/fd_artifact"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
        help="Name of the Hugging Face model to use",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save the results",
    )
    parser.add_argument(
        "-r", "--root_dir", type=str, required=True, help="Root directory"
    )
    parser.add_argument("--mode", type=str, default="Both", help="Root directory")
    parser.add_argument(
        "--doduo_path",
        type=str,
        default=".",
        help="Path to load the doduo model",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=32,
        help="The batch size for parallel inference",
    )
    args = parser.parse_args()
    root_dir = args.root_dir
    dataset_dir = os.path.join(root_dir, "datasets")
    fd_metadata_path = os.path.join(root_dir, "fd_metadata.csv")
    non_fd_metadata_path = os.path.join(root_dir, "non_fd_metadata.csv")
    data_loader = SpiderFDDataLoader(
        dataset_dir, fd_metadata_path, non_fd_metadata_path
    )

    model_name = args.model_name
    save_directory = os.path.join(args.save_dir, "Functional_Dependencies", model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    # save_directory_cell = os.path.join(save_directory, "original_cell_embeddings")
    # save_directory_pairs = os.path.join(save_directory, "pair_embeddings")
    # if not model_name.startswith("doduo"):
    #     if not os.path.exists(save_directory_cell):
    #         os.makedirs(save_directory_cell)
    # if not os.path.exists(save_directory_pairs):
    #     os.makedirs(save_directory_pairs)
    if (
        model_name.startswith("bert")
        or model_name.startswith("roberta")
        or model_name.startswith("google/tapas")
        or model_name.startswith("t5")
    ):
        tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        model = load_transformers_model(model_name, device)
        model.eval()
        get_embedding = functools.partial(
            get_hugging_face_cell_embeddings,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            device=device,
        )
    elif model_name.startswith("doduo"):
        model_args = argparse.Namespace
        model_args.model = "wikitable"
        doduo = Doduo(model_args, basedir=args.doduo_path)
        get_embedding = doduo.get_entity_embeddings

    if args.mode == "Both" or args.mode == "FD":
        with open("FD.txt", "w") as file:
            file.write("Record for list_row_index\n")
        with open(
            f'FD embedding_length_{model_name.replace("/", "")}.txt', "w"
        ) as file:
            file.write("Record for embedding_length\n")
        fd_metadata = data_loader.get_fd_metadata()
        list_pairs_norms_dict = []
        for _, row in fd_metadata.iterrows():
            table_name = row["table_name"]
            determinant = row["determinant"]
            dependent = row["dependent"]

            table = data_loader.read_table(table_name)
            determinant_index = list(table.columns).index(determinant)
            dependent_index = list(table.columns).index(dependent)
            pairs_dict = find_groups(table, determinant, dependent)
            norms_dict = {}

            if model_name.startswith("doduo"):
                for pair, list_row_index in pairs_dict.items():
                    with open("FD.txt", "a") as file:
                        file.write(str(pair))
                        file.write("\n")
                        file.write(str(list_row_index))
                        file.write("\n")
                    l2_norms = []
                    for row_index in list_row_index:
                        # Start with a range of two rows above and below
                        for delta in [2, 1, 0]:
                            try:
                                # Create a smaller table based on the current row index and delta
                                min_index = max(0, row_index - delta)
                                max_index = min(len(table), row_index + delta + 1)
                                part_table = table.iloc[min_index:max_index]
                                part_table = part_table.astype(str)
                                # Adjust the index for the cell_embeddings, considering the boundaries
                                adjusted_index = min(delta, row_index - min_index)

                                tmp_pairs = []
                                tmp_pairs.append(
                                    [[adjusted_index, determinant_index], (-1, "")]
                                )
                                tmp_pairs.append(
                                    [[adjusted_index, dependent_index], (-1, "")]
                                )

                                # Get the entity_embeddings of the part table
                                try:
                                    entity_embeddings = get_embedding(
                                        part_table, tmp_pairs
                                    )
                                except ValueError as e:
                                    print(e)
                                    continue

                                # Get the determinant_embedding and dependent_embedding
                                determinant_embedding, _ = entity_embeddings[
                                    (adjusted_index, determinant_index)
                                ]
                                dependent_embedding, _ = entity_embeddings[
                                    (adjusted_index, dependent_index)
                                ]

                                # If either the determinant_embedding or dependent_embedding is 0, continue to next delta
                                if (
                                    torch.norm(determinant_embedding, p=2) == 0
                                    or torch.norm(dependent_embedding, p=2) == 0
                                ):
                                    continue
                                with open(
                                    f'FD embedding_length_{model_name.replace("/", "")}.txt',
                                    "a",
                                ) as file:
                                    file.write(
                                        f"determinant_embedding length {torch.norm(determinant_embedding, p=2)}\n"
                                    )
                                    file.write(
                                        f"dependent_embedding length {torch.norm(dependent_embedding, p=2)}\n"
                                    )

                                # Calculate the L2 norm and add it to the list
                                l2_norm = torch.norm(
                                    determinant_embedding - dependent_embedding, p=2
                                )
                                l2_norms.append(
                                    l2_norm.item()
                                )  # Convert the torch tensor to a Python float

                                # Break the loop once we get a valid pair of embeddings
                                break
                            except (KeyError, IndexError, AssertionError) as e:
                                print(e)
                                continue

                        if (
                            not l2_norms
                        ):  # If l2_norms is still empty after the loop, continue to the next row
                            continue

                    norms_dict[pair] = l2_norms

            else:
                for pair, list_row_index in pairs_dict.items():
                    with open("FD.txt", "a") as file:
                        file.write(str(pair))
                        file.write("\n")
                        file.write(str(list_row_index))
                        file.write("\n")
                    l2_norms = []
                    for row_index in list_row_index:
                        # Start with a range of two rows above and below
                        for delta in [2, 1, 0]:
                            try:
                                # Create a smaller table based on the current row index and delta
                                min_index = max(0, row_index - delta)
                                max_index = min(len(table), row_index + delta + 1)
                                part_table = table.iloc[min_index:max_index]
                                part_table = part_table.astype(str)
                                # Get the cell embeddings of the part table
                                cell_embeddings = get_embedding(part_table)
                                # Adjust the index for the cell_embeddings, considering the boundaries
                                adjusted_index = min(delta, row_index - min_index) + 1
                                # Get the determinant_embedding and dependent_embedding
                                determinant_embedding = cell_embeddings[adjusted_index][
                                    determinant_index
                                ]
                                dependent_embedding = cell_embeddings[adjusted_index][
                                    dependent_index
                                ]

                                # If either the determinant_embedding or dependent_embedding is 0, continue to next delta
                                if (
                                    torch.norm(determinant_embedding, p=2) == 0
                                    or torch.norm(dependent_embedding, p=2) == 0
                                ):
                                    continue
                                with open(
                                    f'FD embedding_length_{model_name.replace("/", "")}.txt',
                                    "a",
                                ) as file:
                                    file.write(
                                        f"determinant_embedding length {torch.norm(determinant_embedding, p=2)}\n"
                                    )
                                    file.write(
                                        f"dependent_embedding length {torch.norm(dependent_embedding, p=2)}\n"
                                    )
                                # Calculate the L2 norm and add it to the list
                                l2_norm = torch.norm(
                                    determinant_embedding - dependent_embedding, p=2
                                )
                                l2_norms.append(
                                    l2_norm.item()
                                )  # Convert the torch tensor to a Python float

                                # Break the loop once we get a valid pair of embeddings
                                break
                            except (IndexError, AssertionError):
                                continue

                        if (
                            not l2_norms
                        ):  # If l2_norms is still empty after the loop, continue to the next row
                            continue

                    norms_dict[pair] = l2_norms

            list_pairs_norms_dict.append(norms_dict)
            # print(table.head())
            # # infer cell embeddings in determinant column and dependent column
            # break
        torch.save(
            list_pairs_norms_dict,
            os.path.join(save_directory, f"list_pairs_norms_dict.pt"),
        )

    if args.mode == "Both" or args.mode == "Non_FD":
        non_fd_metadata = data_loader.get_non_fd_metadata()
        list_non_pairs_norms_dict = []
        with open("Non_FD.txt", "w") as file:
            file.write("Record for list_row_index\n")
        with open(
            f'Non_FD embedding_length_{model_name.replace("/", "")}.txt', "w"
        ) as file:
            file.write("Record for embedding_length\n")
        for _, row in non_fd_metadata.iterrows():
            table_name = row["table_name"]
            col1 = row["column_1"]
            col2 = row["column_2"]

            table = data_loader.read_table(table_name)
            col1_index = list(table.columns).index(col1)
            col2_index = list(table.columns).index(col2)
            elements_dict = (
                table.reset_index().groupby(col1)["index"].apply(list).to_dict()
            )

            norms_dict = {}

            if model_name.startswith("doduo"):
                for element, list_row_index in elements_dict.items():
                    with open("Non_FD.txt", "a") as file:
                        file.write(str(element))
                        file.write("\n")
                        file.write(str(list_row_index))
                        file.write("\n")
                    l2_norms = []
                    for row_index in list_row_index:
                        # Start with a range of two rows above and below
                        for delta in [2, 1, 0]:
                            try:
                                # Create a smaller table based on the current row index and delta
                                min_index = max(0, row_index - delta)
                                max_index = min(len(table), row_index + delta + 1)
                                part_table = table.iloc[min_index:max_index]
                                part_table = part_table.astype(str)

                                # Adjust the index for the cell_embeddings, considering the boundaries
                                adjusted_index = min(delta, row_index - min_index)

                                tmp_pairs = []
                                tmp_pairs.append(
                                    [[adjusted_index, col1_index], (-1, "")]
                                )
                                tmp_pairs.append(
                                    [[adjusted_index, col2_index], (-1, "")]
                                )

                                # Get the entity_embeddings of the part table
                                try:
                                    entity_embeddings = get_embedding(
                                        part_table, tmp_pairs
                                    )
                                except ValueError as e:
                                    print(e)
                                    continue

                                # Get the col1_embedding and col2_embedding
                                col1_embedding, _ = entity_embeddings[
                                    (adjusted_index, col1_index)
                                ]
                                col2_embedding, _ = entity_embeddings[
                                    (adjusted_index, col2_index)
                                ]

                                # If either the col1_embedding or col2_embedding is 0, continue to next delta
                                if (
                                    torch.norm(col1_embedding, p=2) == 0
                                    or torch.norm(col2_embedding, p=2) == 0
                                ):
                                    continue
                                with open(
                                    f'Non_FD embedding_length_{model_name.replace("/", "")}.txt',
                                    "a",
                                ) as file:
                                    file.write(
                                        f"col1_embedding length {torch.norm(col1_embedding, p=2)}\n"
                                    )
                                    file.write(
                                        f"col2_embedding length {torch.norm(col2_embedding, p=2)}\n"
                                    )
                                # Calculate the L2 norm and add it to the list
                                l2_norm = torch.norm(
                                    col1_embedding - col2_embedding, p=2
                                )
                                l2_norms.append(
                                    l2_norm.item()
                                )  # Convert the torch tensor to a Python float

                                # Break the loop once we get a valid pair of embeddings
                                break
                            # except (KeyError, IndexError, AssertionError) as e:
                            except (KeyError, IndexError) as e:
                                print(e)
                                continue

                        if (
                            not l2_norms
                        ):  # If l2_norms is still empty after the loop, continue to the next row
                            continue

                    norms_dict[str(element)] = l2_norms

            else:
                for element, list_row_index in elements_dict.items():
                    with open("Non_FD.txt", "a") as file:
                        file.write(str(element))
                        file.write("\n")
                        file.write(str(list_row_index))
                        file.write("\n")
                    l2_norms = []
                    for row_index in list_row_index:
                        # Start with a range of two rows above and below
                        for delta in [2, 1, 0]:
                            try:
                                # Create a smaller table based on the current row index and delta
                                min_index = max(0, row_index - delta)
                                max_index = min(len(table), row_index + delta + 1)
                                part_table = table.iloc[min_index:max_index]
                                cell_embeddings = get_embedding(part_table)

                                # Adjust the index for the cell_embeddings, considering the boundaries
                                adjusted_index = min(delta, row_index - min_index) + 1

                                # Get the col1_embedding and col2_embedding
                                col1_embedding = cell_embeddings[adjusted_index][
                                    col1_index
                                ]
                                col2_embedding = cell_embeddings[adjusted_index][
                                    col2_index
                                ]

                                # If either the col1_embedding or col2_embedding is 0, continue to next delta
                                if (
                                    torch.norm(col1_embedding, p=2) == 0
                                    or torch.norm(col2_embedding, p=2) == 0
                                ):
                                    continue
                                with open(
                                    f'Non_FD embedding_length_{model_name.replace("/", "")}.txt',
                                    "a",
                                ) as file:
                                    file.write(
                                        f"col1_embedding length {torch.norm(col1_embedding, p=2)}\n"
                                    )
                                    file.write(
                                        f"col2_embedding length {torch.norm(col2_embedding, p=2)}\n"
                                    )
                                # Calculate the L2 norm and add it to the list
                                l2_norm = torch.norm(
                                    col1_embedding - col2_embedding, p=2
                                )
                                l2_norms.append(
                                    l2_norm.item()
                                )  # Convert the torch tensor to a Python float

                                # Break the loop once we get a valid pair of embeddings
                                break
                            except (IndexError, AssertionError):
                                continue

                        if (
                            not l2_norms
                        ):  # If l2_norms is still empty after the loop, continue to the next row
                            continue

                    norms_dict[element] = l2_norms
            list_non_pairs_norms_dict.append(norms_dict)

        torch.save(
            list_non_pairs_norms_dict,
            os.path.join(save_directory, f"list_non_pairs_norms_dict.pt"),
        )
