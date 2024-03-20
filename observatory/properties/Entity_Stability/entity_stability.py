import os
import pandas as pd
import json
import numpy as np
import sys
import argparse
import functools
import torch
from observatory.models.TURL.utils.util import load_entity_vocab
from observatory.models.hugging_face_cell_embeddings import (
    get_hugging_face_cell_embeddings,
)
from observatory.models.turl import get_entity_embeddings_example
from observatory.models.doduo_entity_embeddings import Doduo


def load_and_process_data(
    get_embedding: callable,
    save_directory: str,
    start: int,
    file_path: str,
    data_dir: str,
    ifentity_info: bool = False,
    min_ent_count: int = 2,
) -> None:
    """ Load and process the data from the given directory, and save the embeddings to the given directory.
    
    Args:
        get_embedding: A callable that takes a pandas DataFrame and returns a torch.Tensor of embeddings.
        save_directory: The directory to save the embeddings to.
        start: The line number to start processing from.
        file_path: The path to the jsonl file of the turl dataset.
        data_dir: The data directory for the Turl dataset.
        ifentity_info: A boolean indicating whether to save entity embeddings.
        min_ent_count: The minimum number of times an entity must appear in the dataset to be included in the vocabulary.
    
    Returns:
        None
    """
    # load entity vocabulary
    entity_vocab = load_entity_vocab(
        data_dir, ignore_bad_title=True, min_ent_count=min_ent_count
    )
    entity_id_map = {entity_vocab[x]["wiki_id"]: x for x in entity_vocab}

    # loop through all files in the directory
    # for filename in os.listdir(data_dir):
        # if filename.endswith(".jsonl"):
            # file_path = os.path.join(data_dir, filename)
    data = read_jsonl(file_path)
    count = 0
    # process each table in the file
    save_directory_cell = os.path.join(
        save_directory, "original_cell_embeddings"
    )
    save_directory_entity = os.path.join(save_directory, "entity_embeddings")
    if not ifentity_info:
        if not os.path.exists(save_directory_cell):
            os.makedirs(save_directory_cell)
    if not os.path.exists(save_directory_entity):
        os.makedirs(save_directory_entity)
    for table_data in data:
        df, tmp_entities = process_one_table(table_data, entity_id_map)
        count += 1
        if count < start:
            continue
        if ifentity_info:
            try:
                entity_embeddings = get_embedding(df, tmp_entities)
                torch.save(
                    entity_embeddings,
                    os.path.join(
                        save_directory_entity,
                        f"line{count}_entity_embeddings.pt",
                    ),
                )
            except ValueError as e:
                print(e)
        else:

            try:
                cell_embeddings = get_embedding(df)
            except Exception as e:
                print(e)
                continue
            torch.save(
                cell_embeddings,
                os.path.join(
                    save_directory_cell, f"line{count}_cell_embeddings.pt"
                ),
            )
            entity_embeddings = {}
            for index, entity in tmp_entities:
                try:
                    i, j = index
                    entity_embeddings[(i, j)] = (
                        cell_embeddings[i + 1][j],
                        entity[0],
                    )
                except IndexError:
                    continue
            torch.save(
                entity_embeddings,
                os.path.join(
                    save_directory_entity, f"line{count}_entity_embeddings.pt"
                ),
            )

                # process df and tmp_entities as you want
                # ...


def read_jsonl(
    file_path: str,
) -> list:
    """ Read a jsonl file and return a list of dictionaries.
    
    Args:
        file_path: The path to the jsonl file.
    
    Returns:
        A list of dictionaries.
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def process_one_table(
    # data, entity_id_map):
    data: dict, 
    entity_id_map: dict
) -> pd.DataFrame, list:
    """ Process a single table and return a DataFrame and a list of entities.
    
    Args:
        data: A dictionary containing the table data.
        entity_id_map: A dictionary mapping entity IDs to entity names.
    
    Returns:
        A tuple containing a DataFrame and a list of entities.
    """
    processed_table_headers = data["processed_tableHeaders"]
    table_data = data["tableData"]
    formatted_table_data = [[cell["text"] for cell in row] for row in table_data]
    df = pd.DataFrame(formatted_table_data, columns=processed_table_headers)

    rows = data.get("tableData", [])
    num_rows = len(rows)
    num_cols = len(rows[0])
    entity_cells = np.array(data.get("entityCell", [[]]))
    entity_columns = data.get("entityColumn", [])
    tmp_entities = []
    for i in range(num_rows):
        for j in range(min(num_cols, 10)):
            if j in entity_columns:
                if entity_cells[i][j] == 1:
                    try:
                        entity_id = entity_id_map[
                            rows[i][j]["surfaceLinks"][0]["target"]["id"]
                        ]
                        entity_text = rows[i][j]["text"]
                        tmp_entities.append([[i, j], (entity_id, entity_text)])
                    except:
                        entity_cells[i][j] = 0
            else:
                entity_cells[i][j] = 0
    return df, tmp_entities


if __name__ == "__main__":
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
        "--data_dir",
        type=str,
        default=".",
        help="Data directory for Turl dataset",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=".",
        help="Path to the jsonl file of the turl dataset",
    )
    parser.add_argument("--start", type=int, default=0, help="Start Line")
    parser.add_argument(
        "--doduo_path",
        type=str,
        default=".",
        help="Path to load the doduo model",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=".",
        help="Config for Turl model",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=".",
        help="CheckPoint file path for Turl model",
    )
    args = parser.parse_args()
    start = args.start
    model_name = args.model_name
    save_directory = os.path.join(args.save_dir, "Entity_Stability", model_name)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    if (
        model_name.startswith("bert")
        or model_name.startswith("roberta")
        or model_name.startswith("google/tapas")
        or model_name.startswith("t5")
    ):
        get_embedding = functools.partial(
            get_hugging_face_cell_embeddings, model_name=model_name
        )
        load_and_process_data(get_embedding, save_directory, start, args.file_path, args.data_dir)
    elif model_name.startswith("turl"):
        all_entity_embeddings = get_entity_embeddings_example(
            args.data_dir, args.config, args.ckpt_path
        )
        save_directory_entity = os.path.join(save_directory, "entity_embeddings")
        if not os.path.exists(save_directory_entity):
            os.makedirs(save_directory_entity)
        for count, embedding_dict in all_entity_embeddings:
            torch.save(
                embedding_dict,
                os.path.join(
                    save_directory_entity, f"line{count}_entity_embeddings.pt"
                ),
            )
    elif model_name.startswith("doduo"):
        model_args = argparse.Namespace

        model_args.model = "wikitable"
        doduo = Doduo(model_args, basedir=args.doduo_path)

        get_embedding = doduo.get_entity_embeddings
        load_and_process_data(get_embedding, save_directory, start, args.file_path, args.data_dir, ifentity_info=True)
