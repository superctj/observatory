import argparse
import json
import os

import numpy as np
import pandas as pd

from doduo.doduo import Doduo


def read_jsonl(file_path):
    data = []
    count = 0
    with open(file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            data.append(json.loads(line))
            count += 1
            if count >= 20:
                break
    return data

def process_one_table(data, entity_id_map):
    processed_table_headers = data['processed_tableHeaders']
    table_data = data['tableData']
    formatted_table_data = [[cell['text'] for cell in row] for row in table_data]
    df = pd.DataFrame(formatted_table_data, columns=processed_table_headers)
    
    rows = data.get("tableData", [])
    num_rows = len(rows)
    num_cols = len(rows[0])
    entity_cells = np.array(data.get("entityCell",[[]]))
    entity_columns = data.get("entityColumn", [])
    tmp_entities = []
    for i in range(num_rows):
        ######################
        for j in range(min(num_cols, 10)):
            if j in entity_columns:
                if entity_cells[i][j] == 1:
                    try:
                        entity_id = entity_id_map[rows[i][j]["surfaceLinks"][0]["target"]["id"]]
                        entity_text = rows[i][j]["text"]
                        tmp_entities.append([[i, j], (entity_id, entity_text)])
                    except:
                        entity_cells[i][j] = 0
            else:
                entity_cells[i][j] = 0
    return df, tmp_entities

def test_entity_embedding_retrival(model):
    from observatory.models.TURL.utils.util import load_entity_vocab

    data_dir = "/home/congtj/observatory/data"
    min_ent_count = 2
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)
    entity_id_map = {entity_vocab[x]["wiki_id"]: x for x in entity_vocab}

    testset_filepath = os.path.join(data_dir, "test_tables.jsonl")
    all_tables = read_jsonl(testset_filepath)

    for table in all_tables:
        df, entity_info = process_one_table(table, entity_id_map)
        try:
            entity_embeddings = model.get_entity_embeddings(df, entity_info)
        except ValueError as e:
            print(e)

        # for entity_cell_pos in entity_embeddings:
        #     print(entity_cell_pos)
        #     print(entity_embeddings[entity_cell_pos][1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="wikitable",
        type=str,
        choices=["wikitable", "viznet"],
        help="Pretrained model"
    )
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="Input file (csv)"
    )
    args = parser.parse_args()

    # if args.input is None:
    #     # Sample table
    #     input_df = pd.read_csv(
    #         "sample_tables/table_4702.csv")
    # else:
    #     input_df = pd.read_csv(args.input)

    # print(input_df.shape)
    # print(input_df.columns)

    doduo = Doduo(args, basedir="/ssd/congtj/observatory/doduo")
    # annotated_df = doduo.annotate_columns(input_df)
    test_entity_embedding_retrival(doduo)
