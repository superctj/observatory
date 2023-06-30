
from TURL.utils.util import load_entity_vocab

min_ent_count = 2
entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)
entity_id_map = {entity_vocab[x]["wiki_id"]: x for x in entity_vocab}

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
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