import itertools
import json
import os
import pickle

from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm

from observatory.models.TURL.data_loader.hybrid_data_loaders import WikiHybridTableDataset
from observatory.models.TURL.utils.util import load_entity_vocab


def build_entity_id_map(data_dir: str, min_ent_count: int, sample_size: int) -> Tuple[dict, dict]:
    """
    Build a dictionary that maps entity ID in Wikipedia to entity ID in TURL.

    Args:
        min_ent_count: consider only entities that appear at least 'min_ent_count' times.

    'entity_vocab' looks like
    {
        0: {'count': -1, 'mid': -1, 'wiki_id': '[PAD]', 'wiki_title': '[PAD]'},
        1: {'count': -1, 'mid': -1, 'wiki_id': '[ENT_MASK]', 'wiki_title': '[ENT_MASK]'},
        2: {'count': -1, 'mid': -1, 'wiki_id': '[PG_ENT_MASK]', 'wiki_title': '[PG_ENT_MASK]'},
        3: {'count': -1, 'mid': -1, 'wiki_id': '[CORE_ENT_MASK]', 'wiki_title': '[CORE_ENT_MASK]'},
        4: {'count': 17865, 'mid': 'm.09c7w0', 'wiki_id': 3434750, 'wiki_title': 'United_States'},
        ...
    }
    The first four IDs are reserved for special tokens in TURL and entity records start at ID 4.
    """

    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)

    entity_id_map = {int(entity_vocab[x]["wiki_id"]): x for x in entity_vocab if x <= sample_size + 3 and x >= 4}

    return entity_vocab, entity_id_map


def build_metadata_input_for_turl(page_title: str, section_title: str, caption: str, headers: List[str], config) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    tokenized_page_title = config.tokenizer.encode(
        page_title, max_length=config.max_title_length, add_special_tokens=False)
    
    tokenized_metadata = tokenized_page_title + \
        config.tokenizer.encode(
            section_title, max_length=config.max_title_length, add_special_tokens=False)

    if caption != section_title:
        tokenized_metadata += config.tokenizer.encode(
            caption, max_length=config.max_title_length, add_special_tokens=False)

    tokenized_headers = [config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False) for header in headers]
    
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_metadata_length = len(tokenized_metadata)
    input_tok += tokenized_metadata
    input_tok_pos += list(range(tokenized_metadata_length))
    input_tok_type += [0]*tokenized_metadata_length
    
    for tokenized_header in tokenized_headers:
        input_tok += tokenized_header
        tokenized_header_length = len(tokenized_header)
        input_tok_pos += list(range(tokenized_header_length))
        input_tok_type += [1]*tokenized_header_length

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])

    return input_tok, input_tok_type, input_tok_pos


def prepare_data_for_turl(data_dir: str, entity_vocab, entity_id_map, tokenizer, device):
    """
    Build a dictionary that maps entity ID in TURL to entity metadata from each table that contains the entity.

    More specifically, each entry in entity metadata is a list of inputs specifically for the TURL model.    
    """

    id2metadata = {}
    dataset = WikiHybridTableDataset(data_dir, entity_vocab, max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length=[50, 10, 10], force_new=False, tokenizer=tokenizer, mode=0)
    
    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        for table in tqdm(f):
            table = json.loads(table.strip())
            caption = table.get("tableCaption", "")
            page_title = table.get("page_title", "")
            section_title = table.get("sectionTitle", "")
            rows = table.get("tableData", [])
            headers = table.get("tableHeaders", [])[0]
            
            input_token, input_token_type, input_token_pos = build_metadata_input_for_turl(page_title, section_title, caption, headers, dataset)
            dup_id = set()
    
            for row in rows:
                for cell in row:
                    if len(cell['surfaceLinks']) > 0:
                        wiki_id = int(cell['surfaceLinks'][0]['target']['id'])
                    else:
                        continue
                    
                    if wiki_id not in entity_id_map:
                        continue
                    else:
                        id = entity_id_map[wiki_id]
                        if id in dup_id:
                            continue
                    
                    dup_id.add(id)
                    
                    if id not in id2metadata:
                        id2metadata[id] = []

                    id2metadata[id].append([input_token.to(device), input_token_type.to(device), input_token_pos.to(device)])
    return id2metadata


def prepare_data_for_lm(data_dir, entity_id_map) -> dict[int, List[List[str]]]:
    """
    Build a dictionary that maps entity ID in TURL to entity metadata from each table that contains the entity.

    More specifically, each entry in entity metadata is a list of two strings: entity text and a concatenation of table metadata which includes caption, page title, section title and headers.    
    """

    id2metadata = {}
    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        for table in tqdm(f):
            table = json.loads(table.strip())
            caption = table.get("tableCaption", "")
            page_title = table.get("pgTitle", "")
            section_title = table.get("sectionTitle", "")
            rows = table.get("tableData", [])
            headers = table.get("tableHeaders", [])[0]
            dup_id = set()

            for row in rows:
                for cell in row:
                    if len(cell['surfaceLinks']) > 0:
                        wiki_id = int(cell['surfaceLinks'][0]['target']['id'])
                        entity_text = cell['surfaceLinks'][0]['target']['title']
                    else:
                        continue
                    
                    if wiki_id not in entity_id_map:
                        continue
                    else:
                        id = entity_id_map[wiki_id]
                        if id in dup_id:
                            continue

                    dup_id.add(id)
                    if id not in id2metadata:
                        id2metadata[id] = []
                    id2metadata[id].append([entity_text, ' '.join([caption, page_title, section_title] + headers)])
    return id2metadata


def process_single_hybrid_table(input_table, config):
    headers, entities, entities_text = input_table
    
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for _, z in headers]

    input_tokens = []
    input_tokens_pos = []
    input_tokens_type = []

    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tokens += list(itertools.chain(*tokenized_headers))
    input_tokens_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
    input_tokens_type += [1] * sum(tokenized_headers_length)

    input_entities = [entity for _, entity in entities]
    input_entities_text = [config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False) if len(entity_text) != 0 else [] for entity_text in entities_text]
    
    input_entities_cell_length = [len(x) if len(x) != 0 else 1 for x in input_entities_text]
    max_cell_length = max(input_entities_cell_length)
    input_entities_text_padded = np.zeros([len(input_entities_text), max_cell_length], dtype=int)
    for i, x in enumerate(input_entities_text):
        input_entities_text_padded[i, :len(x)] = x

    # input_tokens = torch.LongTensor([input_tokens])
    # input_tokens_type = torch.LongTensor([input_tokens_type])
    # input_tokens_pos = torch.LongTensor([input_tokens_pos])

    # input_entities = torch.LongTensor([input_entities])
    # input_entities_text = torch.LongTensor([input_entities_text_padded])
    # input_entities_cell_length = torch.LongTensor([input_entities_cell_length])
    # input_entities_type = torch.LongTensor([input_entities_type])

    return [np.array(input_tokens), np.array(input_tokens_type), np.array(input_tokens_pos), len(input_tokens), np.array(input_entities), input_entities_text_padded, input_entities_cell_length, len(input_entities)]


class TurlWikiDataset(Dataset):
    def __init__(self, data_dir: str, entity_vocab: dict, split: str, tokenizer=None, max_cells: int = 100, max_input_tokens: int = 350, max_input_entities: int = 150, max_header_length: int = 10, max_cell_length: int = 10, force_new: bool = False):
        self.data_dir = data_dir
        self.split = split
        self.entity_vocab = entity_vocab
        self.entity_id_map = {entity_vocab[x]["wiki_id"]: x for x in self.entity_vocab} # A dictionary that maps entity ID in Wikipedia to entity ID in TURL.
        self.tokenizer = tokenizer
        self.max_cells = max_cells
        self.max_input_tokens = max_input_tokens
        self.max_input_entities = max_input_entities
        self.max_header_length = max_header_length
        self.max_cell_length = max_cell_length
        self.force_new = force_new
        self.data = self._preprocess_data()
    
    def _preprocess_data(self):
        preprocessed_filename = os.path.join(self.data_dir, f"preprocessed_hybrid", f"{self.split}.pickle")

        if not self.force_new and os.path.exists(preprocessed_filename):
            print("Try loading preprocessed data from: ", preprocessed_filename)
            with open(preprocessed_filename, "rb") as f:
                return pickle.load(f)

        print("Try creating preprocessed data in: ", preprocessed_filename)
        if not os.path.exists(os.path.join(data_dir, "procressed_hybrid")):
            os.mkdir(os.path.join(data_dir, "procressed_hybrid"))

        num_orig_tables = 0
        actual_tables = []

        with open(os.path.join(data_dir, f"{self.split}_tables.jsonl"), "r") as f:
            for line in tqdm(f):
                num_orig_tables += 1
                table = json.loads(line.strip())
                headers = table.get("processed_tableHeaders", []) 
                rows = table.get("tableData", [])
                entity_columns = table.get("entityColumn", []) # Index of entity columns, e.g., [0, 1, 2]
                entity_cells = np.array(table.get("entityCell",[[]])) # Whether a cell is an entity cell, e.g., [[1, 1, 1, 0, 0], ...] 

                # Headers for entity columns
                headers = [[i, headers[i]] for i in entity_columns]

                # Collect entities and entity mentions
                num_rows = len(rows)
                num_cols = len(rows[0])
                entities, entities_text = [], []
                split = [0]
                tmp_entity_num = 0

                for i in range(num_rows):
                    tmp_entities, tmp_entities_text = [], []

                    for j in range(num_cols):
                        if j in entity_columns:
                            if entity_cells[i][j] == 1:
                                try:
                                    entity = self.entity_id_map[rows[i][j]["surfaceLinks"][0]["target"]["id"]]
                                    entity_cells[i][j] = entity
                                    tmp_entities.append([(i,j), entity])
                                    tmp_entities_text.append(rows[i][j]["text"])
                                except:
                                    entity_cells[i][j] = 0
                        else:
                            entity_cells[i][j] = 0
                    
                    if len(tmp_entities) == 0: continue

                    for (index, entity), entity_text in zip(tmp_entities, tmp_entities_text):
                        entities.append([index, entity])
                        entities_text.append(entity_text)
                        tmp_entity_num += 1

                    if tmp_entity_num >= self.max_cells:
                        split.append(len(entities))
                        tmp_entity_num = 0
            
                if split[-1] != len(entities):
                    split.append(len(entities))
                
                # if split[-2] != 0 and split[-1] - split[-2] < 5:
                #     split[-2] = split[-1]
                #     split = split[:-1]
                
                for i in range(len(split) - 1):
                    actual_tables.append([
                        headers,
                        entities[split[i]:split[i+1]],
                        entities_text[split[i]:split[i+1]]
                    ])
        
        num_actual_tables = len(actual_tables)
        print("=" * 50)
        print("Number of original tables: ", num_orig_tables)
        print("Number of actual tables: ", num_actual_tables)
        print(actual_tables[0])
        print("=" * 50)

        pool = Pool(processes=4)
        processed_data = list(tqdm(
            pool.imap(
                partial(process_single_hybrid_table, config=self), actual_tables, chunksize=2000
                ), total=len(actual_tables)
            ))
        pool.close()

        with open(preprocessed_filename, "wb") as f:
            pickle.dump(processed_data, f)

        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def prepare_inputs_for_column_representations(self):
        pass
    


if __name__ == "__main__":
    """
    The following code is only for development/test purposes.
    Assume it is run from the 'datasets' directory.
    """
    from pprint import pprint

    # Check working directory assumption
    # working_dir = os.path.abspath("./")
    # assert(working_dir.split("/")[-1] == "datasets")

    # project_root_dir = os.path.abspath("../../")
    # data_dir = os.path.join(project_root_dir, "data/")

    # min_ent_count = 2
    # sample_size = 1000
    
    # entity_vocab, entity_id_map = build_entity_id_map(data_dir, min_ent_count, sample_size)

    # pprint(entity_vocab)
    # pprint(entity_id_map)

    data_dir = "/home/congtj/observatory/data/"
    min_ent_count = 2

    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)

    test_data_loader = TurlWikiDataset(data_dir, entity_vocab, split="test")
    print(len(test_data_loader))