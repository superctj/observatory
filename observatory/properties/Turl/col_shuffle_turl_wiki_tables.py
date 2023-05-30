import itertools
import json
import os
import pickle
import random

from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm.autonotebook import tqdm

from  TURL.data_loader.hybrid_data_loaders import WikiHybridTableDataset


# def build_entity_id_map(data_dir: str, min_ent_count: int, sample_size: int) -> Tuple[dict, dict]:
#     """
#     Build a dictionary that maps entity ID in Wikipedia to entity ID in TURL.

#     Args:
#         min_ent_count: consider only entities that appear at least 'min_ent_count' times.

#     'entity_vocab' looks like
#     {
#         0: {'count': -1, 'mid': -1, 'wiki_id': '[PAD]', 'wiki_title': '[PAD]'},
#         1: {'count': -1, 'mid': -1, 'wiki_id': '[ENT_MASK]', 'wiki_title': '[ENT_MASK]'},
#         2: {'count': -1, 'mid': -1, 'wiki_id': '[PG_ENT_MASK]', 'wiki_title': '[PG_ENT_MASK]'},
#         3: {'count': -1, 'mid': -1, 'wiki_id': '[CORE_ENT_MASK]', 'wiki_title': '[CORE_ENT_MASK]'},
#         4: {'count': 17865, 'mid': 'm.09c7w0', 'wiki_id': 3434750, 'wiki_title': 'United_States'},
#         ...
#     }
#     The first four IDs are reserved for special tokens in TURL and entity records start at ID 4.
#     """

#     entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)

#     entity_id_map = {int(entity_vocab[x]["wiki_id"]): x for x in entity_vocab if x <= sample_size + 3 and x >= 4}

#     return entity_vocab, entity_id_map
def fisher_yates_shuffle(seq):
    for i in reversed(range(1, len(seq))):
        j = random.randint(0, i)
        seq[i], seq[j] = seq[j], seq[i]
    return seq

def get_permutations(n, m):
    if n < 10:
        # Generate all permutations
        all_perms = list(itertools.permutations(range(n)))
        # Remove the original sequence
        all_perms.remove(tuple(range(n)))
        # Shuffle the permutations
        random.shuffle(all_perms)
        # If m > n! - 1 (because we removed one permutation), return all permutations
        if m > len(all_perms):
            return all_perms
        # Otherwise, return the first m permutations
        return all_perms[:m]
    else:
        original_seq = list(range(n))
        perms = [original_seq.copy()]
        for _ in range(m):  # we already have one permutation
            while True:
                new_perm = fisher_yates_shuffle(original_seq.copy())
                if new_perm not in perms:
                    perms.append(new_perm)
                    break
        perms.remove(list(range(n)))
        return perms

def shuffle_lists(lists, perms):
    # Add the original list to the shuffled lists
    shuffled_lists = [lists.copy()] 
    # Apply permutations to lists
    for perm in perms:
        shuffled_list = [[row[i] for i in perm] for row in lists]
        shuffled_lists.append(shuffled_list)
    return shuffled_lists


def shuffle_elements(elements, perms):
    # Add the original list to the shuffled elements
    shuffled_elements = [elements.copy()] 
    # Apply permutations to elements
    for perm in perms:
        shuffled_element = [elements[i] for i in perm]
        shuffled_elements.append(shuffled_element)
    return shuffled_elements

def get_new_indices(indices, perm):
    # Create a mapping from old indices to new indices
    index_mapping = {old: new for new, old in enumerate(perm)}
    # Map the original indices to the new indices
    new_indices = [index_mapping[index] for index in indices]
    return new_indices


def shuffle_indices(indices, perms):
    # Add the original list to the shuffled indices
    shuffled_indices = [indices.copy()]
    # Apply permutations to indices
    for perm in perms:
        new_indices = get_new_indices(indices, perm)
        shuffled_indices.append(new_indices)
    return shuffled_indices

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


def process_single_table_for_columns(input_table, config):
    table_id, pg_title, pg_entity, sec_title, caption, headers, entities = input_table

    # Do not take in any metadata except headers as inputs
    pg_title = ""
    pg_entity = -1 
    sec_title = ""
    caption = ""

    tokenized_pg_title = config.tokenizer.encode(pg_title, max_length=config.max_title_length, add_special_tokens=False)
    tokenized_meta = tokenized_pg_title + \
                    config.tokenizer.encode(sec_title, max_length=config.max_title_length, add_special_tokens=False)
    
    if caption != sec_title:
        tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
    
    tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for z in headers]

    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0] * tokenized_meta_length
    tokenized_headers_length = [len(z) for z in tokenized_headers]
    input_tok += list(itertools.chain(*tokenized_headers))
    input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
    input_tok_type += [1] * sum(tokenized_headers_length)

    input_ent = []
    input_ent_text = []
    input_ent_type = []
    column_en_map = {}
    row_en_map = {}

    # index refers to (row_index, column_index) of an entity in the table
    for e_i, (index, (entity_id, entity_text)) in enumerate(entities):
        tokenized_entity_text = config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False)
        
        input_ent.append(entity_id)
        input_ent_text.append(tokenized_entity_text)
        input_ent_type.append(4)

        if index[0] not in row_en_map:
            row_en_map[index[0]] = [e_i]
        else:
            row_en_map[index[0]].append(e_i)

        if index[1] not in column_en_map:
            column_en_map[index[1]] = [e_i]
        else:
            column_en_map[index[1]].append(e_i)
    
    # create column entity mask
    column_entity_mask = np.zeros([len(headers), len(input_ent)], dtype=int)

    # print("=" * 50)
    # print("Length of headers: ", len(headers))
    # print("Length of column entity map: ", len(column_en_map))
    # print("=" * 50)

    for j in range(len(headers)):
        for e_i_1 in column_en_map[j]:
            column_entity_mask[j, e_i_1] = 1
    
    # create column header mask
    start_i = 0
    header_span = {}
    column_header_mask = np.zeros([len(headers), len(input_tok)], dtype=int)
    for j in range(len(headers)):
        header_span[j] = (start_i, start_i+tokenized_headers_length[j])
        column_header_mask[j, tokenized_meta_length+header_span[j][0]:tokenized_meta_length+header_span[j][1]] = 1
        start_i += tokenized_headers_length[j]

    # create input mask
    tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
    meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
    header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
    
    for e_i, (index, _) in enumerate(entities):
        header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    ent_header_mask = np.transpose(header_ent_mask)

    input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
    ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)
    
    ent_ent_mask = np.eye(len(input_ent), dtype=int)
    for _, e_is in column_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    for _, e_is in row_en_map.items():
        for e_i_1 in e_is:
            for e_i_2 in e_is:
                ent_ent_mask[e_i_1, e_i_2] = 1
    input_ent_mask = [np.concatenate([ent_meta_mask, ent_header_mask], axis=1), ent_ent_mask]

    # prepend pgEnt to input_ent, input_ent[0] = pgEnt
    if pg_entity != -1:
        input_tok_mask[1] = np.concatenate([np.ones([len(input_tok), 1], dtype=int), input_tok_mask[1]], axis=1)
    else:
        input_tok_mask[1] = np.concatenate([np.zeros([len(input_tok), 1], dtype=int), input_tok_mask[1]], axis=1)

    input_ent = [pg_entity if pg_entity != -1 else 0] + input_ent
    input_ent_text = [tokenized_pg_title[:config.max_cell_length]] + input_ent_text
    input_ent_type = [2] + input_ent_type

    new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int), np.ones([len(input_ent), len(input_ent)], dtype=int)]
    new_input_ent_mask[0][1:, :] = input_ent_mask[0]
    new_input_ent_mask[1][1:, 1:] = input_ent_mask[1]
    if pg_entity == -1:
        new_input_ent_mask[1][:, 0] = 0
        new_input_ent_mask[1][0, :] = 0
    column_entity_mask = np.concatenate([np.zeros([len(headers), 1], dtype=int),column_entity_mask],axis=1)
    input_ent_mask = new_input_ent_mask

    labels = np.zeros([len(headers), 100], dtype=int) # dummy variable
    # for j, types in enumerate(type_annotations):
    #     for t in types:
    #         labels[j, config.type_vocab[t]] = 1
    input_ent_cell_length = [len(x) if len(x) != 0 else 1 for x in input_ent_text]
    max_cell_length = max(input_ent_cell_length)
    input_ent_text_padded = np.zeros([len(input_ent_text), max_cell_length], dtype=int)
    for i, x in enumerate(input_ent_text):
        input_ent_text_padded[i, :len(x)] = x

    return [table_id, np.array(input_tok), np.array(input_tok_type), np.array(input_tok_pos), (np.array(input_tok_mask[0]), np.array(input_tok_mask[1])), len(input_tok), \
            np.array(input_ent), input_ent_text_padded, input_ent_cell_length, np.array(input_ent_type), (np.array(input_ent_mask[0]), np.array(input_ent_mask[1])), len(input_ent), \
            column_header_mask, column_entity_mask, labels, len(labels)]

    # headers, entities = input_table

    # tokenized_headers = [config.tokenizer.encode(z, max_length=config.max_header_length, add_special_tokens=False) for z in headers]
    # tokenized_headers_length = [len(z) for z in tokenized_headers]

    # input_tokens, input_tokens_pos, input_tokens_type = [], [], []
    # input_tokens += list(itertools.chain(*tokenized_headers))
    # input_tokens_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
    # input_tokens_type += [1] * sum(tokenized_headers_length)

    # input_entities, input_entities_text, input_entities_type = [], [], []
    # row_entity_map, column_entity_map = {}, {}

    # # index refers to (row_index, column_index) of an entity in the table
    # for e_i, (index, (entity_id, entity_text)) in enumerate(entities):
    #     tokenized_entity_text = config.tokenizer.encode(entity_text, max_length=config.max_cell_length, add_special_tokens=False)
        
    #     input_entities.append(entity_id)
    #     input_entities_text.append(tokenized_entity_text)
    #     input_entities_type.append(4)

    #     if index[0] not in row_entity_map:
    #         row_entity_map[index[0]] = [e_i]
    #     else:
    #         row_entity_map[index[0]].append(e_i)

    #     if index[1] not in column_entity_map:
    #         column_entity_map[index[1]] = [e_i]
    #     else:
    #         column_entity_map[index[1]].append(e_i)
    
    # # create column entity mask
    # column_entity_mask = np.zeros([len(headers), len(input_entities)], dtype=int)
    
    # for j in range(len(headers)):
    #     for e_i in column_entity_map[j]:
    #         column_entity_mask[j, e_i] = 1
    
    # # create column header mask
    # start_i = 0
    # header_span = {}
    # column_header_mask = np.zeros([len(headers), len(input_tokens)], dtype=int)
    
    # for j in range(len(headers)):
    #     header_span[j] = (start_i, start_i + tokenized_headers_length[j])
    #     column_header_mask[j, header_span[j][0]:header_span[j][1]] = 1
    #     start_i += tokenized_headers_length[j]
    
    # # create input mask
    # tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
    # meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
    # header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
    
    # for e_i, (index, _) in enumerate(entities):
    #     header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
    # ent_header_mask = np.transpose(header_ent_mask)

    # input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
    # ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)

    # return np.array(input_tokens), np.array(input_tokens_type), np.array(input_tokens_pos), len(input_tokens), np.array(input_entities), input_entities_text_padded, input_entities_cell_length, len(input_entities)


class TurlWikiTableDataset(Dataset):
    def __init__(self, line, entity_vocab: dict, tokenizer, split: str, max_columns: int = 10, max_cells: int = 100, max_input_tokens: int = 350, max_input_entities: int = 150, max_title_length: int = 50, max_header_length: int = 10, max_cell_length: int = 10, force_new: bool = False, num_change: int =1000):
        # self.data_dir = data_dir
        self.entity_vocab = entity_vocab
        self.entity_id_map = {entity_vocab[x]["wiki_id"]: x for x in self.entity_vocab} # A dictionary that maps entity ID in Wikipedia to entity ID in TURL.
        self.tokenizer = tokenizer
        self.split = split
        self.max_columns = max_columns
        self.max_cells = max_cells
        self.max_input_tokens = max_input_tokens
        self.max_input_entities = max_input_entities 
        self.max_title_length = max_title_length
        self.max_header_length = max_header_length
        self.max_cell_length = max_cell_length
        self.force_new = force_new
        self.num_change = num_change 
        self.line = line
        self.data, self.perms,  self.shuffled_entity_columns= self._preprocess_data()
        
    
    def _preprocess_data(self):
        # preprocessed_filename = os.path.join(self.data_dir, f"preprocessed_hybrid", f"{self.split}.pickle")

        # if not self.force_new and os.path.exists(preprocessed_filename):
        #     print("Try loading preprocessed data from: ", preprocessed_filename)
        #     with open(preprocessed_filename, "rb") as f:
        #         return pickle.load(f)

        # print("Try creating preprocessed data in: ", preprocessed_filename)
        # base_dir = os.path.dirname(preprocessed_filename)
        # if not os.path.exists(base_dir):
        #     os.makedirs(base_dir)

        num_orig_tables = 0
        actual_tables = []

        line = self.line
        table = json.loads(line.strip())
        original_rows = table.get("tableData", [])
        original_headers = table.get("processed_tableHeaders", []) 
        original_entity_columns = table.get("entityColumn", []) # Index of entity columns, e.g., [0, 1, 3]
        original_entity_cells = np.array(table.get("entityCell",[[]])) # Whether a cell is an entity cell, e.g., [[1, 1, 0, 1, 0], ...]
        
        
        real_entity_cells = original_entity_cells.copy()

        num_rows = len(real_entity_cells)
        num_cols = len(real_entity_cells[0])

        for i in range(num_rows):
            for j in range(min(num_cols, self.max_columns)):
                if j in original_entity_columns:
                    if real_entity_cells[i][j] == 1:
                        try:
                            entity_id = self.entity_id_map[original_rows[i][j]["surfaceLinks"][0]["target"]["id"]]
                            entity_text = original_rows[i][j]["text"]

                        except:
                            real_entity_cells[i][j] = 0
                else:
                    real_entity_cells[i][j] = 0
            
        
        real_entity_columns = [col for col in original_entity_columns if any(real_entity_cells[i][col] for i in range(len(real_entity_cells)))]
        entity_cells = real_entity_cells[:, :self.max_columns]
        actual_entity_columns = [i for i, num_entities in enumerate(entity_cells.sum(axis=0)) if num_entities > 0]
        
        print("=" * 50)
        print(original_rows[0][0]['text'])
        print("=" * 50)
        # Get permutations
        primary_columns = min(len(original_rows[0]), self.max_columns)
        primary_permutations = get_permutations(primary_columns, self.num_change)
        if primary_columns == len(original_rows[0]):
            perms = primary_permutations
        else:
            rest_permutation = list(range(primary_columns, len(original_rows[0]))) 
            perms = [perm + rest_permutation for perm in primary_permutations]
        list_entity_cells = shuffle_lists(original_entity_cells, perms)
        list_rows = shuffle_lists(original_rows, perms)
        shuffled_headers = shuffle_elements(original_headers, perms)
        shuffled_entity_columns = shuffle_indices(actual_entity_columns, perms)
        # Collect entities (entity ids in TURL and entity mentions)
        for entity_cells, rows, headers, entity_columns  in zip(list_entity_cells, list_rows, shuffled_headers, shuffled_entity_columns):
            entity_cells = np.array(entity_cells)
            num_orig_tables += 1
            table_id = table.get("_id","")
            pg_title = table.get("pgTitle", "").lower()
            pg_entity = table.get("pgId", -1)
            
            if pg_entity != -1:
                try:
                    pg_entity = self.entity_id_map[pg_entity]
                except:
                    pg_entity = -1

            sec_title = table.get("sectionTitle", "").lower()
            caption = table.get("tableCaption", "").lower()
            
            # headers = original_headers.copy()
            # entity_columns = original_entity_columns.copy()
            
            # if num == 0:
            #     rows = original_rows.copy()
            # else:
            #     # print(original_rows)
            #     # print("\n\n\n")
            #     rows = original_rows.copy()
            #     random.shuffle(rows)
            #     # print(rows)

            
            num_rows = len(rows)
            num_cols = len(rows[0])
            entities = []
            # split = [0]
            tmp_entity_num = 0

            for i in range(num_rows):
                tmp_entities = []

                for j in range(min(num_cols, self.max_columns)):
                    if j in entity_columns:
                        if entity_cells[i][j] == 1:
                            try:
                                entity_id = self.entity_id_map[rows[i][j]["surfaceLinks"][0]["target"]["id"]]
                                entity_text = rows[i][j]["text"]
                                tmp_entities.append([[i, j], (entity_id, entity_text)])
                            except:
                                entity_cells[i][j] = 0
                    else:
                        entity_cells[i][j] = 0
                
                if len(tmp_entities) == 0: continue

                for index, entity in tmp_entities:
                    entities.append([index, entity])
                    tmp_entity_num += 1

                # if tmp_entity_num >= self.max_cells:
                #     split.append(len(entities))
                #     tmp_entity_num = 0
        
            # if split[-1] != len(entities):
            #     split.append(len(entities))
            
            # if split[-2] != 0 and split[-1] - split[-2] < 5:
            #     split[-2] = split[-1]
            #     split = split[:-1]

            """
            If no entity cell is found under a header, this header needs to be removed

            Entity column indices also need to be remapped for creating masks later, e.g., [2, 4, 5] --> [0, 1, 2]

            Also check for empty tables with no entity texts
            """
            entity_cells = entity_cells[:, :self.max_columns]
            actual_entity_columns = [i for i, num_entities in enumerate(entity_cells.sum(axis=0)) if num_entities > 0]
            headers = [headers[i] for i in actual_entity_columns] # Headers for entity columns that actually contain entities
            header_idx_map = {old_index: new_idx for new_idx, old_index in enumerate(actual_entity_columns)}

            empty_table = True
            for e_i, (idx, _) in enumerate(entities):
                # print("=" * 30)
                # print(entity_columns)
                # print(actual_entity_columns)
                # print(idx)
                entities[e_i][0][1] = header_idx_map[idx[1]]

                if len(entities[e_i][1][1]) > 0:
                    empty_table = False
                # print(entities[e_i][0])
                # print("=" * 30)
            assert(len(headers) == len(header_idx_map))

            # for i in range(len(split) - 1):
            #     actual_tables.append([
            #         table_id,
            #         pg_title,
            #         pg_entity,
            #         sec_title,
            #         caption,
            #         headers,
            #         entities[split[i]:split[i+1]]])

            if not empty_table:
                actual_tables.append([
                    table_id,
                    pg_title,
                    pg_entity,
                    sec_title,
                    caption,
                    headers,
                    entities
                ])

        num_actual_tables = len(actual_tables)
        print("=" * 50)
        print("Number of original tables: ", num_orig_tables)
        print("Number of actual tables: ", num_actual_tables)
        # print(actual_tables[0])
        # print("-" * 30)
        # print(actual_tables[2])
        print("=" * 50)

        pool = Pool(processes=4)
        processed_data = list(tqdm(
            pool.imap(
                partial(process_single_table_for_columns, config=self), actual_tables, chunksize=2000), total=len(actual_tables)
            ))
        pool.close()

        # with open(preprocessed_filename, "wb") as f:
        #     pickle.dump(processed_data, f)

        return processed_data, [list(range(len(original_rows[0])))] + perms, shuffled_entity_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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

    from TURL.model.transformers import BertTokenizer
    from TURL.utils.util import load_entity_vocab

    data_dir = "../"
    min_ent_count = 2

    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=min_ent_count)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_dataset = TurlWikiTableDataset(data_dir, entity_vocab, tokenizer, split="test")
    print(len(test_dataset))
