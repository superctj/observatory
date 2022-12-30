import json
import os

from typing import List, Tuple

import torch

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


if __name__ == "__main__":
    """
    The following code is only for development/test purposes.
    Assume it is run from the 'datasets' directory.
    """
    from pprint import pprint

    # Check working directory assumption
    working_dir = os.path.abspath("./")
    assert(working_dir.split("/")[-1] == "datasets")

    project_root_dir = os.path.abspath("../../")
    data_dir = os.path.join(project_root_dir, "data/")

    min_ent_count = 2
    sample_size = 1000
    
    entity_vocab, entity_id_map = build_entity_id_map(data_dir, min_ent_count, sample_size)

    pprint(entity_vocab)
    # pprint(entity_id_map)