# from __future__ import absolute_import, division, print_function
import json
import os
import pickle
import sys

sys.path.append("/home/congtj/observatory/models/TURL")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import TapasTokenizer, TapasModel

from TURL.data_loader.hybrid_data_loaders import WikiHybridTableDataset
from TURL.utils.util import load_entity_vocab
from TURL.model.configuration import TableConfig
from TURL.model.model import HybridTableMaskedLM


# from data_loader.hybrid_data_loaders import *
# from data_loader.header_data_loaders import *
# from data_loader.CT_Wiki_data_loaders import *
# from data_loader.RE_data_loaders import *
# from data_loader.EL_data_loaders import *
# from model.configuration import TableConfig
# from model.model import HybridTableMaskedLM
# from utils.util import *


# data_dir = '../data/'
# device = torch.device('cuda')
# config_name = "configs/table-base-config_v2.json"
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# sample_size = 1000
# k = 10
# no_metadata = True
# all_metadata = not no_metadata and True

# load entity vocab from entity_vocab.txt
# entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
# print(len(entity_vocab))
# entity_wikid2id = {int(entity_vocab[x]['wiki_id']):x for x in entity_vocab if x<=sample_size+3 and x>=4}

# bert_matrix = torch.zeros((sample_size, 768), device=device)
# roberta_matrix = torch.zeros((sample_size, 768), device=device)
# roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# roberta = RobertaModel.from_pretrained("roberta-base")
# roberta.to(device)
# roberta.eval()
# bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# bert = BertModel.from_pretrained("bert-base-uncased")
# bert.to(device)
# bert.eval()

# id2metadata = {}
# id2meta = {}
# id2data = {}
# with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
#     for table in tqdm(f):
#         table = json.loads(table.strip())
#         caption = table.get("tableCaption", "")
#         pgTitle = table.get("pgTitle", "")
#         secTitle = table.get("sectionTitle", "")
#         rows = table.get("tableData", [])
#         headers = table.get("tableHeaders", [])[0]
#         # input_token, input_token_type, input_token_pos = build_meta_input(pgTitle, secTitle, caption, headers, dataset)
#         dup_id = set()

#         for row in rows:
#             offset = 0
#             for cell in row:
#                 # num_cell += 1

#                 if len(cell['surfaceLinks']) > 0:
#                     wikid = int(cell['surfaceLinks'][0]['target']['id'])
#                     entity_text = cell['surfaceLinks'][0]['target']['title']
#                     # table_dict[headers[offset]].append(entity_text)
#                     # offset += 1
#                 else:
#                     # table_dict[headers[offset]].append(cell['text'])
#                     # offset += 1
#                     continue

#                 if wikid not in entity_wikid2id:
#                     continue
#                 else:
#                     id = entity_wikid2id[wikid]
#                     if id in dup_id:
#                         continue
#                 dup_id.add(id)

#                 # if id not in id2metadata:
#                 #     id2metadata[id] = []
#                 # id2metadata[id].append([input_token.to(device), input_token_type.to(device), input_token_pos.to(device)])
#                 if id not in id2meta:
#                     id2meta[id] = []
#                 id2meta[id].append([entity_text, ' '.join([caption, pgTitle, secTitle] + headers)])


# count = 0
# sim = 0
# bert_emb = {}
# roberta_emb = {}
# for i in range(4, sample_size+4):
#     if i not in id2meta:
#         continue
#     num_metadata = len(id2meta[i]) if all_metadata else 1

#     with torch.no_grad():
#         for j in range(num_metadata):

#             if no_metadata:
#                 bert_input = bert_tokenizer(id2meta[i][j][0], return_tensors="pt")
#                 roberta_input = roberta_tokenizer(id2meta[i][j][0], return_tensors="pt")
#             else:
#                 bert_input = bert_tokenizer(id2meta[i][j][0] + " " + id2meta[i][j][1], return_tensors="pt")
#                 roberta_input = roberta_tokenizer(id2meta[i][j][0] + " " + id2meta[i][j][1], return_tensors="pt")

#             bert_entity_length = len(bert_tokenizer.tokenize(id2meta[i][j][0]))
#             bert_output = bert(**bert_input.to(device), output_hidden_states=True).last_hidden_state
#             roberta_entity_length = len(roberta_tokenizer.tokenize(id2meta[i][j][0]))
#             roberta_output = roberta(**roberta_input.to(device), output_hidden_states=True).last_hidden_state

#             bert_matrix[count, :] += bert_output[0][1:(1+bert_entity_length)].mean(axis=0)
#             roberta_matrix[count, :] += roberta_output[0][1:(1+roberta_entity_length)].mean(axis=0)

#     bert_matrix[count] /= num_metadata
#     roberta_matrix[count] /= num_metadata
#     bert_emb[i] = bert_matrix[count].tolist()
#     roberta_emb[i] = roberta_matrix[count].tolist()
#     count += 1

# if no_metadata:
#     bert_name = 'bert_emb_no_metadata.pkl'
#     roberta_name = 'roberta_emb_no_metadata.pkl'
# else:
#     bert_name = 'bert_emb_all_metadata.pkl' if all_metadata else 'bert_emb_one_metadata.pkl'
#     roberta_name = 'roberta_emb_all_metadata.pkl' if all_metadata else 'roberta_emb_one_metadata.pkl'
# with open(os.path.join(data_dir, bert_name), 'wb') as f:
#     pickle.dump(bert_emb, f)
# with open(os.path.join(data_dir, roberta_name), 'wb') as f:
#     pickle.dump(roberta_emb, f)


"""
    min_ent_count: consider only entities that appear at least specified times

    entity_vocab: index to dictionary. E.g.,
        39858: {
            'count': 27,
            'mid': 'm.074_w1',
            'wiki_id': 2341216,
            'wiki_title': '1934_European_Athletics_Championships'
        }

    entity_id_map: wiki_id to index. E.g.,
        7515890: 284,
        7515928: 306,
"""


def get_entity_id_map(data_dir, min_ent_count, sample_size) -> dict:
    entity_vocab = load_entity_vocab(
        data_dir, ignore_bad_title=True, min_ent_count=min_ent_count
    )

    entity_id_map = {
        int(entity_vocab[x]["wiki_id"]): x
        for x in entity_vocab
        if x <= sample_size + 3 and x >= 4
    }  # first four ids are reserved for special tokens, actual entity id starts at 4

    return entity_vocab, entity_id_map


def build_metadata_input_for_turl(pgTitle, secTitle, caption, headers, config):
    tokenized_pgTitle = config.tokenizer.encode(
        pgTitle, max_length=config.max_title_length, add_special_tokens=False
    )
    tokenized_meta = tokenized_pgTitle + config.tokenizer.encode(
        secTitle, max_length=config.max_title_length, add_special_tokens=False
    )
    if caption != secTitle:
        tokenized_meta += config.tokenizer.encode(
            caption, max_length=config.max_title_length, add_special_tokens=False
        )
    tokenized_headers = [
        config.tokenizer.encode(
            header, max_length=config.max_header_length, add_special_tokens=False
        )
        for header in headers
    ]
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    tokenized_meta_length = len(tokenized_meta)
    input_tok += tokenized_meta
    input_tok_pos += list(range(tokenized_meta_length))
    input_tok_type += [0] * tokenized_meta_length
    for tokenized_header in tokenized_headers:
        input_tok += tokenized_header
        tokenized_header_length = len(tokenized_header)
        input_tok_pos += list(range(tokenized_header_length))
        input_tok_type += [1] * tokenized_header_length

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])

    return input_tok, input_tok_type, input_tok_pos


def prepare_data_for_lm(data_dir, entity_wikid2id):
    id2meta = {}
    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        for table in tqdm(f):
            table = json.loads(table.strip())
            caption = table.get("tableCaption", "")
            pgTitle = table.get("pgTitle", "")
            secTitle = table.get("sectionTitle", "")
            rows = table.get("tableData", [])
            headers = table.get("tableHeaders", [])[0]
            dup_id = set()

            for row in rows:
                for cell in row:
                    if len(cell["surfaceLinks"]) > 0:
                        wikid = int(cell["surfaceLinks"][0]["target"]["id"])
                        entity_text = cell["surfaceLinks"][0]["target"]["title"]
                    else:
                        continue

                    if wikid not in entity_wikid2id:
                        continue
                    else:
                        id = entity_wikid2id[wikid]
                        if id in dup_id:
                            continue

                    dup_id.add(id)
                    if id not in id2meta:
                        id2meta[id] = []
                    id2meta[id].append(
                        [entity_text, " ".join([caption, pgTitle, secTitle] + headers)]
                    )
    return id2meta


def prepare_data_for_tapas(data_dir, entity_wikid2id):
    id2meta = {}
    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        for table in tqdm(f):
            table = json.loads(table.strip())
            caption = table.get("tableCaption", "")
            pgTitle = table.get("pgTitle", "")
            secTitle = table.get("sectionTitle", "")
            rows = table.get("tableData", [])
            headers = table.get("tableHeaders", [])[0]
            dup_id = set()
            table_dict = {
                "pgTitle": [pgTitle],
                "sectionTitle": [secTitle],
                "tableCaption": [caption],
                "tableHeaders": [" ".join(headers)],
            }
            pandas_table = pd.DataFrame.from_dict(table_dict)

            for row in rows:
                for cell in row:
                    if len(cell["surfaceLinks"]) > 0:
                        wikid = int(cell["surfaceLinks"][0]["target"]["id"])
                    else:
                        continue

                    if wikid not in entity_wikid2id:
                        continue
                    else:
                        id = entity_wikid2id[wikid]
                        if id in dup_id:
                            continue
                    dup_id.add(id)

            for id in dup_id:
                if id not in id2meta:
                    id2meta[id] = []
                id2meta[id].append(pandas_table)
    return id2meta


def prepare_data_for_turl(data_dir, entity_vocab, entity_id_map, tokenizer, device):
    id2metadata = {}
    dataset = WikiHybridTableDataset(
        data_dir,
        entity_vocab,
        max_cell=100,
        max_input_tok=350,
        max_input_ent=150,
        src="dev",
        max_length=[50, 10, 10],
        force_new=False,
        tokenizer=tokenizer,
        mode=0,
    )

    with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
        for table in tqdm(f):
            table = json.loads(table.strip())
            caption = table.get("tableCaption", "")
            pgTitle = table.get("pgTitle", "")
            secTitle = table.get("sectionTitle", "")
            rows = table.get("tableData", [])
            headers = table.get("tableHeaders", [])[0]

            (
                input_token,
                input_token_type,
                input_token_pos,
            ) = build_metadata_input_for_turl(
                pgTitle, secTitle, caption, headers, dataset
            )
            dup_id = set()

            for row in rows:
                for cell in row:
                    if len(cell["surfaceLinks"]) > 0:
                        wiki_id = int(cell["surfaceLinks"][0]["target"]["id"])
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

                    id2metadata[id].append(
                        [
                            input_token.to(device),
                            input_token_type.to(device),
                            input_token_pos.to(device),
                        ]
                    )
    return id2metadata


def load_bert_model(model_name, device):  # "bert-base-uncased"
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        model = BertModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model


def load_turl_model(config_name, ckpt_path, device):
    try:
        tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
    except OSError:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    config = TableConfig.from_pretrained(config_name)
    model = HybridTableMaskedLM(config, is_simple=True)

    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)
    model.eval()

    return tokenizer, model


def bert_embedding_inference(
    tokenizer,
    model,
    sample_size,
    id2metadata,
    all_metadata: bool,
    no_metadata: bool,
    device,
):
    count = 0
    bert_embeddings = {}
    bert_matrix = torch.zeros((sample_size, model.config.hidden_size), device=device)

    for i in range(4, sample_size + 4):
        if i not in id2metadata:
            continue
        num_metadata = len(id2metadata[i]) if all_metadata else 1

        with torch.no_grad():
            for j in range(num_metadata):
                if no_metadata:
                    inputs = tokenizer(id2metadata[i][j][0], return_tensors="pt")
                else:
                    # 0 -> entity text, 1 -> metadata
                    inputs = tokenizer(
                        id2metadata[i][j][0] + " " + id2metadata[i][j][1],
                        return_tensors="pt",
                    )

                entity_length = len(tokenizer.tokenize(id2metadata[i][j][0]))
                outputs = model(
                    **inputs.to(device), output_hidden_states=True
                ).last_hidden_state

                bert_matrix[count, :] += outputs[0][1 : (1 + entity_length)].mean(
                    axis=0
                )

        bert_matrix[count] /= num_metadata
        bert_embeddings[i] = bert_matrix[count].tolist()
        count += 1

        return bert_embeddings


def roberta_embedding_inference():
    pass


def tapas_embedding_inference():
    pass


def turl_embedding_inference(
    model, sample_size, id2metadata, all_metadata: bool, no_metadata: bool, device
):
    count = 0
    turl_embeddings = {}
    turl_matrix = torch.zeros((sample_size, 312), device=device)

    for i in range(4, sample_size + 4):
        if i not in id2metadata:
            continue
        num_metadata = len(id2metadata[i]) if all_metadata else 1

        with torch.no_grad():
            for j in range(num_metadata):
                if no_metadata:
                    tok_outputs_meta, ent_outputs_meta, _ = model.table(
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        torch.LongTensor([[i]]).to(device),
                        None,
                        None,
                        None,
                    )
                else:
                    tok_outputs_meta, ent_outputs_meta, _ = model.table(
                        id2metadata[i][j][0],
                        id2metadata[i][j][1],
                        id2metadata[i][j][2],
                        None,
                        None,
                        None,
                        None,
                        torch.LongTensor([[i]]).to(device),
                        None,
                        None,
                        None,
                    )

                turl_matrix[count, :] += ent_outputs_meta[0][0][0]

        turl_matrix[count] /= num_metadata
        turl_embeddings[i] = turl_matrix[count].tolist()
        count += 1

    return turl_embeddings


def save_embeddings(file_name: str, embeddings):
    with open(file_name, "wb") as f:
        pickle.dump(embeddings, f)


def test_lm():
    data_dir = "../data/"
    device = torch.device("cuda:0")
    min_ent_count = 2
    sample_size = 1000
    no_metadata = True
    all_metadata = True and not no_metadata
    output_file = "./bert_embeddings.pkl"

    _, entity_id_map = get_entity_id_map(data_dir, min_ent_count, sample_size)
    id2metadata = prepare_data_for_lm(data_dir, entity_id_map)

    bert_model_name = "bert-base-uncased"
    bert_tokenizer, bert_model = load_bert_model(bert_model_name, device)

    bert_embeddings = bert_embedding_inference(
        bert_tokenizer,
        bert_model,
        sample_size,
        id2metadata,
        all_metadata,
        no_metadata,
        device,
    )

    save_embeddings(output_file, bert_embeddings)


def test_turl():
    data_dir = "../data/"
    device = torch.device("cuda:0")
    min_ent_count = 2
    sample_size = 1000
    no_metadata = True
    all_metadata = True and not no_metadata
    output_file = "./turl_embeddings.pkl"

    config_name = "./TURL/configs/table-base-config_v2.json"
    ckpt_path = "/ssd/congtj/observatory/pytorch_model.bin"
    tokenizer, turl_model = load_turl_model(config_name, ckpt_path, device)

    entity_vocab, entity_id_map = get_entity_id_map(
        data_dir, min_ent_count, sample_size
    )
    id2metadata = prepare_data_for_turl(
        data_dir, entity_vocab, entity_id_map, tokenizer, device
    )

    turl_embeddings = turl_embedding_inference(
        turl_model, sample_size, id2metadata, all_metadata, no_metadata, device
    )

    save_embeddings(output_file, turl_embeddings)


if __name__ == "__main__":
    test_turl()
