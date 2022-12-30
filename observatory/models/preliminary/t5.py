from __future__ import absolute_import, division, print_function
import torch
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
from transformers import TapasTokenizer, TapasModel
import pandas as pd

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


from tqdm import trange
from tqdm.autonotebook import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.RE_data_loaders import *
from data_loader.EL_data_loaders import *
from model.configuration import TableConfig
from model.model import HybridTableMaskedLM, HybridTableCER, HybridTableModel, TableHeaderRanking, HybridTableCT,HybridTableEL,HybridTableRE,BertRE
from model.transformers import BertConfig,BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers import T5Tokenizer, T5Model
from transformers import BertTokenizer, BertModel

data_dir = 'data/'
device = torch.device('cuda')
config_name = "configs/table-base-config_v2.json"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sample_size = 1000
k = 10
no_metadata = True
all_metadata = not no_metadata and True

# load entity vocab from entity_vocab.txt
entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
print(len(entity_vocab))
entity_wikid2id = {int(entity_vocab[x]['wiki_id']):x for x in entity_vocab if x<=sample_size+3 and x>=4}

MODEL_CLASSES = {
    'CF' : (TableConfig, HybridTableMaskedLM, BertTokenizer)
}

t5_matrix = torch.zeros((sample_size, 768), device=device)
t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5 = T5Model.from_pretrained("t5-base")
t5.to(device)
t5.eval()

# config_class, model_class, _ = MODEL_CLASSES['CF']
# config = config_class.from_pretrained(config_name)
# config.output_attentions = True
# dataset = WikiHybridTableDataset(data_dir,entity_vocab,max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)

# checkpoint = "./"
# model = model_class(config, is_simple=True)
# checkpoint = torch.load(os.path.join(checkpoint, 'pytorch_model.bin'))
# model.load_state_dict(checkpoint)
# model.to(device)
# model.eval()

def pairwise_cosine_knn(a, b, k):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return torch.argsort(res, dim=1, descending=True)[:, 1:k+1]

# def build_meta_input(pgTitle, secTitle, caption, headers, config):
#     tokenized_pgTitle = config.tokenizer.encode(pgTitle, max_length=config.max_title_length, add_special_tokens=False)
#     tokenized_meta = tokenized_pgTitle+\
#                     config.tokenizer.encode(secTitle, max_length=config.max_title_length, add_special_tokens=False)
#     if caption != secTitle:
#         tokenized_meta += config.tokenizer.encode(caption, max_length=config.max_title_length, add_special_tokens=False)
#     tokenized_headers = [config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False) for header in headers]
#     input_tok = []
#     input_tok_pos = []
#     input_tok_type = []
#     tokenized_meta_length = len(tokenized_meta)
#     input_tok += tokenized_meta
#     input_tok_pos += list(range(tokenized_meta_length))
#     input_tok_type += [0]*tokenized_meta_length
#     for tokenized_header in tokenized_headers:
#         input_tok += tokenized_header
#         tokenized_header_length = len(tokenized_header)
#         input_tok_pos += list(range(tokenized_header_length))
#         input_tok_type += [1]*tokenized_header_length

#     input_tok = torch.LongTensor([input_tok])
#     input_tok_type = torch.LongTensor([input_tok_type])
#     input_tok_pos = torch.LongTensor([input_tok_pos])

#     return input_tok, input_tok_type, input_tok_pos

id2table = {}
id2meta = {}
id2data = {}
with open(os.path.join(data_dir, "test_tables.jsonl"), "r") as f:
    for table in tqdm(f):
        table = json.loads(table.strip())
        caption = table.get("tableCaption", "")
        pgTitle = table.get("pgTitle", "")
        secTitle = table.get("sectionTitle", "")
        rows = table.get("tableData", [])
        headers = table.get("tableHeaders", [])[0]
        # input_token, input_token_type, input_token_pos = build_meta_input(pgTitle, secTitle, caption, headers, dataset)
        dup_id = set()

        for row in rows:
            offset = 0
            for cell in row:
                # num_cell += 1

                if len(cell['surfaceLinks']) > 0:
                    wikid = int(cell['surfaceLinks'][0]['target']['id'])
                    entity_text = cell['surfaceLinks'][0]['target']['title']
                    # table_dict[headers[offset]].append(entity_text)
                    # offset += 1
                else:
                    # table_dict[headers[offset]].append(cell['text'])
                    # offset += 1
                    continue
                
                if wikid not in entity_wikid2id:
                    continue
                else:
                    id = entity_wikid2id[wikid]
                    if id in dup_id:
                        continue
                dup_id.add(id)
                
                # if id not in id2table:
                #     id2table[id] = []
                # id2table[id].append([input_token.to(device), input_token_type.to(device), input_token_pos.to(device)])
                if id not in id2meta:
                    id2meta[id] = []
                id2meta[id].append([entity_text, ' '.join([caption, pgTitle, secTitle] + headers)])

        # for id in dup_id:
        #     if id not in id2data:
        #         id2data[id] = []
        #     id2data[id].append(pandas_table)

count = 0
sim = 0
t5_emb = {}

for i in range(4, sample_size+4):
    # if i not in id2table or len(id2table[i]) < 2:
    #     continue
    if i not in id2meta:
        continue
    num_metadata = len(id2meta[i]) if all_metadata else 1
    # base_matrix[count, :] = torch.FloatTensor(dict[str(entity_vocab[i]['wiki_id'])])
    with torch.no_grad():
        for j in range(num_metadata):
        # for j in range(len(id2table[i])):
            t5_entity_length = len(t5_tokenizer.tokenize(id2meta[i][j][0]))
            if no_metadata:
                t5_input = t5_tokenizer(id2meta[i][j][0], return_tensors="pt")
            else:
                t5_input = t5_tokenizer(id2meta[i][j][0] + " " + id2meta[i][j][1], return_tensors="pt")
            t5_output = t5.encoder(input_ids=t5_input["input_ids"].to(device), 
                attention_mask=t5_input["attention_mask"].to(device), 
                return_dict=True,
                output_hidden_states=True).last_hidden_state
            t5_matrix[count, :] += t5_output[0][:t5_entity_length].mean(axis=0)

    t5_matrix[count] /= num_metadata
    t5_emb[i] = t5_matrix[count].tolist()
    count += 1
if no_metadata:
    t5_name = 't5_emb_no_metadata.pkl'
else:
    t5_name = 't5_emb_all_metadata.pkl' if all_metadata else 't5_emb_one_metadata.pkl'
with open(os.path.join(data_dir, t5_name), 'wb') as f:
    pickle.dump(t5_emb, f)

t5_matrix = t5_matrix[:count]
print(t5_matrix.size())
# bert_matrix = bert_matrix[:count]
# roberta_matrix = roberta_matrix[:count]
# print(bert_matrix.size())
# print(roberta_matrix.size())
# bert_knn = pairwise_cosine_knn(bert_matrix, bert_matrix, k)
# roberta_knn = pairwise_cosine_knn(roberta_matrix, roberta_matrix, k)

# base_knn_file = 'tapas_knn_all_metadata.pkl' if all_metadata else 'tapas_knn_one_metadata.pkl'
# turl_knn_file = 'turl_knn_all_metadata.pkl' if all_metadata else 'turl_knn_one_metadata.pkl'
# torch.save(base_knn, os.path.join(data_dir, base_knn_file))
# torch.save(turl_knn, os.path.join(data_dir, turl_knn_file))

