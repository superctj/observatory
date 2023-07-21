import argparse
import os
import operator
import pickle
from functools import reduce

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

import transformers
from transformers import BertTokenizer, BertConfig

from observatory.models.DODUO.doduo.doduo.model import BertForMultiOutputClassification, BertMultiPairPooler


class AnnotatedDataFrame:
    def __init__(self, df):
        self.df = df


def cell_embedding_collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    entity_token_positions = [sample["entity_token_positions"] for sample in samples]
    batch = {"data": data, "label": label, "entity_token_positions": entity_token_positions}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


class SingleTableCellEmbeddingDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 entity_info: list,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 device: torch.device = None):
        if device is None:
            device = torch.device("cpu")

        entity_column_ids = set()
        entity_cells = set()
        entity_cell_id_map = {}
        for entity in entity_info: # entity [[i, j], (entity_id, entity_text)]
            entity_column_ids.add(entity[0][1])
            entity_cells.add(tuple(entity[0]))
            entity_cell_id_map[tuple(entity[0])] = entity[1][0]

        self.entity_token_positions = {}
        cls_token_positions = []
        num_tokens = 0
        for j in range(len(df.columns)):
            cls_token_positions.append(num_tokens)
            num_tokens += 1 # [CLS] token at the beginning of each column
            if j not in entity_column_ids:
                column_values = " ".join([str(x) for x in df.iloc[:, j].dropna().tolist()])
                column_tokens = tokenizer.encode(column_values, add_special_tokens=False, max_length=max_length, truncation=True)
                # print("=" * 50)
                # print(column_tokens)
                num_tokens += len(column_tokens)
            else:
                column_num_tokens = 0
                for i, cell_values in enumerate(df.iloc[:, j].tolist()):
                    cell_tokens = tokenizer.encode(cell_values, add_special_tokens=False)

                    token_budget = max_length - column_num_tokens
                    assert(token_budget >= 0)
                    if len(cell_tokens) <= token_budget:
                        if (i, j) in entity_cells:
                            end_pos = num_tokens + len(cell_tokens)
                            self.entity_token_positions[(i, j)] = [(num_tokens, end_pos), entity_cell_id_map[(i, j)]] # left inclusive, right not inclusive

                        column_num_tokens += len(cell_tokens)
                        num_tokens += len(cell_tokens)
                    else:
                        truncated_cell_tokens = cell_tokens[:token_budget]
                        if (i, j) in entity_cells:
                            end_pos = num_tokens + token_budget
                            self.entity_token_positions[(i, j)] = [(num_tokens, end_pos), entity_cell_id_map[(i, j)]] # left inclusive, right not inclusive

                        column_num_tokens += token_budget
                        num_tokens += token_budget
                    
                    assert(column_num_tokens <= max_length)
                    if column_num_tokens == max_length:
                        break
            
            num_tokens += 1 # [SEP] token at the end of each column

        if num_tokens > 512:
            raise ValueError("Table has more than 512 tokens!")
    
        # print("=" * 50)
        # print(cls_token_positions)

        data_list = []
        for i in range(len(df.columns)):
            data_list.append([
                1,  # Dummy table ID (fixed)
                0,  # Dummy label ID (fixed)
                " ".join([str(x) for x in df.iloc[:, i].dropna().tolist()])
            ])
        self.df = pd.DataFrame(data_list,
                               columns=["table_id", "label_ids", "data"])

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2, truncation=True)).tolist(
                )
            # print("=" * 50)
            # print(token_ids_list)
            # print(len(token_ids_list))
            # print("=" * 50)
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            # print(token_ids)
            # print(len(token_ids))
            # print("=" * 50)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "entity_token_positions": self.entity_token_positions
        }


class Doduo:
    def __init__(self, args=None, basedir="./"):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda:1")

        if args is None:
            args = argparse.Namespace

        self.args = args
        self.args.colpair = True
        self.args.shortcut_name = "bert-base-uncased"
        self.args.batch_size = 16

        ## Load models
        self.tokenizer = BertTokenizer.from_pretrained(self.args.shortcut_name)

        if self.args.model == "viznet":
            coltype_model_path = os.path.join(
                basedir,
                "model/sato0_mosato_bert_bert-base-uncased-bs16-ml-32__sato0-1.00_best_micro_f1.pt"
            )
            coltype_num_labels = 78
        elif self.args.model == "wikitable":
            coltype_model_path = os.path.join(
                basedir,
                "model/turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-16__turl-1.00_turl-re-1.00=turl_best_micro_f1.pt"
            )
            colrel_model_path = os.path.join(
                basedir,
                "model/turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-16__turl-1.00_turl-re-1.00=turl-re_best_micro_f1.pt"
            )
            coltype_num_labels = 255
            colrel_num_labels = 121

            ## Load mlb
            with open(os.path.join(basedir, "data/turl_coltype_mlb.pickle"),
                      "rb") as fin:
                self.coltype_mlb = pickle.load(fin)
            with open(os.path.join(basedir, "data/turl_colrel_mlb.pickle"),
                      "rb") as fin:
                self.colrel_mlb = pickle.load(fin)
        else:
            raise ValueError("Invalid args.model: {}".format(args.model))

        # ===== Not necessary?
        coltype_config = BertConfig.from_pretrained(
            self.args.shortcut_name,
            num_labels=coltype_num_labels,
            output_attentions=True,
            output_hidden_states=True)
        # =====

        self.coltype_model = BertForMultiOutputClassification.from_pretrained(
            self.args.shortcut_name,
            num_labels=coltype_num_labels,
            output_attentions=True,
            output_hidden_states=True,
        ).to(self.device)
        self.coltype_model.load_state_dict(
            torch.load(coltype_model_path, map_location=self.device))
        self.coltype_model.eval()

        if self.args.model == "wikitable":
            self.colrel_model = BertForMultiOutputClassification.from_pretrained(
                self.args.shortcut_name,
                num_labels=colrel_num_labels,
                output_attentions=True,
                output_hidden_states=True).to(self.device)
            if self.args.colpair:
                config = BertConfig.from_pretrained(self.args.shortcut_name)
                self.colrel_model.bert.pooler = BertMultiPairPooler(config).to(
                    self.device)

            self.colrel_model.load_state_dict(
                torch.load(colrel_model_path, map_location=self.device))
            self.colrel_model.eval()

    def get_entity_embeddings(self, df: pd.DataFrame, entity_info):
        ## Dataset
        try:
            input_dataset = SingleTableCellEmbeddingDataset(df, entity_info, self.tokenizer)
        except ValueError as e:
            raise e

        input_dataloader = DataLoader(input_dataset,
                                      batch_size=self.args.batch_size,
                                      collate_fn=cell_embedding_collate_fn)

        ## Prediction
        batch = next(iter(input_dataloader))
        batch["data"] = batch["data"].to(self.device)
        entity_token_positions = batch["entity_token_positions"][0]

        outputs = self.coltype_model.bert.encoder(
            self.coltype_model.bert.embeddings(batch["data"].T),
            output_attentions=True,
            output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state.squeeze(
            0)  # SeqLen * DimSize

        cls_indexes = torch.nonzero(
            batch["data"].T.squeeze(0) ==
            self.tokenizer.cls_token_id).T.squeeze(0).detach().cpu().numpy()
        # print("=" * 50)
        # print(cls_indexes)

        entity_embeddings = {}
        for entity_cell in entity_token_positions:
            (start_idx, end_idx), entity_id = entity_token_positions[entity_cell]

            ent_embedding = torch.zeros(768)
            for index in range(start_idx, end_idx):
                ent_embedding += last_hidden_states[index].squeeze(
                0).detach().cpu()
            
            entity_embeddings[entity_cell] = (ent_embedding, entity_id)

        return entity_embeddings
