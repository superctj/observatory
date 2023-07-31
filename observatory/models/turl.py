import torch
import torch.nn as nn

from observatory.models.TURL.model.configuration import TableConfig
from observatory.models.TURL.model.model import HybridTableMaskedLM

# from observatory.models.transformers import load_transformers_tokenizer


def load_turl_model(config, ckpt_path: str):
    model = HybridTableMaskedLM(config, is_simple=True)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    return model


class TURL(nn.Module):
    def __init__(self, config_name: str, ckpt_path: str):
        super(TURL, self).__init__()

        config = TableConfig.from_pretrained(config_name)
        self.model = load_turl_model(config, ckpt_path)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_column_embeddings(
        self,
        input_tok,
        input_tok_type,
        input_tok_pos,
        input_tok_mask,
        input_ent_text,
        input_ent_text_length,
        input_ent,
        input_ent_type,
        input_ent_mask,
        column_entity_mask,
        column_header_mask,
    ):
        tok_outputs, ent_outputs, _ = self.model.table(
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            None,
            input_ent,
            input_ent_type,
            input_ent_mask,
            None,
        )

        tok_sequence_output = self.dropout(tok_outputs[0])
        tok_col_output = torch.matmul(column_header_mask, tok_sequence_output)
        tok_col_output /= column_header_mask.sum(dim=-1, keepdim=True).clamp(
            1.0, 9999.0
        )

        ent_sequence_output = self.dropout(ent_outputs[0])
        ent_col_output = torch.matmul(column_entity_mask, ent_sequence_output)
        ent_col_output /= column_entity_mask.sum(dim=-1, keepdim=True).clamp(
            1.0, 9999.0
        )

        col_embeddings = torch.cat([tok_col_output, ent_col_output], dim=-1)
        return col_embeddings

    def get_entity_embeddings(
        self,
        input_tok,
        input_tok_type,
        input_tok_pos,
        input_tok_mask,
        input_ent_text,
        input_ent_text_length,
        input_ent,
        input_ent_type,
        input_ent_mask,
    ):
        tok_outputs, ent_outputs, _ = self.model.table(
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            None,
            input_ent,
            input_ent_type,
            input_ent_mask,
            None,
        )
        # # print(type(tok_outputs))
        # print(len(tok_outputs))
        # print(tok_outputs[0].shape)

        # print(type(ent_outputs))
        # print(len(ent_outputs))
        # print(ent_outputs[0].shape)
        return ent_outputs[0]


def get_column_embeddings_example():
    from observatory.datasets.turl_wiki_tables import TurlWikiTableDataset
    from observatory.models.TURL.data_loader.CT_Wiki_data_loaders import CTLoader
    from observatory.models.TURL.model.transformers import BertTokenizer
    from observatory.models.TURL.utils.util import load_entity_vocab

    data_dir = "/home/congtj/observatory/data/"
    min_ent_count = 2

    entity_vocab = load_entity_vocab(
        data_dir, ignore_bad_title=True, min_ent_count=min_ent_count
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_dataset = TurlWikiTableDataset(
        data_dir, entity_vocab, tokenizer, split="test", force_new=False
    )
    test_dataloader = CTLoader(test_dataset, batch_size=1, is_train=False)

    config = "/home/congtj/observatory/observatory/models/TURL/configs/table-base-config_v2.json"
    ckpt_path = "/ssd/congtj/observatory/turl_models/pytorch_model.bin"
    device = torch.device("cuda:1")

    model = TURL(config, ckpt_path)
    model.to(device)

    for batch in test_dataloader:
        (
            table_ids,
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent,
            input_ent_type,
            input_ent_mask,
            column_entity_mask,
            column_header_mask,
            labels_mask,
            labels,
        ) = batch
        input_tok = input_tok.to(device)
        input_tok_type = input_tok_type.to(device)
        input_tok_pos = input_tok_pos.to(device)
        input_tok_mask = input_tok_mask.to(device)
        input_ent_text = input_ent_text.to(device)
        input_ent_text_length = input_ent_text_length.to(device)
        input_ent = input_ent.to(device)
        input_ent_type = input_ent_type.to(device)
        input_ent_mask = input_ent_mask.to(device)
        column_entity_mask = column_entity_mask.to(device)
        column_header_mask = column_header_mask.to(device)
        labels_mask = labels_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            col_embeddings = model.get_column_embeddings(
                input_tok,
                input_tok_type,
                input_tok_pos,
                input_tok_mask,
                input_ent_text,
                input_ent_text_length,
                input_ent,
                input_ent_type,
                input_ent_mask,
                column_entity_mask,
                column_header_mask,
            )

            print("=" * 50)
            print("Number of columns: ", labels.shape)
            print("Column embeddings shape: ", col_embeddings.shape)

            break


def get_entity_embeddings_example(data_dir, config, ckpt_path):
    from observatory.datasets.turl_wiki_tables import (
        TurlWikiTableCellDataset,
        EntityEmbeddingLoader,
    )
    from observatory.models.TURL.model.transformers import BertTokenizer
    from observatory.models.TURL.utils.util import load_entity_vocab

    min_ent_count = 2

    entity_vocab = load_entity_vocab(
        data_dir, ignore_bad_title=True, min_ent_count=min_ent_count
    )
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    test_dataset = TurlWikiTableCellDataset(
        data_dir, entity_vocab, tokenizer, split="test", force_new=False
    )
    test_dataloader = EntityEmbeddingLoader(test_dataset, batch_size=1, is_train=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TURL(config, ckpt_path)
    model.to(device)
    all_entity_embeddings = []
    count = 0
    for batch in test_dataloader:
        count += 1
        while not line_exist[count]:
            count += 1
        (
            table_ids,
            entity_info,
            input_tok,
            input_tok_type,
            input_tok_pos,
            input_tok_mask,
            input_ent_text,
            input_ent_text_length,
            input_ent,
            input_ent_type,
            input_ent_mask,
            column_entity_mask,
            column_header_mask,
            labels_mask,
            labels,
        ) = batch
        print(table_ids)
        print(input_tok.shape)
        print(input_ent.shape)
        print("-" * 50)
        input_tok = input_tok.to(device)
        input_tok_type = input_tok_type.to(device)
        input_tok_pos = input_tok_pos.to(device)
        input_tok_mask = input_tok_mask.to(device)
        input_ent_text = input_ent_text.to(device)
        input_ent_text_length = input_ent_text_length.to(device)
        input_ent = input_ent.to(device)
        input_ent_type = input_ent_type.to(device)
        input_ent_mask = input_ent_mask.to(device)
        column_entity_mask = column_entity_mask.to(device)
        column_header_mask = column_header_mask.to(device)
        labels_mask = labels_mask.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            try:
                entity_embeddings = model.get_entity_embeddings(
                    input_tok,
                    input_tok_type,
                    input_tok_pos,
                    input_tok_mask,
                    input_ent_text,
                    input_ent_text_length,
                    input_ent,
                    input_ent_type,
                    input_ent_mask,
                )
                embedding_dict = {}
                for i, ([r_idx, c_idx], (entity_id, entity_text)) in enumerate(
                    entity_info[0]
                ):
                    embedding_dict[(r_idx, c_idx)] = (
                        entity_embeddings[0][i + 1],
                        entity_id,
                    )  # i+1 because the first embedding is for page entity
                all_entity_embeddings.append((count, embedding_dict))

            except Exception as e:
                print(e)
                break

    return all_entity_embeddings


if __name__ == "__main__":
    # get_column_embeddings_example()
    get_entity_embeddings_example()
