import os
import unittest

import pandas as pd
import torch

from observatory.models.hugging_face_column_embeddings import (
    get_hugging_face_column_embeddings_batched,
)
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)


class TestGetColEmbeddingsBERT(unittest.TestCase):
    def setUp(self):
        self.model_name = "bert-base-uncased"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = load_transformers_model(self.model_name, self.device)
        self.tokenizer, self.max_length = (
            load_transformers_tokenizer_and_max_length(self.model_name)
        )

    def test_get_col_embeddings_from_wikitables(self):
        wikitables_dir = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            "sample_data/wiki_tables",
        )
        table_files = [
            f for f in os.listdir(wikitables_dir) if f.endswith(".csv")
        ]

        all_tables = []
        for f in table_files:
            table = pd.read_csv(os.path.join(wikitables_dir, f))
            all_tables.append(table)

        all_embeddings = get_hugging_face_column_embeddings_batched(
            all_tables,
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            model=self.model,
            batch_size=32,
            device=self.device,
        )

        assert len(all_embeddings) == len(all_tables)

    def test_get_col_embeddings_from_sotab(self):
        pass
