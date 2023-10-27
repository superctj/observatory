import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import itertools
import random

import pandas as pd
import torch
import numpy as np
from observatory.models.huggingface_models import (
    load_transformers_model,
    load_transformers_tokenizer_and_max_length,
)
from observatory.common_util.row_based_truncate import row_based_truncate
from observatory.common_util.mcv import compute_mcv
from torch.linalg import inv, norm
from observatory.models.hugging_face_row_embeddings import (
    get_hugging_face_row_embeddings_batched, get_hugging_face_row_embeddings
)

model_name = "roberta-base"
tokenizer, max_length = load_transformers_tokenizer_and_max_length(model_name)
print(max_length)
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = load_transformers_model(model_name, device)
model.eval()
padding_token = "<pad>" if model_name.startswith("t5") else "[PAD]"
data = {
    "Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"],
    "Age": ["56", "45", "59"],
    "Number of movies": ["87", "53", "69"],
}
table = pd.DataFrame.from_dict(data)
tables = [table, table, table, table]
all_embeddings = get_hugging_face_row_embeddings_batched(
        tables,
        model_name,
        tokenizer,
        max_length,
        model,
    )
