"""
Load tokenizers and models from HuggingFace Transformers library
"""

from transformers import (
    BertTokenizer, BertModel,
    RobertaTokenizer, RobertaModel,
    TapasTokenizer, TapasModel,
    T5Tokenizer, T5Model
)


SUPPORTED_MODELS = ["BERT", "RoBERTa", "TAPAS", "T5"]


def load_transformers_tokenizer(model_name: str):
    if model_name.startswith("bert"):
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(model_name)
    elif model_name.startswith("roberta"):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name.startswith("google/tapas"):
        try:
            tokenizer = TapasTokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            tokenizer = TapasTokenizer.from_pretrained(model_name)
    elif model_name.startswith("t5"):
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
        except OSError:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
    else:
        print(f"Unrecognized tokenizer name: {model_name}")
        print(f"You may choose one of: {SUPPORTED_MODELS}")
    
    return tokenizer


def load_transformers_model(model_name: str, device): # "bert-base-uncased"
    if model_name.startswith("bert"):
        try:
            model = BertModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = BertModel.from_pretrained(model_name)
    elif model_name.startswith("roberta"):
        try:
            model = RobertaModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = RobertaModel.from_pretrained(model_name)
    elif model_name.startswith("google/tapas"):
        try:
            model = TapasModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = TapasModel.from_pretrained(model_name)
    elif model_name.startswith("t5"):
        try:
            model = T5Model.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = T5Model.from_pretrained(model_name)
    else:
        print(f"Unrecognized model name: {model_name}")
        print(f"You may choose any variant of: {SUPPORTED_MODELS}")

    model.to(device)
    model.eval()

    return model