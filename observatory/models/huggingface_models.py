"""
Load tokenizers and models from HuggingFace Transformers library
"""

from transformers import (
    BertTokenizer,
    BertModel,
    RobertaTokenizer,
    RobertaModel,
    TapasTokenizer,
    TapasModel,
    T5Tokenizer,
    T5Model,
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
            tokenizer = RobertaTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
    elif model_name.startswith("google/tapas"):
        try:
            tokenizer = TapasTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
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


def load_transformers_tokenizer_and_max_length(model_name: str):
    max_length = None
    if model_name.startswith("bert"):
        try:
            tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
            model = BertModel.from_pretrained(model_name)
            max_length = model.config.max_position_embeddings
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertModel.from_pretrained(model_name)
            max_length = model.config.max_position_embeddings
    elif model_name.startswith("roberta"):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            model = RobertaModel.from_pretrained(model_name)
            max_length = model.config.max_position_embeddings
        except OSError:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name)
            max_length = model.config.max_position_embeddings
    elif model_name.startswith("google/tapas"):
        try:
            tokenizer = TapasTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            max_length = 512  # TAPAS typically has a maximum sequence length of 512
        except OSError:
            tokenizer = TapasTokenizer.from_pretrained(model_name)
            max_length = 512
    elif model_name.startswith("t5"):
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
            model = T5Model.from_pretrained(model_name)
            max_length = model.config.max_position_embeddings
        except OSError:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5Model.from_pretrained(model_name)
            try:
                max_length = model.config.max_position_embeddings
            except:
                max_length = 512
    else:
        print(f"Unrecognized tokenizer name: {model_name}")
        print(f"You may choose one of: {SUPPORTED_MODELS}")

    return tokenizer, max_length


def load_transformers_model(model_name: str, device):  # "bert-base-uncased"
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
