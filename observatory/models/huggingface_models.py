"""
Load tokenizers and models from HuggingFace Transformers library
"""
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    TapasModel,
    TapasTokenizer,
    T5Model,
    T5Tokenizer,
)

SUPPORTED_MODELS = ["BERT", "RoBERTa", "TAPAS", "T5", "TAPEX"]


def load_transformers_tokenizer(model_name: str) -> object:
    """Load a tokenizer from the HuggingFace Transformers library.
    
    Args:
        model_name: The name of the model.
        
    Returns:
        tokenizer: The tokenizer.
    """
    if model_name.startswith("bert"):
        try:
            tokenizer = BertTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
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
            tokenizer = T5Tokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif model_name.startswith("microsoft/tapex"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    else:
        print(f"Unrecognized tokenizer name: {model_name}")
        print(f"You may choose one of: {SUPPORTED_MODELS}")

    return tokenizer


def load_transformers_tokenizer_and_max_length(model_name: str) -> tuple:
    """Load a tokenizer from the HuggingFace Transformers library.
    
    Args:
        model_name: The name of the model.
        
    Returns:
        tokenizer: The tokenizer.
        max_length: The maximum length of the tokens.
    """
    max_length = None
    if model_name.startswith("bert"):
        try:
            tokenizer = BertTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = BertTokenizer.from_pretrained(model_name)
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 512
    elif model_name.startswith("roberta"):
        try:
            tokenizer = RobertaTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 512
    elif model_name.startswith("google/tapas"):
        try:
            tokenizer = TapasTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = TapasTokenizer.from_pretrained(model_name)
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 512
    elif model_name.startswith("t5"):
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = T5Tokenizer.from_pretrained(model_name)
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 512
    elif model_name.startswith("microsoft/tapex"):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        try:
            max_length = tokenizer.model_max_length
        except AttributeError:
            max_length = 1024
    else:
        print(f"Unrecognized tokenizer name: {model_name}")
        print(f"You may choose one of: {SUPPORTED_MODELS}")

    return tokenizer, max_length


def load_transformers_model(model_name: str, device: torch.device) -> object:
    """Load a model from the HuggingFace Transformers library.
    
    Args:
        model_name: The name of the model.
        device: The device to use.
        
    Returns:
        model: The model.
    """
# "bert-base-uncased"
    if model_name.startswith("bert"):
        try:
            model = BertModel.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = BertModel.from_pretrained(model_name)
    elif model_name.startswith("roberta"):
        try:
            model = RobertaModel.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            model = RobertaModel.from_pretrained(model_name)
    elif model_name.startswith("google/tapas"):
        try:
            model = TapasModel.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            model = TapasModel.from_pretrained(model_name)
    elif model_name.startswith("t5"):
        try:
            model = T5Model.from_pretrained(model_name, local_files_only=True)
        except OSError:
            model = T5Model.from_pretrained(model_name)
    elif model_name.startswith("microsoft/tapex"):
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, local_files_only=True
            )
        except OSError:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        print(f"Unrecognized model name: {model_name}")
        print(f"You may choose any variant of: {SUPPORTED_MODELS}")

    model = model.to(device)
    model.eval()

    return model
