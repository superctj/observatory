import torch

from observatory.datasets.turl_wiki_tables import prepare_data_for_lm
from observatory.models.huggingface_models import load_transformers_model, load_transformers_tokenizer
from observatory.common_util.util import save_embeddings_to_pickle


def bert_embedding_inference(tokenizer, model, sample_size: int, id2metadata: dict, use_metadata: str, device) -> dict:
    """
    Build a dictionary that maps entity ID to their embeddings.
    """

    count = 0
    bert_embeddings = {}
    bert_matrix = torch.zeros(
        (sample_size, model.config.hidden_size), device=device)

    for i in range(4, sample_size+4):
        if i not in id2metadata:
            continue

        num_metadata = len(id2metadata[i]) if use_metadata == "all" else 1
    
        with torch.no_grad():
            for j in range(num_metadata):
                if use_metadata == "no":
                    inputs = tokenizer(id2metadata[i][j][0], return_tensors="pt")
                else:
                    # 0 -> entity text, 1 -> metadata
                    inputs = tokenizer(id2metadata[i][j][0] + " " + id2metadata[i][j][1], return_tensors="pt")
                
                entity_length = len(tokenizer.tokenize(id2metadata[i][j][0]))
                outputs = model(**inputs.to(device), output_hidden_states=True).last_hidden_state
                
                bert_matrix[count, :] += outputs[0][1:(1+entity_length)].mean(axis=0)

        bert_matrix[count] /= num_metadata
        bert_embeddings[i] = bert_matrix[count].tolist()
        count += 1

    return bert_embeddings


def run_bert(data_dir: str, entity_id_map: dict, model_name: str, sample_size: int, use_metadata: str, output_file_path: str, device):
    id2metadata = prepare_data_for_lm(data_dir, entity_id_map)

    tokenizer = load_transformers_tokenizer(model_name)
    model = load_transformers_model(model_name, device)

    bert_embeddings = bert_embedding_inference(tokenizer, model, sample_size, id2metadata, use_metadata, device)

    save_embeddings_to_pickle(output_file_path, bert_embeddings)