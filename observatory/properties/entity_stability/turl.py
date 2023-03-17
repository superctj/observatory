import torch

from observatory.datasets.turl_wiki_tables import prepare_data_for_turl
from observatory.models.huggingface_models import load_transformers_tokenizer
from observatory.models.turl import load_turl_model

from observatory.common_util.util import save_embeddings_to_pickle


def turl_embedding_inference(model, sample_size, id2metadata, use_metadata: str, device) -> dict:
    """
    Build a dictionary that maps entity ID to their embeddings.
    """

    count = 0
    turl_embeddings = {}
    turl_matrix = torch.zeros((sample_size, 312), device=device)

    for i in range(4, sample_size+4):
        if i not in id2metadata:
            continue
        
        num_metadata = len(id2metadata[i]) if use_metadata == "all" else 1
        
        with torch.no_grad():
            for j in range(num_metadata):
                if use_metadata == "no":
                    tok_outputs_meta, ent_outputs_meta, _ = model.table(
                        None, None, None, None,
                        None, None, None, torch.LongTensor([[i]]).to(device),
                        None, None, None)
                else:
                    tok_outputs_meta, ent_outputs_meta, _ = model.table(
                        id2metadata[i][j][0], id2metadata[i][j][1], id2metadata[i][j][2], None, None, None, None, torch.LongTensor([[i]]).to(device), None, None, None)

                turl_matrix[count, :] += ent_outputs_meta[0][0][0]

        turl_matrix[count] /= num_metadata
        turl_embeddings[i] = turl_matrix[count].tolist()
        count += 1
    
    return turl_embeddings


def run_turl(data_dir: str, entity_vocab: dict, entity_id_map: dict, model_config: str, ckpt_path: str, sample_size: int, use_metadata: str, output_file_path: str, device):
    tokenizer = load_transformers_tokenizer("bert-base-uncased")
    model = load_turl_model(model_config, ckpt_path, device)
    
    id2metadata = prepare_data_for_turl(data_dir, entity_vocab, entity_id_map, tokenizer, device)

    turl_embeddings = turl_embedding_inference(model, sample_size, id2metadata, use_metadata, device)

    save_embeddings_to_pickle(output_file_path, turl_embeddings)