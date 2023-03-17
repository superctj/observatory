import torch
import pickle

from observatory.models.turl import load_entity_vocab, load_turl_model, WikiHybridTableDataset
from observatory.models.huggingface_models import load_transformers_model, load_transformers_tokenizer
from observatory.datasets.gittable_schemas import build_lm_input, build_t5_input, build_tapas_input, build_turl_input

from torch import nn
from scipy import stats


def get_topk_schemas_and_attr_cooccurence(data_dir, k):
    """
        Select the top k frequent schemas.
        Build a dictionary that maps attributes to the top schemas they belong to,
        and another dictionary that records attribute pair cooccurence.
    """
    with open(data_dir + 'schema_count.pkl', 'rb') as f:
        schema_count = pickle.load(f)
    attr2schemas = {}
    cooccurence = {}
    topk_schemas = sorted(schema_count.items(), key=lambda x:x[1], reverse=True)[:k]
    for item in topk_schemas:
        schema = item[0]
        for i in range(len(schema)):
            for j in range(i+1, len(schema)):
                pair = frozenset((schema[i], schema[j]))
                if pair not in cooccurence:
                    cooccurence[pair] = 0

    total = 0
    for k, v in schema_count.items():
        total += v
        for pair in cooccurence.keys():
            p = list(pair)
            if p[0] in k and p[1] in k:
                cooccurence[pair] += v
            for attr in pair:
                if attr not in attr2schemas:
                    attr2schemas[attr] = set()
                if attr in k:
                    attr2schemas[attr].add(k)
    return attr2schemas, cooccurence


def lm_attribute_embedding_inference(tokenizer, model, attr2schemas, device):
    """
        Build a dictionary that maps attributes to their BERT/ROBERTA embeddings.
    """
    attr_emb = {}
    for attr in attr2schemas:
        attr_emb[attr] = torch.zeros((1, 768), device=device)
        num_emb = min(len(attr2schemas[attr]), 10)
        miss = 0
        with torch.no_grad():
            for i in range(num_emb):
                lm_input, attr2pos = build_lm_input(list(attr2schemas[attr])[i], tokenizer)
                if attr not in attr2pos:
                    miss += 1
                    continue
                lm_output = model(**lm_input.to(device), output_hidden_states=True).last_hidden_state
                attr_emb[attr] += lm_output[0][attr2pos[attr][0]:attr2pos[attr][1]].mean(axis=0)
            if num_emb == miss:
                print('invalid attr')
                del attr_emb[attr]
            else:
                attr_emb[attr] /= (num_emb - miss)

    return attr_emb
    
    
def turl_attribute_embedding_inference(config, model, attr2schemas, device):
    """
        Build a dictionary that maps attributes to their TURL embeddings.
    """
    attr_emb = {}
    for attr in attr2schemas:
        attr_emb[attr] = torch.zeros((1, 312), device=device)
        num_emb = min(len(attr2schemas[attr]), 10)
        with torch.no_grad():
            for i in range(num_emb):
                input_token, input_token_type, input_token_pos, attr2pos = build_turl_input(list(attr2schemas[attr])[i], config)
                tok_outputs_meta, _, _ = model.table(input_token.to(device), input_token_type.to(device), input_token_pos.to(device), None,
                                    None, None, None,
                                    None, None, None, None)
                attr_emb[attr] += tok_outputs_meta[0][0][attr2pos[attr][0]:attr2pos[attr][1]].mean(axis=0)
            attr_emb[attr] /= num_emb
            
    return attr_emb
    
    
def tapas_attribute_embedding_inference(tokenizer, model, attr2schemas):
    """
        Build a dictionary that maps attributes to their TAPAS embeddings.
    """
    attr_emb = {}
    for attr in attr2schemas:
        attr_emb[attr] = torch.zeros((1, 768))
        num_emb = min(len(attr2schemas[attr]), 10)
        with torch.no_grad():
            for i in range(num_emb):
                table = build_tapas_input(list(attr2schemas[attr])[i])
                tapas_input = tokenizer(table=table, queries=[attr], truncation=True, return_tensors="pt")
                tapas_output = model(**tapas_input).last_hidden_state
                attr_emb[attr] += tapas_output[0].mean(axis=0)
            attr_emb[attr] /= num_emb
    
    return attr_emb


def t5_attribute_embedding_inference(tokenizer, model, attr2schemas, device):
    """
        Build a dictionary that maps attributes to their T5 embeddings.
    """
    attr_emb = {}
    for attr in attr2schemas:
        attr_emb[attr] = torch.zeros((1, 768), device=device)
        num_emb = min(len(attr2schemas[attr]), 10)
        miss = 0
        with torch.no_grad():
            for i in range(num_emb):
                t5_input, attr2pos = build_t5_input(list(attr2schemas[attr])[i], tokenizer)
                t5_output = model.encoder(input_ids=t5_input["input_ids"].to(device), 
                    attention_mask=t5_input["attention_mask"].to(device), 
                    return_dict=True,
                    output_hidden_states=True).last_hidden_state
                attr_emb[attr] += t5_output[0][attr2pos[attr][0]:attr2pos[attr][1]].mean(axis=0)
            if num_emb == miss:
                print('invalid attr')
                del attr_emb[attr]
            else:
                attr_emb[attr] /= (num_emb - miss)
                
    return attr_emb


def calculate_spearman_correlation(attr_emb, cooccurence):
    """
        Calculate the spearman correlation between attribute embeddings and their cooccurence probabilities.
    """
    emb_sim = []
    co_prob = []
    cos = nn.CosineSimilarity()
    for pair in cooccurence:
        p = list(pair)
        if p[0] in attr_emb and p[1] in attr_emb:
            emb_sim.append(cos(attr_emb[p[0]], attr_emb[p[1]]))
            co_prob.append(cooccurence[pair])
    n = len(emb_sim)
    print('pairs of attr: ')
    print(n)

    res = stats.spearmanr(emb_sim, co_prob)
    return res


def test_turl(data_dir, k):
    entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
    dataset = WikiHybridTableDataset(data_dir, entity_vocab, max_cell=100, max_input_tok=350, max_input_ent=150, src="dev", max_length = [50, 10, 10], force_new=False, tokenizer = None, mode=0)
    device = torch.device("cuda:0")
    config_name = "../../../TURL/configs/table-base-config_v2.json"
    ckpt_path = "../../pytorch_model.bin"
    turl = load_turl_model(config_name, ckpt_path, device)
    attr2schemas, cooccurence = get_topk_schemas_and_attr_cooccurence(data_dir, k)
    attr_emb = turl_attribute_embedding_inference(dataset, turl, attr2schemas, device)
    corr = calculate_spearman_correlation(attr_emb, cooccurence)
    print(corr)


if __name__ == "__main__":
    data_dir = '../../../'
    k = 50
    test_turl(data_dir, k)