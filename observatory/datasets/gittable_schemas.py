import pyarrow.parquet as pq
import pandas as pd
import json
import requests
import os
import pickle
import torch
from observatory.metrics.comparison import jaccard

def get_gittable_schema_count(ACCESS_TOKEN, data_dir, jaccard_threshold):
    """
        Download all the gittables and save a pickle file for the schema counts.
        If two schemas have a high jaccard similarity, we treat them as the same schema.
        We do not double count the same schemas in the same repo.
    """
    r = requests.get('https://zenodo.org/api/records/6517052', 
        params={'access_token': ACCESS_TOKEN})
    download_urls = [f['links']['self'] for f in r.json()['files']]
    filenames = [f['key'] for f in r.json()['files']]
    for filename, url in zip(filenames, download_urls):
        print("Downloading:", filename)
        filename = os.path.join(data_dir, filename)
        r = requests.get(url, params={'access_token': ACCESS_TOKEN})
        with open(filename, 'wb') as f:
            f.write(r.content)

    schemas = {}
    for f in os.listdir(data_dir):
        if f.endswith('.parquet'):
            found = False
            try:
                table = pq.read_table(os.path.join(data_dir, f))
            except:
                print('read failure')
            url = json.loads(table.schema.metadata['gittables'.encode()].decode('utf-8'))['csv_url']
            user_repo = url.split('/')[3] + '/' + url.split('/')[4] # extract the username and repo name
            curr_schema = tuple(map(str.lower, table.column_names))
            for key in schemas.keys():
                if jaccard(curr_schema, key) > jaccard_threshold:
                    schemas[key].add(user_repo)
                    found = True
                    break
            if not found:
                schemas[curr_schema] = set([user_repo])

    for k, v in schemas.items():
        schemas[k] = len(v)
    print('num of schemas: ')
    print(len(schemas))
    with open(data_dir + 'schema_count.pkl', 'wb') as f:
        pickle.dump(schemas, f)
        
        
def build_lm_input(headers, tokenizer):
    """
        Build the input to be fed into BERT/ROBERTA.
        Also return a dictionary that maps attributes to their starting and ending positions so that we can extract the embedding for each attribute later.
    """
    header_str = ' '.join(headers)
    lm_input = tokenizer(header_str, return_tensors="pt", truncation=True)
    attr2pos = {}
    last_idx = 0
    for i in range(len(headers)):
        tokenized_header_length = len(tokenizer.tokenize(headers[i]))
        attr2pos[headers[i]] = (last_idx+1, last_idx+tokenized_header_length+1)
        last_idx = last_idx+tokenized_header_length+1
        if last_idx >= 511:
            break

    return lm_input, attr2pos


def build_t5_input(headers, tokenizer):
    """
        Build the input to be fed into T5.
        Also return a dictionary that maps attributes to their starting and ending positions so that we can extract the embedding for each attribute later.
    """
    header_str = ' '.join(headers)
    t5_input = tokenizer(header_str, return_tensors="pt", truncation=True)
    attr2pos = {}
    last_idx = 0
    for i in range(len(headers)):
        tokenized_header_length = len(tokenizer.tokenize(headers[i]))
        attr2pos[headers[i]] = (last_idx, last_idx+tokenized_header_length)
        last_idx = last_idx+tokenized_header_length
        if last_idx >= 511:
            break

    return t5_input, attr2pos


def build_turl_input(headers, config):
    """
        Build the inputs to be fed into TURL.
        Also return a dictionary that maps attributes to their starting and ending positions so that we can extract the embedding for each attribute later.
    """
    tokenized_headers = [config.tokenizer.encode(header, max_length=config.max_header_length, add_special_tokens=False) for header in headers]
    input_tok = []
    input_tok_pos = []
    input_tok_type = []
    attr2pos = {}
    for i in range(len(tokenized_headers)):
        tokenized_header = tokenized_headers[i]
        tokenized_header_length = len(tokenized_header)
        attr2pos[headers[i]] = (len(input_tok), len(input_tok)+tokenized_header_length)
        input_tok += tokenized_header
        input_tok_pos += list(range(tokenized_header_length))
        input_tok_type += [1]*tokenized_header_length

    input_tok = torch.LongTensor([input_tok])
    input_tok_type = torch.LongTensor([input_tok_type])
    input_tok_pos = torch.LongTensor([input_tok_pos])

    return input_tok, input_tok_type, input_tok_pos, attr2pos


def build_tapas_input(headers):
    """
        Build the input table to be fed into TAPAS.
        The attribute would directly be the query to TAPAS, so we do not need attr2pos here.
    """
    header_str = ' '.join(headers)
    table_dict = {'headers': [header_str]}
    pandas_table = pd.DataFrame.from_dict(table_dict)

    return pandas_table


if __name__ == "__main__":
    ACCESS_TOKEN = 'nBM6EEUv1uq0N8XBux3cQLuWEf1gjMyTLHyxsAaDWyn5Ow0AAgxPacYLZsHE'
    data_dir = '../data/gittables/'
    jaccard_threshold = 0.8
    get_gittable_schema_count(ACCESS_TOKEN, data_dir, jaccard_threshold)
