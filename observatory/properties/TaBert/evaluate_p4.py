#  python create_row_shuffle_embeddings.py -r normal_TD  -s RI_bert_TD -m bert-base-uncased -n 1000
import os
import argparse
import pandas as pd
import torch
from readCompare import compare_directories
# from concurrent.futures import ThreadPoolExecutor
from torch.linalg import inv, norm
from mcv import compute_mcv
import random
import math
import itertools

from table_bert import Table, Column
from table_bert import TableBertModel
def convert_to_table(df, tokenizer):

    header = []
    data = []

    for col in df.columns:
        try:
            # Remove commas and attempt to convert to float
            val = float(str(df[col].iloc[0]).replace(',', ''))
            # If conversion is successful, it's a real column
            col_type = 'real'
            sample_value = df[col][0]
        except (ValueError, AttributeError):
            # If conversion fails, it's a text column
            col_type = 'text'
            sample_value = df[col][0]

        # Create a Column object
        header.append(Column(col, col_type, sample_value=sample_value))
        
        # Add the column data to 'data' list
    for row_index in range(len(df)):
        data.append(list(df.iloc[row_index]))
        # print()
        # print(col_type)
        # print(sample_value)
    # Create the Table
    table = Table(id='', header=header, data=data)

    # Tokenize
    table.tokenize(tokenizer)

    return table




def generate_p4_embeddings( model, device, tables):
    all_shuffled_embeddings = []
    # sampled_tables = shuffle_df(table,num_samples, percentage )
    
    for processed_table in tables:
    # for j in range(num_samples + 1):
    #     if j == 0:
    #         processed_table = table
    #     else:
    #         processed_table = sample_rows(table, percentage)
        processed_table = processed_table.reset_index(drop=True)

        processed_table = processed_table.astype(str)
        processed_table = convert_to_table(processed_table, model.tokenizer)
        context = ''
        with torch.no_grad():
            context_encoding, column_encoding, info_dict = model.encode(
                contexts=[model.tokenizer.tokenize(context)],
                tables=[processed_table]
            )
        embeddings = column_encoding[0]
        all_shuffled_embeddings.append(embeddings)
        # Free up some memory by deleting column_encoding and info_dict variables
        del column_encoding
        del info_dict
        del context_encoding
        del embeddings
        # Empty the cache
        torch.cuda.empty_cache()

    return all_shuffled_embeddings

def analyze_embeddings(all_shuffled_embeddings, changed_column_lists):
    cosine_similarities_dict = {}

    for table_index, changed_columns in enumerate(changed_column_lists):
        for column_index in changed_columns:
            original_embedding = all_shuffled_embeddings[0][column_index]
            shuffled_embedding = all_shuffled_embeddings[table_index+1][column_index]

            cosine_similarity = torch.dot(original_embedding, shuffled_embedding) / (norm(original_embedding) * norm(shuffled_embedding))
            
            if column_index not in cosine_similarities_dict:
                cosine_similarities_dict[column_index] = []

            cosine_similarities_dict[column_index].append(cosine_similarity.item())

    return cosine_similarities_dict

def process_table_wrapper(tables, args, model_name, model,device,key, changed_column_list):
    save_directory_results  = os.path.join('/nfs/turbo/coe-jag/zjsun', 'p4', args.save_directory, model_name ,'results')
    save_directory_embeddings  = os.path.join('/nfs/turbo/coe-jag/zjsun','p4',  args.save_directory, model_name ,'embeddings')
    # save_directory_results  = os.path.join(  args.save_directory, model_name ,'results')
    # save_directory_embeddings  = os.path.join(args.save_directory, model_name ,'embeddings')
    # Create the directories if they don't exist
    if not os.path.exists(save_directory_embeddings):
        os.makedirs(save_directory_embeddings)
    if not os.path.exists(save_directory_results):
        os.makedirs(save_directory_results)


    all_shuffled_embeddings = generate_p4_embeddings( model, device, tables)
    torch.save(all_shuffled_embeddings, os.path.join(save_directory_embeddings, f"{key}_embeddings.pt"))
    cosine_similarities_dict = analyze_embeddings(all_shuffled_embeddings, changed_column_list)
    for column_index, similarities in cosine_similarities_dict.items():
        print(f"Column {column_index}:")
        for i, similarity in enumerate(similarities):
            print(f"\tCosine similarity with modified table {i+1}: {similarity}")
    torch.save(cosine_similarities_dict, os.path.join(save_directory_results, f"{key}_results.pt"))
    

def process_and_save_embeddings(model_name, args, result_dict):
    device = torch.device("cuda")
    print()
    print(device)
    print()

    model = TableBertModel.from_pretrained(
        '/home/zjsun/TaBert/TaBERT/tabert_base_k3/model.bin',
    )
    model = model.to(device)
    model.eval()
    
    for key, value in result_dict.items():
        # truncated_tables = []
        # list_max_rows_fit = []
        # for table in value[0]:
        #     max_rows_fit = truncate_index(table, tokenizer, max_length, model_name)
        #     list_max_rows_fit.append(max_rows_fit)
        # min_max_rows_fit = min(list_max_rows_fit)
        
        # for table in value[0]:
        #     truncated_table = table.iloc[:min_max_rows_fit, :]
        #     truncated_tables.append(truncated_table)
        process_table_wrapper(value[0], args, model_name, model, device, key,value[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process tables and save embeddings.')
    parser.add_argument('-o', '--original_directory', type=str, required=True, help='Directory of the original tables.')
    parser.add_argument('-c', '--changed_directory', type=str, required=True, help='Directory of the modified tables.')
    parser.add_argument('-s', '--save_directory', type=str, required=True, help='Directory to save embeddings to')
    parser.add_argument('-m', '--model_name', type=str, default="", help='Name of the Hugging Face model to use')
    args = parser.parse_args()
    
    result_dict = compare_directories(args.original_directory, args.changed_directory)



    model_name = args.model_name

    print()
    print("Evaluate row shuffle for: ",model_name)
    print()

    
    process_and_save_embeddings(model_name, args, result_dict)


