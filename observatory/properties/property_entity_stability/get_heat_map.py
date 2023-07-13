import os
import torch
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle

def load_embeddings_and_ids_from_directories(directory):
    embeddings_dict = {}
    entity_ids_dict = {}
    print(directory)
    for file in os.listdir(directory):
        if file.endswith('.pt'):
            count = int(file.split('_')[0][4:])  # Extract the count from filename
            data = torch.load(os.path.join(directory, file), map_location=torch.device('cpu'))
            for (row, column), value in data.items():
                cell_embedding, entity_id = value
                embeddings_dict[(count, row, column)] = cell_embedding
                entity_ids_dict[(count, row, column)] = entity_id
    return embeddings_dict, entity_ids_dict

def find_common_keys(dicts):
    common_keys = set(dicts[0].keys())
    for d in dicts[1:]:
        common_keys.intersection_update(d.keys())
    return list(common_keys)

def get_top_k_similar(query_embedding, embeddings, K):
    embeddings_list = list(embeddings.items())
    embeddings_keys = [key for key, _ in embeddings_list]
    embeddings_values = np.array([value.detach().cpu().numpy() for _, value in embeddings_list])

    similarities = cosine_similarity([query_embedding.detach().cpu().numpy()], embeddings_values)[0]
    top_k_indices = similarities.argsort()[-K:]

    return [embeddings_keys[index] for index in top_k_indices]

def multiset_jaccard_similarity(list1, list2):
    counter1 = Counter(list1)
    counter2 = Counter(list2)
    overlap = list((counter1 & counter2).elements())
    union = list((counter1 | counter2).elements())
    return 2 * len(overlap) / (len(union) + len(overlap))

import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directories",type=str, nargs='+', help="List of directories")
    parser.add_argument("--labels", type=str, nargs='+', help="Corresponding labels")
    parser.add_argument("--save_dir", type=str, help="Directory to save the figures")
    parser.add_argument("--K_values", nargs='+', type=int, help="List of K values for top-K similar cells")
    parser.add_argument("--if_only_plot", type=str, default= "False",  help="Whether only do the plots")
    parser.add_argument("--if_double_entity", type=str, default= "False",  help="Whether fix the entity similarity")

    args = parser.parse_args()
    directories = args.directories  # List of directories
    # print(directories)
    save_directory = args.save_dir
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    labels = args.labels  # Corresponding labels
    K_values = args.K_values  # List of K values
    if args.if_only_plot == "True":
        for K in K_values:
            with open(os.path.join(save_directory, f"overlaps_dict_K{K}.pkl"), "rb") as f:
                overlaps_dict = pickle.load(f)
            with open(os.path.join(save_directory, f"entity_similarity_dict_K{K}.pkl"), "rb") as f:
                entity_similarity_dict = pickle.load(f)
            if args.if_double_entity == "True":
                new_entity_similarity_dict = {k: v * 2.0 for k, v in entity_similarity_dict.items()}
                entity_similarity_dict = new_entity_similarity_dict
                with open(os.path.join(save_directory, f"entity_similarity_dict_K{K}.pkl"), "wb") as f:
                    pickle.dump(entity_similarity_dict, f)


            position_overlaps = np.mean(list(overlaps_dict.values()), axis=0) 
            entity_similarity = np.mean(list(entity_similarity_dict.values()), axis=0) 
            # 4. Plot heatmaps
            plt.figure(figsize=(10, 10))
            sns.heatmap(position_overlaps, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
            plt.title(f'Overlap for K={K}')
            plt.savefig(os.path.join(save_directory, f'position_heatmap_K{K}.png'))

            plt.figure(figsize=(10, 10))
            sns.heatmap(entity_similarity, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
            plt.title(f'Entity Overlap for K={K}')
            plt.savefig(os.path.join(save_directory, f'entity_heatmap_K{K}.png'))
        sys.exit()
    # 1. Load all embeddings and entity ids
    all_data = [load_embeddings_and_ids_from_directories(dir) for dir in directories]
    all_embeddings = [data[0] for data in all_data]
    all_entity_ids = [data[1] for data in all_data]

    # 2. Find common keys and randomly select 1000 keys
    common_keys = find_common_keys(all_embeddings)
    print("number of common_keys: ", len(common_keys))
    selected_keys = random.sample(common_keys, 1000)
    with open(os.path.join(save_directory, "selected_keys.pkl"), "wb") as f:
        pickle.dump(selected_keys, f)
    # 3. Compute overlaps and similarities for each pair of labels
    # Start by finding the maximum K value. This will be used to calculate similarities once,
    # and then smaller K values will be subsets of these similarities.
    max_K = max(K_values)

    # Initialize dictionaries to hold overlaps and entity similarity data for all K values.
    overlaps_dict_all_K = {K: {} for K in K_values}
    entity_similarity_dict_all_K = {K: {} for K in K_values}

    # Loop over each key in selected_keys.
    for key in selected_keys:
        key_similarities_all_K = []
        key_entity_ids_all_K = []
        # For each key, calculate the top max_K similar keys. These will be stored and subset for each K value.
        for embeddings, entity_ids in zip(all_embeddings, all_entity_ids):
            similar_keys = get_top_k_similar(embeddings[key], embeddings, max_K)
            key_similarities_all_K.append({K: similar_keys[-K:] for K in K_values})
            key_entity_ids_all_K.append({K: [entity_ids[k] for k in similar_keys[-K:]] for K in K_values})
        
        # Loop over each K value to calculate overlap and entity similarity.
        for K in K_values:
            overlaps_key = np.zeros((len(labels), len(labels)))
            entity_similarity_key = np.zeros((len(labels), len(labels)))
            for i in range(len(labels)):
                overlaps_key[i, i] = 1
                entity_similarity_key[i, i] = 0.5
                for j in range(i + 1, len(labels)):
                    overlap_count = len(set(key_similarities_all_K[i][K]) & set(key_similarities_all_K[j][K]))
                    overlaps_key[i, j] = overlap_count / K
                    overlaps_key[j, i] = overlap_count / K
                    entity_overlap = multiset_jaccard_similarity(key_entity_ids_all_K[i][K], key_entity_ids_all_K[j][K])
                    entity_similarity_key[i, j] = entity_overlap
                    entity_similarity_key[j, i] = entity_overlap
            overlaps_dict_all_K[K][key] = overlaps_key
            entity_similarity_dict_all_K[K][key] = entity_similarity_key

    # Finally, write the overlaps and entity similarity data to pickle files for each K value.
    for K in K_values:
        with open(os.path.join(save_directory, f"overlaps_dict_K{K}.pkl"), "wb") as f:
            pickle.dump(overlaps_dict_all_K[K], f)
        with open(os.path.join(save_directory, f"entity_similarity_dict_K{K}.pkl"), "wb") as f:
            pickle.dump(entity_similarity_dict_all_K[K], f)


        position_overlaps = np.mean(list(overlaps_dict_all_K[K].values()), axis=0)
        entity_similarity = np.mean(list(entity_similarity_dict_all_K[K].values()), axis=0)
        # 4. Plot heatmaps
        plt.figure(figsize=(10, 10))
        sns.heatmap(position_overlaps, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.title(f'Overlap for K={K}')
        plt.savefig(os.path.join(save_directory, f'position_heatmap_K{K}.png'))

        plt.figure(figsize=(10, 10))
        sns.heatmap(entity_similarity, annot=True, fmt=".2f", cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
        plt.title(f'Entity Overlap for K={K}')
        plt.savefig(os.path.join(save_directory, f'entity_heatmap_K{K}.png'))
