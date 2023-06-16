import os
import torch
import argparse
from mcv import compute_mcv
from torch.linalg import inv, norm

def analyze_embeddings(all_shuffled_embeddings):
    avg_cosine_similarities = []
    mcvs = []

    for i in range(len(all_shuffled_embeddings[0])):
        column_cosine_similarities = []
        column_embeddings = []
        
        for j in range(len(all_shuffled_embeddings)):
            column_embeddings.append(all_shuffled_embeddings[j][i])

        for j in range(1, len(all_shuffled_embeddings)):
            truncated_embedding = all_shuffled_embeddings[0][i]
            shuffled_embedding = all_shuffled_embeddings[j][i]

            cosine_similarity = torch.dot(truncated_embedding, shuffled_embedding) / (norm(truncated_embedding) * norm(shuffled_embedding))
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = torch.mean(torch.tensor(column_cosine_similarities))
        mcv = compute_mcv(torch.stack(column_embeddings))

        avg_cosine_similarities.append(avg_cosine_similarity.item())
        mcvs.append(mcv)

    table_avg_cosine_similarity = torch.mean(torch.tensor(avg_cosine_similarities))
    table_avg_mcv = torch.mean(torch.tensor(mcvs))

    return avg_cosine_similarities, mcvs, table_avg_cosine_similarity.item(), table_avg_mcv.item()

def fix(parent_directory):
    # derive save directories for embeddings and results
    print("="*50)
    print()
    print(parent_directory)
    print()
    print("="*50)
    save_directory_embeddings = os.path.join(parent_directory, "embeddings")
    save_directory_results = os.path.join(parent_directory, "results")

    # list all files in the embeddings directory
    embedding_files = os.listdir(save_directory_embeddings)

    # iterate over all files
    for embedding_file in embedding_files:
        # check if the file is an embedding file
        if embedding_file.endswith("_embeddings.pt"):
            # extract the table_index from the file name
            table_index = embedding_file.replace("_embeddings.pt", "")
            # load the embeddings
            all_shuffled_embeddings = torch.load(os.path.join(save_directory_embeddings, embedding_file), map_location=torch.device('cpu'))

            if len(all_shuffled_embeddings) < 24:
                results = {
                    "avg_cosine_similarities": [],
                    "mcvs": [],
                    "table_avg_cosine_similarity": None,
                    "table_avg_mcv": None
                    }
                torch.save(results, os.path.join(save_directory_results, f"{table_index}_results.pt"))
                continue
                

            # apply the analysis to the loaded embeddings
            avg_cosine_similarities, mcvs, table_avg_cosine_similarity, table_avg_mcv = analyze_embeddings(all_shuffled_embeddings)

            # create a dictionary with the results
            results = {
                "avg_cosine_similarities": avg_cosine_similarities,
                "mcvs": mcvs,
                "table_avg_cosine_similarity": table_avg_cosine_similarity,
                "table_avg_mcv": table_avg_mcv
            }

            # print the results
            print(f"Table {table_index}:")
            print("Average Cosine Similarities:", results["avg_cosine_similarities"])
            print("MCVs:", results["mcvs"])
            print("Table Average Cosine Similarity:", results["table_avg_cosine_similarity"])
            print("Table Average MCV:", results["table_avg_mcv"])

            # save the results
            torch.save(results, os.path.join(save_directory_results, f"{table_index}_results.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze embeddings and save results.')
    parser.add_argument('--parent_directories', type=str, required=True, nargs='+', help='Space separated paths to the parent directories containing embeddings and results folders.')
    
    args = parser.parse_args()
    for parent_directory in args.parent_directories:
        fix(parent_directory)
        
