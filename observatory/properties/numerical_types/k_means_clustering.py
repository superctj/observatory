import argparse

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn import metrics


NON_TEXT_TYPE_ID_MAP = {"numberOfPages": 0, "isbn": 1, "url": 2, "ratingValue": 3, "price": 4, "date": 5, "telephone": 6, "postalCode": 7, "faxNumber": 8, "weight": 9}

TEXT_TYPE_ID_MAP = {"language": 0, "currency": 1, "organization": 2, "review": 3, "personName": 4, "musicAlbum": 5, "jobTitle": 6, "country": 7, "streetAddress": 8, "addressRegion": 9}


def compute_purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

def analyze(embeddings):
    X = [item[0].detach().cpu().numpy() for item in embeddings] # column embedding
    y = [LABEL_ID_MAP[item[1]] for item in embeddings] # label to ID
    kmeans = KMeans(n_clusters=len(LABEL_ID_MAP), random_state=12345, n_init="auto").fit(X)
    y_pred = kmeans.labels_
    purity_score = compute_purity_score(y, y_pred)
    return purity_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--embedding_file')
    parser.add_argument('--label')
    args = parser.parse_args()
    

    data = torch.load(args.embedding_file)
    col_itself = data["col_itself"] 
    subj_col_as_context = data["subj_col_as_context"] 
    neighbor_col_as_context = data["neighbor_col_as_context"] 
    entire_table_as_context = data["entire_table_as_context"] 
    print("Evaluate for: ", args.label )
    print()

    purity_score = analyze(col_itself)
    print("Scores for: ", "col_itself" )
    print(purity_score)
    print()
    purity_score = analyze(subj_col_as_context)
    print("Scores for: ", "subj_col_as_context" )
    print(purity_score)
    print()
    purity_score = analyze(neighbor_col_as_context)
    print("Scores for: ", "neighbor_col_as_context" )
    print(purity_score)
    print()
    purity_score = analyze(entire_table_as_context)
    print("Scores for: ", "entire_table_as_context" )
    print(purity_score)
    print()
    