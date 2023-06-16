import argparse

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn import metrics


LABEL_ID_MAP = {"date": 0, "isbn": 1, "postalCode": 2, "price": 3, "weight": 4}


def compute_purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--embedding_file')
    parser.add_argument('-k', default=5)
    args = parser.parse_args()

    embeddings = torch.load(args.embedding_file)
    X = [item[0] for item in embeddings] # column embedding
    y = [LABEL_ID_MAP[item[1]] for item in embeddings] # label to ID

    kmeans = KMeans(n_clusters=args.k, random_state=12345, n_init="auto").fit(X)
    y_pred = kmeans.labels_

    purity_score = compute_purity_score(y, y_pred)
    