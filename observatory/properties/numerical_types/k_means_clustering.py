import argparse

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn import metrics




def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--embedding_file')
    args = parser.parse_args()

    embeddings = torch.load(args.embedding_file)
    X = [item[0] for item in embeddings]
    y = [item[1] for item in embeddings]

    kmeans = KMeans(n_clusters=5, random_state=12345, n_init="auto").fit(X)
    y_pred = kmeans.predict(X)

    