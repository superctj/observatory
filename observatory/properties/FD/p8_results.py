import argparse
import torch
import numpy as np

def compute_avg_variance(path):
    list_pairs_norms_dict = torch.load(f"{path}/list_pairs_norms_dict.pt")
    list_non_pairs_norms_dict = torch.load(f"{path}/list_non_pairs_norms_dict.pt")
    pairs_variances = []
    non_pairs_variances = []

    for norms_dict in list_pairs_norms_dict:
        for l2_norms in norms_dict.values():
            if l2_norms:
                variance = np.var(l2_norms)
                pairs_variances.append(variance)

    for norms_dict in list_non_pairs_norms_dict:
        for l2_norms in norms_dict.values():
            if l2_norms:
                variance = np.var(l2_norms)
                non_pairs_variances.append(variance)

    avg_variance_pairs = np.mean(pairs_variances)
    avg_variance_non_pairs = np.mean(non_pairs_variances)

    return avg_variance_pairs, avg_variance_non_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute average variances of L2 norms.')
    parser.add_argument('--folders', nargs='+', required=True, help='List of folder paths')
    parser.add_argument('--labels', nargs='+', required=True, help='List of corresponding labels')
    parser.add_argument('--output', default='results.txt', help='Output file name')

    args = parser.parse_args()

    assert len(args.folders) == len(args.labels), "Number of folders and labels must be equal"

    with open(args.output, 'w') as f:
        for folder, label in zip(args.folders, args.labels):
            print(f"Processing label: {label}")
            avg_variance_pairs, avg_variance_non_pairs = compute_avg_variance(folder)
            f.write(f"Label: {label}, Avg variance pairs: {avg_variance_pairs}, Avg variance non-pairs: {avg_variance_non_pairs}\n")

