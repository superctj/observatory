# import argparse
# import torch
# import numpy as np
# from collections import defaultdict

# def compute_avg_variance(path, min_length, max_length):
#     list_pairs_norms_dict = torch.load(f"{path}/list_pairs_norms_dict.pt")
#     list_non_pairs_norms_dict = torch.load(f"{path}/list_non_pairs_norms_dict.pt")
#     pairs_variances = defaultdict(list)
#     non_pairs_variances = defaultdict(list)

#     for norms_dict in list_pairs_norms_dict:
#         for l2_norms in norms_dict.values():
#             length = len(l2_norms)
#             if min_length <= length <= max_length:
#                 variance = np.var(l2_norms)
#                 pairs_variances[length].append(variance)

#     for norms_dict in list_non_pairs_norms_dict:
#         for l2_norms in norms_dict.values():
#             length = len(l2_norms)
#             if min_length <= length <= max_length:
#                 variance = np.var(l2_norms)
#                 non_pairs_variances[length].append(variance)

#     avg_variance_pairs = {length: np.mean(variances) for length, variances in pairs_variances.items()}
#     avg_variance_non_pairs = {length: np.mean(variances) for length, variances in non_pairs_variances.items()}

#     return avg_variance_pairs, avg_variance_non_pairs

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description='Compute average variances of L2 norms.')
#     parser.add_argument('--folders', nargs='+', required=True, help='List of folder paths')
#     parser.add_argument('--labels', nargs='+', required=True, help='List of corresponding labels')
#     parser.add_argument('--output', default='results.txt', help='Output file name')
#     parser.add_argument('--min_length', type=int, default=2, help='Minimum list length to include in variance calculation')
#     parser.add_argument('--max_length', type=int, default=10, help='Maximum list length to include in variance calculation')

#     args = parser.parse_args()

#     assert len(args.folders) == len(args.labels), "Number of folders and labels must be equal"

#     with open(args.output, 'w') as f:
#         for folder, label in zip(args.folders, args.labels):
#             print(f"Processing label: {label}")
#             avg_variance_pairs, avg_variance_non_pairs = compute_avg_variance(folder, args.min_length, args.max_length)
#             f.write(f"Label: {label}\n")
#             for length in range(args.min_length, args.max_length + 1):
#                 f.write(f"\nLength {length}, \n Avg variance pairs: {avg_variance_pairs.get(length, 'N/A')},\n Avg variance non-pairs: {avg_variance_non_pairs.get(length, 'N/A')}\n")

















import argparse
import torch
import numpy as np

def compute_avg_variance(path, min_length):
    list_pairs_norms_dict = torch.load(f"{path}/list_pairs_norms_dict.pt")
    list_non_pairs_norms_dict = torch.load(f"{path}/list_non_pairs_norms_dict.pt")
    pairs_variances = []
    non_pairs_variances = []

    for norms_dict in list_pairs_norms_dict:
        for l2_norms in norms_dict.values():
            if l2_norms and len(l2_norms) >= min_length:
                variance = np.var(l2_norms)
                pairs_variances.append(variance)

    for norms_dict in list_non_pairs_norms_dict:
        for l2_norms in norms_dict.values():
            if l2_norms and len(l2_norms) >= min_length:
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
    parser.add_argument('--min_length', type=int, default=1, help='Minimum list length to include in variance calculation')

    args = parser.parse_args()

    assert len(args.folders) == len(args.labels), "Number of folders and labels must be equal"

    with open(args.output, 'w') as f:
        for folder, label in zip(args.folders, args.labels):
            print(f"Processing label: {label}")
            avg_variance_pairs, avg_variance_non_pairs = compute_avg_variance(folder, args.min_length)
            f.write(f"Label: {label}, \n Avg variance pairs: {avg_variance_pairs}, \n Avg variance non-pairs: {avg_variance_non_pairs}\n")
