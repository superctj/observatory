import argparse
import torch
from scipy import stats
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_spearman(file_path):
    # Load the tensor from the .pt file
    pairs = torch.load(file_path)

    # If pairs is a tensor of shape (N, 2), we can convert it to a list of tuples.
    pairs_list = [tuple(x) for x in pairs]

    # Then we separate the pairs into two lists
    list1, list2 = zip(*pairs_list)

    # Then we calculate Spearman's coefficient and the p-value
    coef, p = stats.spearmanr(list1, list2)

    print(f"File: {file_path}")
    print("Spearman's coefficient: ", coef)
    print("p-value: ", p)
    print("-----------------------------")

    return coef, p

def save_results_and_plot(coefs, ps, labels, save_directory):
    # Make sure save directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save results to a text file
    with open(os.path.join(save_directory, 'results.txt'), 'w') as f:
        for label, coef, p in zip(labels, coefs, ps):
            f.write(f"Label: {label}\nSpearman's coefficient: {coef}\np-value: {p}\n-----------------------------\n")

    # Create bar charts
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(labels, coefs, color='skyblue')
    ax.set_title("Spearman's coefficients")
    ax.set_ylim([0, 1])
    # Save the chart as a PNG image
    plt.savefig(os.path.join(save_directory, "coefficients.png"))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(labels, ps, color='lightgreen')
    ax.set_title('p-values')
    ax.set_ylim([0, 1])
    # Save the chart as a PNG image
    plt.savefig(os.path.join(save_directory, "p_values.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some .pt files.')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='the paths to the result.pt files, space separated')
    parser.add_argument('--labels', type=str, nargs='+',
                        help='the labels for result.pt files')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='the directory where to save results and plots')

    args = parser.parse_args()

    coefs = []
    ps = []
    for file_path in args.files:
        coef, p = calculate_spearman(file_path)
        coefs.append(coef)
        ps.append(p)

    save_results_and_plot(coefs, ps, args.labels, args.save_dir)
