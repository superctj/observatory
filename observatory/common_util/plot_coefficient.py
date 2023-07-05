import argparse
import torch
from scipy import stats
import os
import matplotlib.pyplot as plt
import numpy as np

def calculate_spearman(file_path):
    # Load the dictionary from the .pt file
    results = torch.load(file_path)

    # Extract values for each metric
    cosine_similarities = [result['cosine_similarity'] for result in results]
    containments = [result['containment'] for result in results]
    jaccard_similarities = [result['jaccard_similarity'] for result in results]
    multiset_jaccard_similarities = [result['multiset_jaccard_similarity'] for result in results]

    # Calculate Spearman's coefficient and the p-value for each metric
    coef_containment, p_containment = stats.spearmanr(cosine_similarities, containments)
    coef_jaccard, p_jaccard = stats.spearmanr(cosine_similarities, jaccard_similarities)
    coef_multiset_jaccard, p_multiset_jaccard = stats.spearmanr(cosine_similarities, multiset_jaccard_similarities)

    print(f"File: {file_path}")
    print("Spearman's coefficients: ", coef_containment, coef_jaccard, coef_multiset_jaccard)
    print("p-values: ", p_containment, p_jaccard, p_multiset_jaccard)
    print("-----------------------------")

    return {'containment': (coef_containment, p_containment), 
            'jaccard_similarity': (coef_jaccard, p_jaccard), 
            'multiset_jaccard_similarity': (coef_multiset_jaccard, p_multiset_jaccard)}

def save_results_and_plot(results, labels, save_directory):
    # Make sure save directory exists, if not, create it
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    metrics = list(results[0].keys())

    # Save results to a text file
    with open(os.path.join(save_directory, 'results.txt'), 'w') as f:
        for metric in metrics:
            for i, result in enumerate(results):
                coef, p = result[metric]
                f.write(f"File: {labels[i]}, Metric: {metric}\nSpearman's coefficient: {coef}\np-value: {p}\n-----------------------------\n")

    # Create bar charts
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10,5))

        coefs = [result[metric][0] for result in results]
        ax.bar(labels, coefs, color='skyblue')
        ax.set_title(f"Spearman's coefficients for {metric}")
        ax.set_ylim([0, 1])
        # Save the chart as a PNG image
        plt.savefig(os.path.join(save_directory, f"{metric}_coefficients.png"))

        fig, ax = plt.subplots(figsize=(10,5))

        ps = [result[metric][1] for result in results]
        ax.bar(labels, ps, color='lightgreen')
        ax.set_title(f'p-values for {metric}')
        ax.set_ylim([0, 1])
        # Save the chart as a PNG image
        plt.savefig(os.path.join(save_directory, f"{metric}_p_values.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some .pt files.')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                        help='the paths to the result.pt files, space separated')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                        help='the labels for result.pt files')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='the directory where to save results and plots')

    args = parser.parse_args()

    results = []
    for file_path in args.files:
        result = calculate_spearman(file_path)
        results.append(result)

    save_results_and_plot(results, args.labels, args.save_dir)
