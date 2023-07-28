import os

import torch

import numpy as np
import argparse

import matplotlib.pyplot as plt


def process_directory(dir_path):
    result_pairs = []

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pt"):
            results = torch.load(os.path.join(dir_path, file_name))

            # Iterate through the avg_cosine_similarities and mcvs simultaneously
            for cos_sim, mcv in zip(
                results["avg_cosine_similarities"], results["mcvs"]
            ):
                # Check if both are not NaN before appending
                if not np.isnan(cos_sim) and not np.isnan(mcv):
                    result_pairs.append((cos_sim, mcv))

    return result_pairs


import json


def plot_result(
    directories,
    labels,
    avg_cosine_similarities_file,
    mcvs_file,
    plot_file,
    result_pairs_file,
):

    result_pairs = {}

    for dir_path, label in zip(directories, labels):
        pairs = process_directory(dir_path)

        # Create dictionary entries for each label
        result_pairs[label] = pairs

    # Save dictionaries to json files
    with open(result_pairs_file, "w") as f:
        json.dump(result_pairs, f)

    # Continue with the plot as before
    fig, axs = plt.subplots(2)

    # Calculate means and standard deviations for cosine similarities and mcvs
    means_cosine = [
        np.mean([pair[0] for pair in data]) for data in result_pairs.values()
    ]
    std_cosine = [np.std([pair[0] for pair in data]) for data in result_pairs.values()]

    means_mcvs = [np.mean([pair[1] for pair in data]) for data in result_pairs.values()]
    std_mcvs = [np.std([pair[1] for pair in data]) for data in result_pairs.values()]

    with open(avg_cosine_similarities_file, "w") as f:
        for label, item in zip(labels, means_cosine):
            f.write(f"{label}: {np.mean(item)}\n")

    with open(mcvs_file, "w") as f:
        for label, item in zip(labels, means_mcvs):
            f.write(f"{label}: {np.mean(item)}\n")
    axs[0].errorbar(labels, means_cosine, yerr=std_cosine, fmt="o")
    axs[0].set_title("Average Cosine Similarities")

    axs[1].errorbar(labels, means_mcvs, yerr=std_mcvs, fmt="o")
    axs[1].set_title("MCVS")

    # Save the plot to a file
    plt.savefig(plot_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process .pt files in directories.")

    parser.add_argument(
        "--dirs", nargs="+", required=True, help="Directories to process."
    )

    parser.add_argument(
        "--labels", nargs="+", required=True, help="Labels for the directories."
    )

    parser.add_argument(
        "--avg_cosine_similarities_file",
        required=True,
        help="File to save average cosine similarities.",
    )

    parser.add_argument("--mcvs_file", required=True, help="File to save mcvs.")

    parser.add_argument("--plot_file", required=True, help="File to save the plot.")

    parser.add_argument(
        "--result_pairs_file", required=True, help="File to save the result pairs."
    )

    args = parser.parse_args()

    if len(args.dirs) != len(args.labels):

        print("Number of directories and labels should be equal!")

        exit(1)

    plot_result(
        args.dirs,
        args.labels,
        args.avg_cosine_similarities_file,
        args.mcvs_file,
        args.plot_file,
        args.result_pairs_file,
    )
