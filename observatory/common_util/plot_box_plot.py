import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json


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


def plot_cosine_similarity_distribution(
    read_folders, labels, save_folder, picture_name
):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    data = {}
    mean_values = {}

    for read_folder, label in zip(read_folders, labels):
        result_pairs = process_directory(read_folder)
        cosine_similarities = [pair[0] for pair in result_pairs]
        mcvs = [pair[1] for pair in result_pairs]
        data[label] = (cosine_similarities, mcvs)

        # Calculating mean for both cosine_similarities and mcvs
        mean_cos_sim = np.mean(cosine_similarities)
        mean_mcv = np.mean(mcvs)
        mean_values[label] = (mean_cos_sim, mean_mcv)

    # Plot boxplot for Cosine Similarity
    fig1, ax1 = plt.subplots()
    ax1.boxplot(
        [value[0] for key, value in data.items()],
        labels=[key for key, value in data.items()],
    )
    ax1.set_ylabel("Cosine Similarity")
    plt.savefig(os.path.join(save_folder, f"cosine_{picture_name}"))
    plt.show()

    # Plot boxplot for mcv
    fig2, ax2 = plt.subplots()
    ax2.boxplot(
        [value[1] for key, value in data.items()],
        labels=[key for key, value in data.items()],
    )
    ax2.set_ylabel("mcv")
    plt.savefig(os.path.join(save_folder, f"mcv_{picture_name}"))
    plt.show()

    # Create the directories if they don't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the data
    torch.save(data, os.path.join(save_folder, "data.pt"))

    # Save mean values
    with open(os.path.join(save_folder, "mean_values.txt"), "w") as f:
        json.dump(mean_values, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot cosine similarity distribution")
    parser.add_argument(
        "--read_folders",
        type=str,
        nargs="+",
        required=True,
        help="The folders to read the results from",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="The labels corresponding to each read folder",
    )
    parser.add_argument(
        "--save_folder", type=str, required=True, help="The folder to save the plot in"
    )
    parser.add_argument(
        "--picture_name",
        type=str,
        required=True,
        help="The name of the picture to save",
    )

    args = parser.parse_args()

    plot_cosine_similarity_distribution(
        args.read_folders, args.labels, args.save_folder, args.picture_name
    )
