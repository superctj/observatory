import os

import torch

import numpy as np
import argparse

import matplotlib.pyplot as plt


def process_directory(dir_path):
    avg_cosine_similarities = []
    mcvs = []
    nan_count = 0

    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pt"):
            results = torch.load(os.path.join(dir_path, file_name))
            for val in results["avg_cosine_similarities"]:
                if not np.isnan(val):
                    avg_cosine_similarities.append(val)
                else:
                    nan_count += 1
            for val in results["mcvs"]:
                if not np.isnan(val):
                    mcvs.append(val)
                else:
                    nan_count += 1

    print(f'Total NaN values in directory {dir_path}: {nan_count}')
    return avg_cosine_similarities, mcvs


def plot_result(directories, labels, avg_cosine_similarities_file, mcvs_file, plot_file):

    avg_cosine_similarities_result = []

    mcvs_result = []


    for dir_path in directories:

        avg_cosine_similarities, mcvs = process_directory(dir_path)
        avg_cosine_similarities_result.append(avg_cosine_similarities)
        mcvs_result.append(mcvs)


    # Write avg_cosine_similarities and mcvs to file
    with open(avg_cosine_similarities_file, "w") as f:
        for label, item in zip(labels, avg_cosine_similarities_result):
            f.write(f"{label}: {np.mean(item)}\n")

    with open(mcvs_file, "w") as f:
        for label, item in zip(labels, mcvs_result):
            f.write(f"{label}: {np.mean(item)}\n")


    # Plot the results with error bars

    fig, axs = plt.subplots(2)


    # Calculate means and standard deviations

    means_cosine = [np.mean(data) for data in avg_cosine_similarities_result]

    std_cosine = [np.std(data) for data in avg_cosine_similarities_result]

    means_mcvs = [np.mean(data) for data in mcvs_result]

    std_mcvs = [np.std(data) for data in mcvs_result]


    axs[0].errorbar(labels, means_cosine, yerr=std_cosine, fmt='o')

    axs[0].set_title('Average Cosine Similarities')


    axs[1].errorbar(labels, means_mcvs, yerr=std_mcvs, fmt='o')

    axs[1].set_title('MCVS')


    # Save the plot to a file
    plt.savefig(plot_file)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process .pt files in directories.')

    parser.add_argument('--dirs', nargs='+', required=True, help='Directories to process.')

    parser.add_argument('--labels', nargs='+', required=True, help='Labels for the directories.')

    parser.add_argument('--avg_cosine_similarities_file', required=True, help='File to save average cosine similarities.')

    parser.add_argument('--mcvs_file', required=True, help='File to save mcvs.')

    parser.add_argument('--plot_file', required=True, help='File to save the plot.')

    args = parser.parse_args()
    

    if len(args.dirs) != len(args.labels):

        print("Number of directories and labels should be equal!")

        exit(1)

    plot_result(args.dirs, args.labels, args.avg_cosine_similarities_file, args.mcvs_file, args.plot_file)

