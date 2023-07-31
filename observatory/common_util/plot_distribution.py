import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import os


def plot_cosine_similarity_distribution(
    read_folders, labels, save_folder, picture_name
):
    data = []
    saved_data = {}
    for read_folder, label in zip(read_folders, labels):

        # Collect all the cosine similarities
        all_cosine_similarities = []
        for file in os.listdir(read_folder):
            if file.endswith("_results.pt"):
                cosine_similarities_dict = torch.load(os.path.join(read_folder, file))
                for column_index, similarities in cosine_similarities_dict.items():
                    all_cosine_similarities.extend(similarities)
        saved_data[label] = all_cosine_similarities
        data.append((all_cosine_similarities, label))

    # Plot boxplots for each dataset
    fig, ax = plt.subplots()
    ax.boxplot([item[0] for item in data], labels=[item[1] for item in data])
    # plt.title('Boxplot of Cosine Similarities')
    # plt.xlabel('Labels')
    plt.ylabel("Cosine Similarity")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save the data
    torch.save(saved_data, os.path.join(save_folder, "data.pt"))
    # Save the figure
    try:
        plt.savefig(os.path.join(save_folder, picture_name))
        plt.show()
    except Exception as e:
        print(f"Error saving figure: {e}")


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
