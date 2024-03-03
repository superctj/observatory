import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="The directory where the results and embeddings subfolders are located",  # noqa: 501
    )

    parser.add_argument(
        "--mcv_threshold",
        type=float,
        required=True,
        help="Minimum mean column value (mcv) threshold",
    )

    parser.add_argument(
        "--cosine_similarity_threshold",
        type=float,
        required=True,
        help="Minimum cosine similarity threshold",
    )

    parser.add_argument(
        "--n_components",
        type=int,
        default=2,
        help="Number of dimensions for t-SNE visualization",
    )

    args = parser.parse_args()

    # Define the directories
    results_dir = os.path.join(args.root_dir, "results")
    embeddings_dir = os.path.join(args.root_dir, "embeddings")
    save_dir = os.path.join(args.root_dir, "save")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Iterate over each result file
    for result_file in os.listdir(results_dir):
        if result_file.endswith("_results.pt"):
            # Load results
            results = torch.load(os.path.join(results_dir, result_file))

            # Find indices of columns that meet the criteria
            indices = [
                i
                for i, (cos_sim, mcv) in enumerate(
                    zip(results["avg_cosine_similarities"], results["mcvs"])
                )
                if cos_sim > args.cosine_similarity_threshold
                and mcv > args.mcv_threshold
            ]

            # If any columns meet the criteria, load the corresponding
            # embeddings
            if indices:
                embedding_file = result_file.replace("results", "embeddings")
                all_shuffled_embeddings = torch.load(
                    os.path.join(embeddings_dir, embedding_file)
                )

                # Collect the corresponding column embeddings
                for index in indices:
                    print()
                    print(index)
                    column_embeddings = []

                    for i in range(len(all_shuffled_embeddings[0])):
                        print(all_shuffled_embeddings[i][index].shape)
                        column_embeddings.append(
                            all_shuffled_embeddings[i][index].numpy()
                        )

                    column_embeddings = np.array(column_embeddings)

                    if len(column_embeddings) < 31:
                        continue

                    # Perform t-SNE
                    tsne = TSNE(n_components=args.n_components)
                    tsne_embeddings = tsne.fit_transform(column_embeddings)

                    # Plot and save each t-SNE embedding
                    plt.figure()

                    if args.n_components == 2:
                        plt.scatter(
                            tsne_embeddings[:, 0], tsne_embeddings[:, 1]
                        )
                    elif args.n_components == 3:
                        ax = plt.figure().add_subplot(111, projection="3d")
                        ax.scatter(
                            tsne_embeddings[:, 0],
                            tsne_embeddings[:, 1],
                            tsne_embeddings[:, 2],
                        )

                    plt.title(
                        f"Table {result_file.split('_')[1]} Column {index}"
                    )

                    save_path = os.path.join(
                        save_dir,
                        f"table_{result_file.split('_')[1]}_column_{index}.png",
                    )
                    plt.savefig(save_path)
                    plt.close()
