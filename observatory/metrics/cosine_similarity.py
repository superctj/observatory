import os

import numpy as np

from scipy.spatial.distance import cosine


def cosine_similarity(a, b):
    return 1 - cosine(a, b)


def main(args):
    dir_a_files = sorted(
        [f for f in os.listdir(args.directory_A) if f.endswith(".npy")]
    )
    dir_b_files = sorted(
        [f for f in os.listdir(args.directory_B) if f.endswith(".npy")]
    )

    assert len(dir_a_files) == len(dir_b_files), (
        "The number of files in the directories do not match. "
        "Please use the same dataset."
    )

    total_similarity = 0
    total_cols = 0

    for file_a, file_b in zip(dir_a_files, dir_b_files):
        embeddings_a = np.load(os.path.join(args.directory_A, file_a))
        embeddings_b = np.load(os.path.join(args.directory_B, file_b))

        assert embeddings_a.shape[0] == embeddings_b.shape[0], (
            "The number of columns in corresponding tables are different. "
            "Please use the same dataset."
        )

        for embedding_a, embedding_b in zip(embeddings_a, embeddings_b):
            total_similarity += cosine_similarity(embedding_a, embedding_b)
            total_cols += 1

    avg_similarity = total_similarity / total_cols if total_cols > 0 else 0
    print(f"Average cosine similarity: {avg_similarity}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute average cosine similarity between column embeddings in "
            "two directories."
        )
    )
    parser.add_argument(
        "-A",
        "--directory_A",
        type=str,
        required=True,
        help="Directory A to read column embeddings from",
    )
    parser.add_argument(
        "-B",
        "--directory_B",
        type=str,
        required=True,
        help="Directory B to read column embeddings from",
    )

    args = parser.parse_args()
    main(args)
