import torch
from observatory.common_util.mcv import compute_mcv
from torch import nn

def analyze_embeddings(
    all_embeddings: list[list[torch.FloatTensor]],
) -> tuple[list[float], list[float], float, float]:
    """Analyzes column embedding populations induced by permutations.

    Computes the average of pairwise cosine similarities and multivariate
    coefficient of variation (MCV) for each column embedding population.

    Args:
        all_embeddings: A list of lists of column embeddings where each list
            contains column embeddings of a table induced by a permutation.

    Returns:
        avg_cosine_similarities: A list of the average pairwise cosine
            similarities. E.g., avg_cosine_similarities[0] is the average of
            pairwise cosine similarities of the embedding population induced by
            the first column in the original table.
        mcvs: A list of MCV values. E.g., mcvs[0] is the MCV of the embedding
            population induced by the first column in the original table.
        table_avg_cosine_similarity: The cosine similarity averaged over
            columns, i.e., the average of `avg_cosine_similarities`.
        table_avg_mcv: The MCV value averaged over columns, i.e., the average
            of `mcvs`.
    """

    avg_cosine_similarities = []
    mcvs = []

    for i in range(len(all_embeddings[0])):
        column_cosine_similarities = []
        column_embeddings = []

        for j in range(len(all_embeddings)):
            column_embeddings.append(all_embeddings[j][i])

        for j in range(1, len(all_embeddings)):
            truncated_embedding = all_embeddings[0][i]
            shuffled_embedding = all_embeddings[j][i]

            cosine_similarity = nn.functional.cosine_similarity(
                truncated_embedding, shuffled_embedding, dim=0
            )
            column_cosine_similarities.append(cosine_similarity.item())

        avg_cosine_similarity = sum(column_cosine_similarities) / len(
            column_cosine_similarities
        )
        mcv = compute_mcv(torch.stack(column_embeddings))

        avg_cosine_similarities.append(avg_cosine_similarity)
        mcvs.append(mcv)

    table_avg_cosine_similarity = sum(avg_cosine_similarities) / len(
        avg_cosine_similarities
    )
    table_avg_mcv = sum(mcvs) / len(mcvs)

    return (
        avg_cosine_similarities,
        mcvs,
        table_avg_cosine_similarity,
        table_avg_mcv,
    )

