import torch


def compute_mean_vector(embeddings: torch.FloatTensor) -> torch.FloatTensor:
    """Computes the mean value of `embeddings` in the first dimension.

    Args:
        embeddings: A 2D tensor of shape (number of embeddings,
            embedding dimension).

    Returns:
        The mean tensor of 1 dimension.
    """

    return torch.mean(embeddings, dim=0)


def compute_covariance_matrix(embeddings: torch.FloatTensor):
    """Computes the covariance matrix of `embeddings`.

    Args:
        embeddings: A 2D tensor of shape (number of embeddings,
            embedding dimension).

    Returns:
        The covariance matrix of `embeddings`.
    """

    mean_vector = compute_mean_vector(embeddings)
    zero_mean_embeddings = embeddings - mean_vector.unsqueeze(0)
    covariance_matrix = torch.matmul(
        zero_mean_embeddings.T, zero_mean_embeddings
    ) / (embeddings.shape[0] - 1)

    return covariance_matrix


def _compute_mcv(embeddings: torch.FloatTensor) -> float:
    """Computes multivariate coefficient of variation (MCV).

    Args:
        embeddings: A 2D tensor of shape (number of embeddings,
            embedding dimension).

    Returns:
        The MCV value.
    """

    assert embeddings.ndim == 2

    mean_vector = torch.mean(embeddings, dim=0)
    covariance_matrix = compute_covariance_matrix(embeddings)
    mcv = torch.sqrt(
        torch.matmul(mean_vector, torch.matmul(covariance_matrix, mean_vector))
        / (torch.matmul(mean_vector, mean_vector) ** 2)
    )

    return mcv.item()


def compute_mcv(embeddings: torch.FloatTensor) -> float:
    """Computes multivariate coefficient of variation (MCV).

    Args:
        embeddings: A 2D tensor of shape (number of embeddings,
            embedding dimension).

    Returns:
        The MCV value.
    """

    assert embeddings.ndim == 2

    mean_vector = torch.mean(embeddings, dim=0)
    covariance_matrix = torch.cov(embeddings.T)
    mcv = torch.sqrt(
        torch.matmul(mean_vector, torch.matmul(covariance_matrix, mean_vector))
        / (torch.matmul(mean_vector, mean_vector) ** 2)
    )

    return mcv.item()
