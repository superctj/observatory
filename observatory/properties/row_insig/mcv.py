import torch

def compute_mean_vector(embeddings):
    return torch.mean(embeddings, dim=0)

def compute_covariance_matrix(embeddings):
    mean_vector = compute_mean_vector(embeddings)
    zero_mean_embeddings = embeddings - mean_vector.unsqueeze(0)
    covariance_matrix = torch.matmul(zero_mean_embeddings.T, zero_mean_embeddings) / (embeddings.shape[0] - 1)
    return covariance_matrix

def compute_mcv(embeddings):
    mean_vector = compute_mean_vector(embeddings)
    covariance_matrix = compute_covariance_matrix(embeddings)
    # inv_covariance_matrix = torch.linalg.inv(covariance_matrix)
    # print(mean_vector.permute(-1, *range(mean_vector.ndim - 1)).shape)
    mcv = torch.sqrt(torch.matmul(mean_vector.T, torch.matmul(covariance_matrix, mean_vector))/(torch.matmul(mean_vector.T, mean_vector)**2))

    return mcv.item()
