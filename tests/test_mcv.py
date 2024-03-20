import time
import unittest

import torch

from observatory.common_util.mcv import (
    compute_mean_vector,
    compute_covariance_matrix,
    _compute_mcv,
    compute_mcv,
)


class MCV(unittest.TestCase):
    def setUp(self):
        self.embeddings = torch.tensor(
            [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [6, 6, 6, 6, 6]],
            dtype=torch.float,
        )

    def test_compute_mean_vector(self):
        mean_vector = compute_mean_vector(self.embeddings)

        assert mean_vector.ndim == 1
        assert mean_vector.shape[0] == self.embeddings.shape[1]
        assert mean_vector.tolist() == [3, 3, 3, 3, 3]

    def test_compute_covariance_matrix(self):
        start = time.time()
        covariance_matrix = compute_covariance_matrix(self.embeddings)
        end = time.time()

        print("Self-computed covariance matrix: ", covariance_matrix)
        print(f"Computing time: {end - start} s")

        start = time.time()
        torch_covariance_matrix = torch.cov(self.embeddings.T)
        end = time.time()
        print("Torch-computed covariance matrix: ", torch_covariance_matrix)
        print(f"Computing time: {end - start} s")

        assert covariance_matrix.ndim == 2
        assert covariance_matrix.shape[0] == self.embeddings.shape[1]
        assert covariance_matrix.shape[1] == self.embeddings.shape[1]

    def test_compute_mcv(self):
        start = time.time()
        mcv = _compute_mcv(self.embeddings)
        end = time.time()

        print(f"MCV using self-computed covariance matrix: {mcv}")
        print(f"Computing time: {end - start} s")

        start = time.time()
        torch_mcv = compute_mcv(self.embeddings)
        end = time.time()

        print(f"MCV using torch-computed covariance matrix: {torch_mcv}")
        print(f"Computing time: {end - start} s")

        assert mcv == torch_mcv
