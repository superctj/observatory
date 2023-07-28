import torch


def pairwise_cosine_knn(a, b, k):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return torch.argsort(res, dim=1, descending=True)[:, 1 : k + 1]


def overlap(m1, m2, count, k, knn):
    sum = 0
    for i in range(count):
        a = set()
        b = set()
        for j in range(k):
            a.add(int(knn[m1][i][j]))
            b.add(int(knn[m2][i][j]))
        sum += len(a.intersection(b)) / k

    print(m1 + " and " + m2 + " overlap:")
    print(sum / count)


def jaccard(t1, t2):
    intersection = len(set(t1).intersection(t2))
    union = len(t1) + len(t2) - intersection
    return intersection / union
