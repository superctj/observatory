import torch


def pairwise_cosine_knn(a, b, k):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0,1))
    return torch.argsort(res, dim=1, descending=True)[:, 1:k+1]


def overlap(knn, m1, m2, count, k):
    sum  = 0
    for i in range(count):
        a = set()
        b = set()
        for j in range(k):
            a.add(int(knn[m1][i][j]))
            b.add(int(knn[m2][i][j]))
        sum += len(a.intersection(b)) / k

    print(m1 + ' and ' + m2 + ' overlap:')
    print(sum/count)