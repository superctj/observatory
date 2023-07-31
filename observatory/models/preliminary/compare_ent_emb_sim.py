import torch
import os
import pickle

data_dir = "data/"
with open(os.path.join(data_dir, "tapas_emb_no_metadata.pkl"), "rb") as f:
    tapas_emb = pickle.load(f)
with open(os.path.join(data_dir, "turl_emb_no_metadata.pkl"), "rb") as f:
    turl_emb = pickle.load(f)
with open(os.path.join(data_dir, "bert_emb_no_metadata.pkl"), "rb") as f:
    bert_emb = pickle.load(f)
with open(os.path.join(data_dir, "roberta_emb_no_metadata.pkl"), "rb") as f:
    roberta_emb = pickle.load(f)
with open(os.path.join(data_dir, "t5_emb_no_metadata.pkl"), "rb") as f:
    t5_emb = pickle.load(f)

count = 919
k = 10
sum = 0
sample_size = 1000
tapas_mat = torch.zeros((count, 768))
turl_mat = torch.zeros((count, 312))
bert_mat = torch.zeros((count, 768))
roberta_mat = torch.zeros((count, 768))
t5_mat = torch.zeros((count, 768))

j = 0
for i in range(4, sample_size + 4):
    if i in tapas_emb:
        tapas_mat[j, :] = torch.FloatTensor(tapas_emb[i])
        turl_mat[j, :] = torch.FloatTensor(turl_emb[i])
        bert_mat[j, :] = torch.FloatTensor(bert_emb[i])
        roberta_mat[j, :] = torch.FloatTensor(roberta_emb[i])
        t5_mat[j, :] = torch.FloatTensor(t5_emb[i])
        j += 1
assert j == count


def pairwise_cosine_knn(a, b, k):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    res = torch.mm(a_norm, b_norm.transpose(0, 1))
    return torch.argsort(res, dim=1, descending=True)[:, 1 : k + 1]


tapas_knn = pairwise_cosine_knn(tapas_mat, tapas_mat, k)
turl_knn = pairwise_cosine_knn(turl_mat, turl_mat, k)
bert_knn = pairwise_cosine_knn(bert_mat, bert_mat, k)
roberta_knn = pairwise_cosine_knn(roberta_mat, roberta_mat, k)
t5_knn = pairwise_cosine_knn(t5_mat, t5_mat, k)
knn = {
    "tapas": tapas_knn,
    "turl": turl_knn,
    "bert": bert_knn,
    "roberta": roberta_knn,
    "t5": t5_knn,
}


def overlap(m1, m2, count, k):
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


models = ["tapas", "turl", "bert", "roberta", "t5"]
for i in range(5):
    for j in range(i + 1, 5):
        overlap(models[i], models[j], count, k)
