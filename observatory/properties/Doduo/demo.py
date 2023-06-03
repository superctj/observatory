from doduo import Doduo
import torch

if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd

    args = argparse.Namespace
    args.model = "wikitable" # two models available "wikitable" and "viznet"
    doduo = Doduo(args, basedir=".")

    import numpy as np

    # Setting a seed for reproducibility
    np.random.seed(0)

    # Data
    data = {
        "Name": ["Name_" + str(i) for i in range(1, 201)],
        "Age": np.random.randint(20, 70, size=200),
        "City": ["City_" + str(np.random.randint(1, 5)) for _ in range(200)]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.DataFrame(data)
    annot_df = doduo.annotate_columns(df)
    embeddings = annot_df.colemb
    embeddings = [torch.tensor(embeddings[j]) for j in range(len(embeddings))]
    # print(annot_df.coltypes)
    # print(annot_df.colrels)
    print("Number of contextualized column embeddings: ", len(annot_df.colemb)) 
    print("first contextualized column embeddings: ", annot_df.colemb[0].shape) 
    print("first contextualized column embeddings: ", torch.tensor(annot_df.colemb[0])) 