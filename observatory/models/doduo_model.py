from observatory.models.DODUO.doduo import Doduo


if __name__ == "__main__":
    import argparse
    import os

    import pandas as pd

    args = argparse.Namespace
    args.model = "wikitable" # two models available "wikitable" and "viznet"
    doduo = Doduo(args, basedir="/ssd/congtj/observatory/doduo")

    data_dir = "/home/congtj/observatory/observatory/models/DODUO/sample_tables"
    # df = pd.read_csv(os.path.join(data_dir, "sample_table0.csv"), index_col=0)
    df = pd.read_csv(os.path.join(data_dir, "sample_table0.csv"), index_col=False)

    annot_df = doduo.annotate_columns(df)
    # print(annot_df.coltypes)
    # print(annot_df.colrels)
    print("Number of contextualized column embeddings: ", len(annot_df.colemb)) # list of numpy
