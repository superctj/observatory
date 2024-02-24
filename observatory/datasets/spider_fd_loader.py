import os

import pandas as pd


class SpiderFDDataLoader:
    def __init__(
        self, dataset_dir: str, fd_metadata_path: str, non_fd_metadata_path: str
    ):
        self.dataset_dir = dataset_dir
        self.fd_metadata_path = fd_metadata_path
        self.non_fd_metadata_path = non_fd_metadata_path

    def read_table(
        self, table_name: str, drop_nan=True, **kwargs
    ) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, table_name)
        table = pd.read_csv(
            table_path, on_bad_lines="skip", lineterminator="\n", **kwargs
        )

        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_fd_metadata(self):
        fd_metadata = pd.read_csv(self.fd_metadata_path, sep=",")
        return fd_metadata

    def get_non_fd_metadata(self):
        non_fd_metadata = pd.read_csv(self.non_fd_metadata_path, sep=",")
        return non_fd_metadata


if __name__ == "__main__":
    root_dir = "/ssd/congtj/observatory/spider_datasets/fd_artifact"
    dataset_dir = os.path.join(root_dir, "datasets")
    fd_metadata_path = os.path.join(root_dir, "fd_metadata.csv")
    non_fd_metadata_path = os.path.join(root_dir, "non_fd_metadata.csv")

    data_loader = SpiderFDDataLoader(
        dataset_dir, fd_metadata_path, non_fd_metadata_path
    )

    non_fd_metadata = data_loader.get_non_fd_metadata()
    for _, row in non_fd_metadata.iterrows():
        table_name = row["table_name"]
        col1 = row["column_1"]
        col2 = row["column_2"]

        table = data_loader.read_table(table_name)
        print(table.head())

        break
