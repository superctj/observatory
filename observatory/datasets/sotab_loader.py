import os

import pandas as pd


class SOTABDataLoader():
    def __init__(self, dataset_dir: str, metadata_path: str):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{dataset_dir} does not exist.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{metadata_path} does not exist.")

        self.dataset_dir = dataset_dir        
        self.metadata = self._read_metadata(metadata_path)

    def _read_metadata(self, metadata_path: str):
        metadata = pd.read_csv(metadata_path)
        return metadata
    
    def read_table(self, filename: str) -> pd.DataFrame:
        filepath = os.path.join(self.dataset_dir, filename)
        table = pd.read_json(filepath, compression="gzip", lines=True)

        return table


if __name__ == "__main__":
    root_dir = "/ssd/congtj/observatory/sotab_numerical_data_type_datasets"
    dataset_dir = os.path.join(root_dir, "tables")
    metadata_path = os.path.join(root_dir, "metadata.csv") # "nontext_types_10-classes_metadata.csv", "text_types_10-classes_metadata.csv" 

    data_loader = SOTABDataLoader(dataset_dir, metadata_path)
    for _, row in data_loader.metadata.iterrows():
        table_name = row["table_name"]
        table = data_loader.read_table(table_name)
        print(table.columns)
        
        """
        # Only consider numerical column alone for representation inference
        numerical_col_idx = row["column_index"]
        numerical_col = table.iloc[:, numerical_col_idx]
        representation = f(numerical_col)

        # Consider the subject column as context of numerical column for representation inference
        subj_col_idx = row["subject_column_index"]
        two_col_table = table.iloc[:, [subj_col_idx, numerical_col]]
        representation = f(two_col_table)

        # Consider the entire table as context of numerical column for representation inference
        representation = f(table)
        """
        break
