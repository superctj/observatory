import os

from observatory.datasets.sotab_loader import SOTABDataLoader


def get_embeddings():
    pass


if __name__ == "__main__":
    root_dir = "/ssd/congtj/observatory/sotab_numerical_data_type_datasets"
    dataset_dir = os.path.join(root_dir, "tables")
    metadata_path = os.path.join(root_dir, "metadata.csv")

    data_loader = SOTABDataLoader(dataset_dir, metadata_path)

    col_itself = []
    subj_col_as_context = []
    neighbor_col_as_context = []
    entire_table_as_context = []

    for _, row in data_loader.metadata.iterrows():
        table_name = row["table_name"]
        table = data_loader.read_table(table_name)

        input_tables = []
        # Only consider numerical column alone for representation inference
        numerical_col_idx = row["column_index"]
        numerical_col = table.iloc[:, numerical_col_idx]
        input_tables.append(numerical_col)

        # Consider the subject column as context of numerical column for representation inference
        subj_col_idx = row["subject_column_index"]
        two_col_table = table.iloc[:, [subj_col_idx, numerical_col_idx]]
        input_tables.append(two_col_table)

        # Consider immediate neighboring columns as context of numerical column for representation inference
        num_cols = len(list(table.columns))
        if numerical_col_idx > 0 and numerical_col_idx < num_cols - 1:
            three_col_table = table.iloc[:, [numerical_col_idx-1, numerical_col_idx, numerical_col_idx+1]]
        elif numerical_col_idx == num_cols - 1:
            three_col_table = table.iloc[:, [numerical_col_idx-1, numerical_col_idx]]
        
        input_tables.append(three_col_table)

        # Consider the entire table as context of numerical column for representation inference
        input_tables.append(table)

        embeddings = get_embeddings[input_tables]
        
        col_itself.append((embeddings[0][0], row["label"]))
        subj_col_as_context.append((embeddings[0][1], row["label"]))
        neighbor_col_as_context.append(embeddings[1][1], row["label"])
        entire_table_as_context.append(embeddings[2][numerical_col_idx], row["label"])

        # Save embeddings

