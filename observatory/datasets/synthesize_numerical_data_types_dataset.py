import csv
import os

import pandas as pd


TARGET_TYPES = {"datePublished": 100, "startDate": 100, "endDate": 100, "dateCreated": 100, "birthDate": 100, "isbn": 500, "postalCode": 500, "price": 500, "weight": 500}


if __name__ == "__main__":
    root_dir = "/ssd/congtj/data/semtab2023/Round1-SOTAB-CPA-SCH"
    annotation_dir = os.path.join(root_dir, "annotations")
    table_dir = os.path.join(root_dir, "tables")

    # label_filepath = os.path.join(annotation_dir, "cpa_labels_round1.txt")
    train_filepath = os.path.join(annotation_dir, "sotab_cpa_train_round1.csv")
    # valid_filepath = os.path.join(annotation_dir, "sotab_cpa_validation_round1.csv")
    output_filepath = os.path.join(annotation_dir, "numerical_data_types_metadata.csv")

    train_annotations = pd.read_csv(train_filepath)

    with open(output_filepath, "w") as f:
        header = ["table_name", "subject_column_index", "column_index", "label"]
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()

        for target_type, sample_size in TARGET_TYPES.items():
            target_df = train_annotations.loc[train_annotations["label"] == target_type]
            sample_df = target_df.sample(n=sample_size, random_state=12345)
            for _, row in sample_df.iterrows():
                csv_writer.writerow({"table_name": row["table_name"], "subject_column_index": row["main_column_index"], "column_index": row["column_index"], "label": row["label"]})
