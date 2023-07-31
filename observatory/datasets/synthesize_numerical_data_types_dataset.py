import csv
import os

import pandas as pd


# NON_TEXT_TYPES = {"datePublished": 100, "startDate": 100, "endDate": 100, "dateCreated": 100, "birthDate": 100, "isbn": 500, "postalCode": 500, "price": 500, "weight": 500}

NON_TEXT_TYPES = {
    "numberOfPages": 300,
    "isbn": 300,
    "image": 150,
    "url": 150,
    "ratingValue": 300,
    "price": 300,
    "startDate": 50,
    "endDate": 50,
    "validFrom": 50,
    "dateCreated": 50,
    "releaseDate": 50,
    "datePublished": 50,
    "telephone": 300,
    "postalCode": 300,
    "faxNumber": 300,
    "weight": 300,
}

TEXT_TYPES = {
    "inLanguage": 300,
    "priceCurrency": 300,
    "publisher": 300,
    "review": 300,
    "director": 100,
    "actor": 100,
    "byArtist": 100,
    "inAlbum": 300,
    "jobTitle": 300,
    "addressCountry": 150,
    "nationality": 150,
    "streetAddress": 300,
    "addressRegion": 300,
}

NON_TEXT_TYPE_MAPPING = {
    "numberOfPages": "numberOfPages",
    "isbn": "isbn",
    "image": "url",
    "url": "url",
    "ratingValue": "ratingValue",
    "price": "price",
    "startDate": "date",
    "endDate": "date",
    "validFrom": "date",
    "dateCreated": "date",
    "releaseDate": "date",
    "datePublished": "date",
    "telephone": "telephone",
    "postalCode": "postalCode",
    "faxNumber": "faxNumber",
    "weight": "weight",
}

TEXT_TYPE_MAPPING = {
    "inLanguage": "language",
    "priceCurrency": "currency",
    "publisher": "organization",
    "review": "review",
    "director": "personName",
    "actor": "personName",
    "byArtist": "personName",
    "inAlbum": "musicAlbum",
    "jobTitle": "jobTitle",
    "addressCountry": "country",
    "nationality": "country",
    "streetAddress": "streetAddress",
    "addressRegion": "addressRegion",
}  # "author": "person", "director": "person", "actor": "person", "byArtist": "person"


def check_single_table(filepath):
    table = pd.read_json(filepath, compression="gzip", lines=True)
    pd.set_option("display.max_columns", None)
    print(table.head(n=5))


def collect_text_type_data(annotation_dir, train_annotations):
    unique_labels = set()
    for val in TEXT_TYPE_MAPPING.values():
        unique_labels.add(val)
    num_unique_labels = len(unique_labels)

    output_filepath = os.path.join(
        annotation_dir, f"text_types_{num_unique_labels}-classes_metadata.csv"
    )

    with open(output_filepath, "w") as f:
        header = ["table_name", "subject_column_index", "column_index", "label"]
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()

        for target_type, sample_size in TEXT_TYPES.items():
            target_df = train_annotations.loc[train_annotations["label"] == target_type]
            print(target_type)
            print("=" * 50)
            sample_df = target_df.sample(n=sample_size, random_state=12345)
            label = TEXT_TYPE_MAPPING[target_type]

            for _, row in sample_df.iterrows():
                csv_writer.writerow(
                    {
                        "table_name": row["table_name"],
                        "subject_column_index": row["main_column_index"],
                        "column_index": row["column_index"],
                        "label": label,
                    }
                )


if __name__ == "__main__":
    root_dir = (
        "/ssd/congtj/data/semtab2023/Round1-SOTAB-CPA-SCH"  # Round1-SOTAB-CPA-SCH"
    )
    annotation_dir = os.path.join(root_dir, "annotations")
    table_dir = os.path.join(root_dir, "tables")

    # filename = "Product_3bears.co.uk_September2020_CPA.json.gz"
    # filepath = os.path.join(table_dir, filename)
    # check_single_table(filepath)

    train_filepath = os.path.join(annotation_dir, "sotab_cpa_train_round1.csv")
    train_annotations = pd.read_csv(train_filepath)
    # valid_filepath = os.path.join(annotation_dir, "sotab_cpa_validation_round1.csv")

    collect_text_type_data(annotation_dir, train_annotations)

    # unique_labels = set()
    # for val in NON_TEXT_TYPE_MAPPING.values():
    #     unique_labels.add(val)
    # num_unique_labels = len(unique_labels)

    # output_filepath = os.path.join(annotation_dir, f"non_textual_data_types_{num_unique_labels}_classes_metadata.csv")

    # with open(output_filepath, "w") as f:
    #     header = ["table_name", "subject_column_index", "column_index", "label"]
    #     csv_writer = csv.DictWriter(f, fieldnames=header)
    #     csv_writer.writeheader()

    #     for target_type, sample_size in NON_TEXT_TYPES.items():
    #         target_df = train_annotations.loc[train_annotations["label"] == target_type]
    #         print(target_type)
    #         print("=" * 50)
    #         sample_df = target_df.sample(n=sample_size, random_state=12345)
    #         label = NON_TEXT_TYPE_MAPPING[target_type]

    #         for _, row in sample_df.iterrows():
    #             csv_writer.writerow({"table_name": row["table_name"], "subject_column_index": row["main_column_index"], "column_index": row["column_index"], "label": label})
