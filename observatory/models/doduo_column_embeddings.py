import torch

from observatory.common_util.column_based_truncate import column_based_truncate


def get_doduo_embeddings(tables, model, tokenizer, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    model_name = "bert-base-uncased"
    truncated_tables = []

    for table in tables:
        max_rows_fit = column_based_truncate(
            table, tokenizer, max_length, model_name
        )
        truncated_table = table.iloc[:max_rows_fit, :]
        truncated_tables.append(truncated_table)

    try:
        all_embeddings = []

        for table in truncated_tables:
            table = table.reset_index(drop=True)

            table = table.astype(str)
            annot_df = model.annotate_columns(table)
            embeddings = annot_df.colemb
            embeddings = [
                torch.tensor(embeddings[j]) for j in range(len(embeddings))
            ]

            all_embeddings.append(embeddings)

        return all_embeddings

    except Exception as e:
        print("Error message:", e)
        return []
