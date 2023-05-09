import argparse
import json
import os
import pandas as pd

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def process_one_table(data):
    processed_table_headers = data['processed_tableHeaders']
    table_data = data['tableData']
    formatted_table_data = [[cell['text'] for cell in row] for row in table_data]
    df = pd.DataFrame(formatted_table_data, columns=processed_table_headers)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script processes a jsonl file and saves each table as a separate csv file in the given directory.")
    parser.add_argument('-f', '--file_path', type=str, required=True, help="Path to the jsonl file to be processed. For example: -f dev_tables.jsonl")
    parser.add_argument('-d', '--directory', type=str, required=True, help="Directory where the csv files will be saved. For example: -d tables_csv")
    args = parser.parse_args()

    jsonl_data = read_jsonl(args.file_path)

    tables = []
    for data in jsonl_data:
        tables.append(process_one_table(data))

    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    for i, table in enumerate(tables):
        table.to_csv(f"{args.directory}/table_{i}.csv", index=False, na_rep='')
