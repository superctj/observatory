import glob
import os
import pickle
import sqlite3

import pandas as pd


def export_sqlite_to_csv(db_file_path: str, output_dir: str):
    conn = sqlite3.connect(db_file_path)
    cur = conn.cursor()

    res = cur.execute("SELECT tbl_name FROM sqlite_master WHERE type='table' and tbl_name not like 'metadata%' and tbl_name not like 'sqlite_%' and tbl_name!='dataset_profile';").fetchall()

    for row in res:
        table_name = row[0]
        try:
            db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except:
            print(table_name)
            # sqlite3.OperationalError: Could not decode to UTF-8 column 'last_name' with text 'Treyes Albarrac��N'
            conn.text_factory = lambda b: b.decode(errors="ignore")
            db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        # except: # b'Web_client_accelerator'
        #     table_name = row[0].decode("utf-8")
        #     db_df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        output_path = os.path.join(output_dir, f"{table_name}.csv")
        db_df.columns= db_df.columns.str.lower()
        db_df.to_csv(output_path, index=False)
    
    conn.close()


def export_all():
    root_dir = "C:/Users/Ben/Documents/table_eval/great_lake/property4/data/DB_schema_abbreviation/database_post_perturbation/"
    output_dir = "C:/Users/Ben/Documents/table_eval/great_lake/property4/abbreviation"


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for db_name in os.listdir(root_dir):
        db_file_path = os.path.join(root_dir, f"{db_name}/{db_name}.sqlite")
        csv_output_dir = os.path.join(output_dir, db_name)

        if not os.path.exists(csv_output_dir):
            os.makedirs(csv_output_dir)
        try:
            export_sqlite_to_csv(db_file_path, csv_output_dir)
        except:
            print(db_name)

export_all()
