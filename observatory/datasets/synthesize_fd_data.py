import csv
import glob
import logging
import os
import random
random.seed(12345)
import shutil
from typing import Any
from pprint import pformat, pprint
from typing import Dict, List

import pandas as pd

from openclean_metanome.algorithm.hyfd import hyfd
from openclean_metanome.download import download_jar


def create_new_directory(path: str, force: bool = False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if force:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            print("Directory exists...")


def custom_logger(logger_name: str, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    format_string="[%(levelname)s] %(message)s"
    log_format = logging.Formatter(format_string)
    
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="w")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def is_numeric_heuristic(value: Any) -> bool:
    """
    Check if a given column value can be converted to a float.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False

    
def read_table(table_path: str, drop_nan=True, **kwargs) -> pd.DataFrame:
    table = pd.read_csv(
        table_path, on_bad_lines="skip", lineterminator="\n", **kwargs)
    
    if drop_nan:
        table.dropna(axis=1, how="all", inplace=True)
        table.dropna(axis=0, how="any", inplace=True)

    return table


def fd_detection(df: pd.DataFrame, max_lhs_size) -> List:
    try:
        fds = hyfd(df, max_lhs_size=max_lhs_size)
    except Exception as e:
        print("=" * 80)
        print("Caught Exception:")
        print(str(e))
        
        print("-" * 80)
        error_line = int(str(e).split("\n")[2].split(" ")[-1])
        print(f"Error line number: {error_line}")
        
        print(f"Error line data:")
        pprint(df.iloc[error_line-2])
        print("-" * 80)
        pprint(df.iloc[error_line-1])
        print("-" * 80)
        pprint(df.iloc[error_line])
        exit()
    
    return fds


def select_context(df: pd.DataFrame, fds: List) -> Dict:
    col_types = df.dtypes
    all_contexts = {}
    
    # Retrieve contexts
    for fd in fds:
        if len(fd.rhs) == 1 and col_types[fd.rhs[0]] == "object":
            dependent = fd.rhs[0]
            try:
                assert(len(fd.lhs) != 0)
            except:
                continue
            if dependent in all_contexts:
                all_contexts[dependent].append(fd.lhs)
            else:
                all_contexts[dependent] = [fd.lhs]

    # Count non-NaN values in each column
    non_nan_histogram = df.count()
    col_context_map = {}

    for dependent in all_contexts:
        valid_ratio = 0
        for context in all_contexts[dependent]:
            if len(context) == 1:
                determinant = context[0]
                # try:
                #     df[determinant][0]
                # except:
                #     pprint("=" * 80)
                #     pprint(df)
                #     pprint("=" * 50)
                #     pprint(determinant)
                #     pprint(df[determinant].iat[0])
                #     exit()

                # Skip single determinant that is a ID column
                if col_types[determinant] == "object" and not is_numeric_heuristic(df[determinant].iat[0]):
                    col_context_map[dependent] = [determinant]
                    continue
            else:
                local_ratio = 0
                for determinant in context:
                    local_ratio += non_nan_histogram[determinant]

                local_ratio /= len(context)
                if local_ratio > valid_ratio:
                    valid_ratio = local_ratio
                    col_context_map[dependent] = context

    return col_context_map


def extract_fd(data_dir: str, output_filepath, logger: logging.Logger) -> Dict:
    with open(output_filepath, "w") as f:
        header = ["table_name", "determinant", "dependent"]
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()

        for table_path in glob.glob(f"{data_dir}/*/*.csv"):
            table_name = "/".join(table_path.split("/")[-2:])
            df = read_table(table_path, drop_nan=True)
            if df.empty:
                logger.info(f"Table *{table_name}* is empty after preprocessing...")
                logger.info("=" * 50)
                continue

            if df.shape[0] < 10:
                continue
            
            # Detect functional dependency
            if df.shape[0] > 1000:
                df = df.sample(n=1000, random_state=12345)
            fds = fd_detection(df, max_lhs_size=1)

            # Select context
            col_context_map = select_context(df, fds)
            # Populate entries
            for key, value in col_context_map.items():
                for determinant in value:
                    csv_writer.writerow({"table_name": table_name, "determinant": determinant, "dependent": key})

            # Log detection results
            logger.info(f"Table name: {table_name}")
            logger.info("Column context map: ")
            logger.info(pformat(col_context_map))
            logger.info("=" * 50)

    logger.info("Functional dependency detection completed...")


def extract_non_fd(data_dir: str, fd_filepath: str, output_filepath: str):
    fd_df = pd.read_csv(fd_filepath, sep=",")
    with open(output_filepath, "w") as f:
        header = ["table_name", "column_1", "column_2"]
        csv_writer = csv.DictWriter(f, fieldnames=header)
        csv_writer.writeheader()

        for group_id, group_df in fd_df.groupby("table_name"):
            fds = {}
            num_fds = group_df.shape[0]
            for _, row in group_df.iterrows():
                if row["determinant"] in fds:
                    fds[row["determinant"]].append(row["dependent"])
                else:
                    fds[row["determinant"]] = [row["dependent"]]
            
            table_path = os.path.join(data_dir, group_id)
            table = read_table(table_path)
            col_names = table.columns.tolist()
            unique_pairs = set()

            num_non_fds = 0
            while num_non_fds < num_fds:
                col1_idx, col2_idx = random.randint(0, len(col_names)-1), random.randint(0, len(col_names)-1)

                if col1_idx == col2_idx:
                    continue

                col1_name = col_names[col1_idx]
                col2_name = col_names[col2_idx]
                if col1_name in fds and col2_name in fds[col1_name]:
                    continue

                csv_writer.writerow({"table_name": group_id, "column_1": col1_name, "column_2": col2_name})
                num_non_fds += 1


def main():
    # Download the 'Metanome.jar' file if no copy exists on the local machine at the path that is defined by config.JARFILE().
    # env = {config.METANOME_JARPATH: config.JARFILE()}
    # download_jar(verbose=True)

    # dataset_name = "spider"
    # data_dir = "/ssd/congtj/observatory/spider_datasets/spider_artifact/csv_extended"
    # fd_output_filepath = "/ssd/congtj/observatory/spider_datasets/spider_artifact/fd_metadata.csv"
    # log_dir = f"./log/{dataset_name}"
    # create_new_directory(log_dir)

    # # Log detected functional dependency for inspection
    # log_file = os.path.join(log_dir, "log.txt")
    # logger = custom_logger(log_file, level=logging.INFO)
    # extract_fd(data_dir, fd_output_filepath, logger)

    data_dir = "/ssd/congtj/observatory/spider_datasets/spider_artifact/csv_extended"
    fd_output_filepath = "/ssd/congtj/observatory/spider_datasets/spider_artifact/fd_metadata.csv"
    non_fd_output_filepath = "/ssd/congtj/observatory/spider_datasets/spider_artifact/non_fd_metadata.csv"
    extract_non_fd(data_dir, fd_output_filepath, non_fd_output_filepath)


if __name__ == "__main__":
    main()
