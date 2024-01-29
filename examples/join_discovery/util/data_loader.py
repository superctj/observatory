import glob
import os

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import pandas as pd
# from pyspark import SparkConf
# from pyspark.sql import SparkSession


class CSVDataLoader(ABC):
    @abstractmethod
    def _read_metadata(self, metadata_path: str) -> Tuple[List[str], Dict]:
        pass

    @abstractmethod
    def read_table(self, table_name: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_table_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_queries(self) -> Dict[str, List[str]]:
        pass


class SpiderCSVDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, metadata_path: str):
        self.dataset_dir = dataset_dir        
        self.table_names, self.queries = self._read_metadata(metadata_path)
    
    def _read_metadata(self, metadata_path: str) -> Tuple[List[str], Dict]:
        table_names = []
        queries = {}

        metadata_df = pd.read_csv(metadata_path)

        for _, row in metadata_df.iterrows():
            db_name = row["db name"]
            t1_name, c1_name = row["t1 name"], row["c1 name"]
            t2_name, c2_name = row["t2 name"], row["c2 name"]
            subj_name = row["subject name"]

            # Skip ID column that does not have subject column
            if subj_name == "-": 
                continue

            t1_path = os.path.join(db_name, t1_name)
            if t1_path not in table_names:
                table_names.append(t1_path)
            
            t2_path = os.path.join(db_name, t2_name)
            if t2_path not in table_names:
                table_names.append(t2_path)

            query_id = t1_path + "!" + subj_name
            answer = t2_path + "!" + subj_name
            if query_id in queries:
                queries[query_id].append(answer)
            else:
                queries[query_id] = [answer]

        return table_names, queries
    
    def read_table(self, table_name: str, drop_nan=True, **kwargs) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, f"{table_name}.csv")
        table = pd.read_csv(
            table_path, on_bad_lines="skip", lineterminator="\n", **kwargs)
        
        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_table_names(self) -> List[str]:
        return self.table_names
    
    def get_queries(self) -> Dict[str, List[str]]:
        return self.queries


# Adopted from https://github.com/dtim-upc/NextiaJD/blob/master/experiments/LSH%20Ensemble/LSHEnsemble_comparison.py
class NextiaJDSparkDataLoader():
    def __init__(self, dataset_dir: str, metadata_path: str, ground_truth_path: str):
        conf = SparkConf() \
            .set("spark.driver.maxResultSize", "8G") \
            .set("spark.driver.memory", "8G") \
            .set("spark.sql.execution.arrow.pyspark.enabled", "true") # Enable Arrow-based columnar data transfers
        self.spark = SparkSession.builder.config(conf=conf).master("local").getOrCreate()

        self.dataset_dir = dataset_dir
        self.metadata = self.read_metadata(metadata_path)
        self.ground_truth = self.read_ground_truth(ground_truth_path)

    def read_dataset(self, file_path: str, delimiter: str, multiline: str, null_value: str, ignore_trailing: str) -> pd.DataFrame:
        return self.spark.read \
            .option("header", "true") \
            .option("inferSchema", "true") \
            .option("delimiter", delimiter) \
            .option("multiline", multiline) \
            .option("quote", "\"") \
            .option("escape", "\"") \
            .option("nullValue", null_value) \
            .option("ignoreTrailingWhiteSpace", ignore_trailing) \
            .csv(file_path)
        
    def read_metadata(self, metadata_path: str) -> Dict[str, Dict[str, str]]:
        spark_df = self.read_dataset(metadata_path, delimiter=",", multiline="false", null_value="", ignore_trailing="true")
        metadata = {}

        for row in spark_df.select("filename", "delimiter", "multiline", "nullVal", "file_size", "ignoreTrailing").distinct().collect():
            metadata[row["filename"]] = {
                "delimiter": row["delimiter"],
                "multiline": row["multiline"],
                "null_val": row["nullVal"],
                "ignore_trailing": row["ignoreTrailing"]
            }

        return metadata
    
    def read_ground_truth(self, ground_truth_path: str) -> pd.DataFrame:
        df = self.read_dataset(ground_truth_path, delimiter=",", multiline="false", null_value="", ignore_trailing="true")
        return df.select("*").toPandas()

    def get_tables(self) -> List[str]:
        return self.metadata.keys()

    def read_table(self, table_name: str) -> pd.DataFrame:
        file_path = os.path.join(self.dataset_dir, table_name)
        spark_df = self.read_dataset(
            file_path,
            delimiter=self.metadata[table_name]["delimiter"],
            multiline=self.metadata[table_name]["multiline"],
            null_value=self.metadata[table_name]["null_val"],
            ignore_trailing=self.metadata[table_name]["ignore_trailing"]
        )

        return spark_df.select("*").toPandas()
    
    def get_queries(self) -> Dict[str, List[str]]:
        queries = {}

        for _, row in self.ground_truth.iterrows():
            if row["trueQuality"] > 2.0:
                query_id = row["ds_name"] + "!" + row["att_name"]
                answer = row["ds_name_2"] + "!" + row["att_name_2"]
                if query_id in queries:
                    queries[query_id].append(answer)
                else:
                    queries[query_id] = [answer]
            
        return queries


class NextiaJDCSVDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, metadata_path: str, ground_truth_path):
        if not os.path.exists(dataset_dir):
            raise FileNotFoundError(f"{dataset_dir} does not exist.")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"{metadata_path} does not exist.")
        if not os.path.exists(ground_truth_path):
            raise FileNotFoundError(f"{ground_truth_path} does not exist.")

        self.dataset_dir = dataset_dir        
        self.metadata = self._read_metadata(metadata_path)
        self.ground_truth = self._read_ground_truth(ground_truth_path)
    
    def _read_metadata(self, metadata_path: str) -> pd.DataFrame:
        df = pd.read_csv(metadata_path)
        metadata = {}

        for _, row in df.iterrows():
            metadata[row["filename"]] = {
                "delimiter": row["delimiter"],
                "null_val": row["nullVal"],
                "ignore_trailing": row["ignoreTrailing"]
            }

        return metadata
    
    def _read_ground_truth(self, ground_truth_path: str) -> pd.DataFrame:
        return pd.read_csv(ground_truth_path)

    def read_table(self, table_name: str, drop_nan: bool = True, **kwargs) -> pd.DataFrame:
        file_path = os.path.join(self.dataset_dir, table_name)
        try:
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                low_memory=False,
                **kwargs
                #escapechar="\"",
            )
        except UnicodeDecodeError: # To open CSV files with UnicodeDecodeError
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                encoding="iso-8859-1",
                low_memory=False,
                **kwargs
            )
        except ValueError: # python engine of pandas does not support "low_memory" argument
            table = pd.read_csv(
                file_path,
                delimiter=self.metadata[table_name]["delimiter"],
                na_values=self.metadata[table_name]["null_val"],
                skipinitialspace=self.metadata[table_name]["ignore_trailing"],
                quotechar="\"",
                on_bad_lines="skip",
                lineterminator="\n",
                **kwargs
            )
        # finally:
        #     print(table_name)

        # Remove hidden characters (e.g., "\r") in DataFrame header and data
        table.columns = table.columns.str.replace("[\r]", "", regex=True)
        table.replace(to_replace=["\r", "\n"], value="", regex=True, inplace=True)

        # Drop empty columns and rows with any missing values
        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_table_names(self) -> List[str]:
        return self.metadata.keys()
    
    def get_queries(self) -> Dict[str, List[str]]:
        queries = {}

        for _, row in self.ground_truth.iterrows():
            if row["trueQuality"] > 2.0:
                query_id = row["ds_name"] + "!" + row["att_name"]
                answer = row["ds_name_2"] + "!" + row["att_name_2"]
                if query_id in queries:
                    queries[query_id].append(answer)
                else:
                    queries[query_id] = [answer]
            
        return queries


class SigmaExampleCSVDataLoader(CSVDataLoader):
    def __init__(self, dataset_dir: str, metadata_path: str):
        self.dataset_dir = dataset_dir        
        self.table_names, self.queries = self._read_metadata(metadata_path)
    
    def _read_metadata(self, metadata_path: str) -> Tuple[List[str], Dict]:
        csv_paths = glob.glob(f"{self.dataset_dir}/*.csv")
        table_names = [os.path.basename(x)[:-4] for x in csv_paths]

        queries = {}
        metadata_df = pd.read_csv(metadata_path, delimiter=",")

        for _, row in metadata_df.iterrows():
            db_name = row["db name"]
            tbl_name = row["table name"]
            col_name = row["column name"]
            answer = "place!holder"

            tbl_id = db_name + "+" + tbl_name
            query_id = tbl_id + "!" + col_name

            if query_id in queries:
                queries[query_id].append(answer)
            else:
                queries[query_id] = [answer]

        return table_names, queries
    
    def read_table(self, table_name: str, drop_nan: bool = True) -> pd.DataFrame:
        table_path = os.path.join(self.dataset_dir, f"{table_name}.csv")
        table = pd.read_csv(
            table_path, on_bad_lines="skip", lineterminator="\n")

        # Drop empty columns and rows with any missing values
        if drop_nan:
            table.dropna(axis=1, how="all", inplace=True)
            table.dropna(axis=0, how="any", inplace=True)

        return table

    def get_table_names(self) -> List[str]:
        return self.table_names
    
    def get_queries(self) -> Dict[str, List[str]]:
        return self.queries



def compute_basic_statistics(dataloader: Union[NextiaJDCSVDataLoader, NextiaJDSparkDataLoader]) -> None:
    num_rows, num_cols, num_datasets = 0, 0, 0

    for table_name in dataloader.get_table_names():
        table_data = dataloader.read_table(table_name, drop_nan=False)
        num_rows += table_data.shape[0]
        num_cols += table_data.shape[1]
        num_datasets += 1
    
    print("=" * 50)
    print(f"Number of datasets: {num_datasets}")
    print(f"Number of columns: {num_cols}")
    print(f"Average number of rows: {num_rows / num_datasets : .2f}")
    print(f"Average number of columns: {num_cols / num_datasets : .2f}")
    print("-" * 35)

    num_queries, num_answers = 0, 0
    queries = dataloader.get_queries()
    for query_id in queries:
        num_queries += 1
        num_answers += len(queries[query_id])
    
    assert(num_queries == len(queries))
    print(f"Number of queries: {num_queries}")
    print(f"Average number of answers: {num_answers / num_queries : .2f}")
    print("=" * 50)


if __name__ == "__main__":
    """
    Test loading Pylon models
    """
    # ckpt_path = "/data/pylon_models/420_wte_cl_epoch_9.ckpt"
    # model = load_pylon_wte_model(ckpt_path)

    """
    Test CSV dataloader
    """
    testbed = "testbedXS"
    dataset_dir = f"/data/nextiaJD_datasets/{testbed}/datasets/"
    metadata_path = f"/data/nextiaJD_datasets/{testbed}/datasetInformation_{testbed}.csv"
    ground_truth_path = f"/data/nextiaJD_datasets/{testbed}/groundTruth_{testbed}.csv"
    
    dataloader = NextiaJDCSVDataLoader(dataset_dir, metadata_path, ground_truth_path)
    compute_basic_statistics(dataloader)

    """
    Test Spark dataloader
    """
    # dataset_dir = "/data/nextiaJD_datasets/testbedS/datasets/"
    # metadata_path = "/data/nextiaJD_datasets/testbedS/datasetInformation_testbedS.csv"
    # ground_truth_path = "/data/nextiaJD_datasets/testbedS/groundTruth_testbedS.csv"
    # dataloader = SparkDataLoader(dataset_dir, metadata_path, ground_truth_path)
    
    # table = dataloader.read_table("makemytrip_com-travel_sample.csv")
    # print(table.columns)

    """
    Test data loader of Sigma Sample Database
    """
    # dataset_dir = "/data/sigma_sample_data_artifact/db_csv/"
    # metadata_path = "/data/sigma_sample_data_artifact/metadata_tmp.csv"
    # dataloader = SigmaExampleCSVDataLoader(dataset_dir, metadata_path)
    # compute_basic_statistics(dataloader)

    """
    Test Spider dataloader
    """
    # dataset_dir = "/data/spider_artifact/db_csv_extended/"
    # metadata_path = "/data/spider_artifact/dev_join_data_extended.csv"
    # dataloader = SpiderCSVDataLoader(dataset_dir, metadata_path)
    # compute_basic_statistics(dataloader)
