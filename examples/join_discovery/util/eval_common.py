import os
import shutil
import time

from typing import List, Tuple

from d3l.querying.query_engine import QueryEngine
from tqdm import tqdm

from util.data_loader import CSVDataLoader
from util.warpgate_logging import custom_logger, log_query_and_answer, log_search_results


def aggregate_func(similarity_scores: List[float]) -> float:
    avg_score = sum(similarity_scores) / len(similarity_scores)
    return avg_score


def create_new_directory(path: str, force: bool = False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if force:
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            raise FileNotFoundError


def topk_search_and_eval(query_engine: QueryEngine, dataloader: CSVDataLoader, output_dir: str, k: int) -> Tuple[float, float]:
    queries = dataloader.get_queries()
    
    precision, recall = [], []
    num_queries = 0
    total_lookup_time = 0

    for query_id in tqdm(queries.keys()):
        query_table_name, query_attribute_name = query_id.split("!")
        query_table = dataloader.read_table(table_name=query_table_name)
        try:
            query_attribute = query_table[query_attribute_name]
        except KeyError:
            print("=" * 50)
            print("Missing query attribute in the preprocessed table...")
            print(f"Query table name: {query_table_name}")
            print(f"Query column name: {query_attribute_name}")
            print("=" * 50)
            continue

        # Skip numerical column
        if query_attribute.dtype != "object":
            continue
        
        output_file = os.path.join(output_dir, f"q{num_queries+1}.txt")
        logger = custom_logger(output_file)
        log_query_and_answer(logger, query_table_name, query_attribute_name, answers=queries[query_id])

        # results = query_engine.column_query(table_data=query_table, table_name=query_table_name, query_col_name=query_attribute_name, k=k+1) # +1 to account for the query itself in the search results
        start_time = time.time()
        results = query_engine.column_query(column=query_attribute, k=k+1) # +1 to account for the query itself in the search results
        end_time = time.time()
        total_lookup_time += (end_time - start_time)

        num_corr = 0
        for res in results:
            if res[0] == query_id: continue # Skip the query itself
            
            if res[0] in queries[query_id]:
                num_corr += 1

            candidate_table_name, candidate_attribute_name = res[0].split("!")
            log_search_results(logger, candidate_table_name, candidate_attribute_name, score=res[1][0])

            # for answer in queries[query_id]:
            #     answer_table_name = answer.split("!")[0]
            #     if answer_table_name == candidate_table_name:
            #         num_corr += 1
            #         break

        # num_corr = num_corr if num_corr <= len(queries[query_id]) else len(queries[query_id])
        precision.append(num_corr / k)
        recall.append(num_corr / len(queries[query_id]))
        num_queries += 1

    avg_precision = sum(precision) / num_queries
    avg_recall = sum(recall) / num_queries
    avg_lookup_time = total_lookup_time / num_queries
    return avg_precision, avg_recall, avg_lookup_time, num_queries


def contextual_topk_search_and_eval(query_engine, dataloader: CSVDataLoader, output_dir: str, k: int) -> Tuple[float, float]:
    queries = dataloader.get_queries()
    
    precision, recall = [], []
    num_queries = 0
    total_lookup_time = 0

    for query_id in tqdm(queries.keys()):
        query_table_name, query_attribute_name = query_id.split("!")
        query_table = dataloader.read_table(table_name=query_table_name)
        try:
            query_attribute = query_table[query_attribute_name]
        except KeyError:
            print("=" * 50)
            print(f"Query table name: {query_table_name}")
            print(f"Query column name: {query_attribute_name}")
            print("=" * 50)
            continue

        # Skip numerical column
        if query_attribute.dtype != "object":
            continue
        
        output_file = os.path.join(output_dir, f"q{num_queries+1}.txt")
        logger = custom_logger(output_file)
        log_query_and_answer(logger, query_table_name, query_attribute_name, answers=queries[query_id])

        # results = query_engine.column_query(table_data=query_table, table_name=query_table_name, query_col_name=query_attribute_name, k=k+1) # +1 to account for the query itself in the search results
        start_time = time.time()
        results = query_engine.column_query(table_data=query_table, table_name=query_table_name, query_col_name=query_attribute_name, k=k+1) # +1 to account for the query itself in the search results
        end_time = time.time()
        total_lookup_time += (end_time - start_time)

        num_corr = 0
        for res in results:
            if res[0] == query_id: continue # Skip the query itself
            
            if res[0] in queries[query_id]:
                num_corr += 1

            candidate_table_name, candidate_attribute_name = res[0].split("!")
            log_search_results(logger, candidate_table_name, candidate_attribute_name, score=res[1][0])

            # for answer in queries[query_id]:
            #     answer_table_name = answer.split("!")[0]
            #     if answer_table_name == candidate_table_name:
            #         num_corr += 1
            #         break

        # num_corr = num_corr if num_corr <= len(queries[query_id]) else len(queries[query_id])
        precision.append(num_corr / k)
        recall.append(num_corr / len(queries[query_id]))
        num_queries += 1

    avg_precision = sum(precision) / num_queries
    avg_recall = sum(recall) / num_queries
    avg_lookup_time = total_lookup_time / num_queries
    return avg_precision, avg_recall, avg_lookup_time, num_queries