import argparse
import logging
import os
import sys

from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object

from d3l_extension import BertEmbeddingIndex
sys.path.append("/home/ubuntu/join_discovery/")
from util.data_loader import SpiderCSVDataLoader
from util.eval_common import create_new_directory, topk_search_and_eval
from util.logging import custom_logger, log_args_and_metrics


def create_or_load_index(index_dir: str, index_name: str, dataloader: SpiderCSVDataLoader, lsh_threshold: float) -> BertEmbeddingIndex:
    index_path = os.path.join(index_dir, f"{index_name}.lsh")

    if os.path.exists(index_path):
        embedding_index = unpickle_python_object(index_path)
        print(f"{index_name} Embedding Index: LOADED!")
    else:
        print(f"{index_name} Embedding Index: STARTED!")
        
        embedding_index = BertEmbeddingIndex(
            model_name=index_name.split("_")[0],
            dataloader=dataloader,
            index_similarity_threshold=lsh_threshold
        )

        pickle_python_object(embedding_index, index_path)
        print(f"{index_name} Embedding Index: SAVED!")
    
    return embedding_index


# def topk_search_and_eval(dataloader: SpiderCSVDataLoader, embedding_index: BertEmbeddingIndex, output_dir: str, k: int) -> Tuple[float, float]:
#     qe = QueryEngine(embedding_index)
#     queries = dataloader.get_queries()
    
#     precision, recall = [], []
#     num_queries = 0

#     for query_id in tqdm(queries.keys()):
#         query_table_name, query_attribute_name = query_id.split("!")
#         query_table = dataloader.read_table(table_name=query_table_name)
#         query_attribute = query_table[query_attribute_name].dropna()

#         # Skip numerical column
#         if query_attribute.dtype != "object":
#             continue
        
#         output_file = os.path.join(output_dir, f"q{num_queries+1}.txt")
#         logger = custom_logger(output_file, level=logging.INFO)
#         log_query_and_answer(logger, query_table_name, query_attribute_name, answers=queries[query_id])

#         results = qe.column_query(column=query_attribute, k=k)
#         num_corr = 0

#         for res in results:
#             # if candidate_table == query.answer_table and candidate_attribute == query.answer_attribute:
#             if res[0] in queries[query_id] or res[0] == query_id:
#                 num_corr += 1
#                 candidate_table_name, candidate_attribute_name = res[0].split("!")
#                 log_search_results(logger, candidate_table_name, candidate_attribute_name, score=res[1])
        
#         precision.append(num_corr / k)
#         recall.append(num_corr / (len(queries[query_id])+1))
#         num_queries += 1

#     avg_precision = sum(precision) / num_queries
#     avg_recall = sum(recall) / num_queries
#     return avg_precision, avg_recall


def main(args):
    # Create CSV data loader
    dataloader = SpiderCSVDataLoader(
        dataset_dir=args.dataset_dir, 
        metadata_path=args.metadata_path
    )

    # Create or load embedding index
    index_name = f"{args.model_name}_lsh_{args.lsh_threshold}"
    embedding_index = create_or_load_index(
        args.index_dir, index_name, dataloader, args.lsh_threshold
    )
    
    # Create output directory (overwrite the directory if exists)
    output_dir = os.path.join(
        args.output_dir, f"{index_name}_topk_{args.top_k}"
    )
    create_new_directory(output_dir, force=True)

    # Top-k search and evaluation
    query_engine = QueryEngine(embedding_index)
    metrics = topk_search_and_eval(query_engine, dataloader, output_dir, args.top_k)

    # Log command-line arguments for reproduction and metrics
    meta_log_file = os.path.join(output_dir, "log.txt")
    meta_logger = custom_logger(meta_log_file, level=logging.INFO)
    log_args_and_metrics(meta_logger, args, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSH for Top-K Column Search",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset_dir", type=str, default="", help="")
    parser.add_argument("--metadata_path", type=str, default="", help="")
    
    parser.add_argument("--index_dir", type=str, default="", help="")
    parser.add_argument("--output_dir", type=str, default="", help="")
    
    parser.add_argument("--lsh_threshold", type=float, default=0.7, help="")
    parser.add_argument("--top_k", type=int, default=1, help="")

    parser.add_argument("--model_name", type=str, default="", help="")
    parser.add_argument("--num_samples", type=int, default=-1, help="Maximum number of rows to sample from each table to construct embeddings.")

    main(parser.parse_args())