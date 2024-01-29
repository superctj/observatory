import argparse
import logging
import os
import sys

from d3l.querying.query_engine import QueryEngine
from d3l.utils.functions import pickle_python_object, unpickle_python_object

sys.path.append("/home/congtj/observatory/applications/join_discovery")
from d3l_extension import BertEmbeddingIndex
from util.data_loader import NextiaJDCSVDataLoader
from util.eval_common import create_new_directory, topk_search_and_eval
from util.warpgate_logging import custom_logger, log_args_and_metrics


def create_or_load_index(index_dir: str, index_name: str, dataloader: NextiaJDCSVDataLoader, args: argparse.Namespace) -> BertEmbeddingIndex:
    index_path = os.path.join(index_dir, f"{index_name}.lsh")

    if os.path.exists(index_path):
        embedding_index = unpickle_python_object(index_path)
        print(f"{index_name} Embedding Index: LOADED!")
    else:
        print(f"{index_name} Embedding Index: STARTED!")
        
        embedding_index = BertEmbeddingIndex(
            model_name=index_name.split("_")[0],
            dataloader=dataloader,
            num_samples=args.num_samples,
            index_similarity_threshold=args.lsh_threshold
        )

        pickle_python_object(embedding_index, index_path)
        print(f"{index_name} Embedding Index: SAVED!")
    
    return embedding_index


def main(args):
    # Create CSV data loader
    dataloader = NextiaJDCSVDataLoader(
        dataset_dir=args.dataset_dir, 
        metadata_path=args.metadata_path,
        ground_truth_path=args.ground_truth_path
    )

    # Create or load embedding index
    index_name = f"{args.model_name}_lsh_{args.lsh_threshold}_samples_{args.num_samples}"
    embedding_index = create_or_load_index(
        args.index_dir, index_name, dataloader, args
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
    parser.add_argument("--ground_truth_path", type=str, default="", help="")
    
    parser.add_argument("--index_dir", type=str, default="", help="")
    parser.add_argument("--output_dir", type=str, default="", help="")
    
    parser.add_argument("--lsh_threshold", type=float, default=0.7, help="")
    parser.add_argument("--top_k", type=int, default=1, help="")

    parser.add_argument("--model_name", type=str, default="", help="")
    parser.add_argument("--num_samples", type=int, default=-1, help="Maximum number of rows to sample from each table to construct embeddings.")

    main(parser.parse_args())