import logging
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

from argparse import Namespace
from typing import List, Tuple


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


def log_query_and_answer(logger: logging.Logger, query_table_name: str, query_attribute_name: str, answers: List[str]):
    logger.info(f"Query table: {query_table_name}")
    logger.info(f"Query attribute: {query_attribute_name}")

    for answer_id in answers:
        logger.info("-" * 50)
        answer_table_name, answer_attribute_name = answer_id.split("!")
        logger.info(f"Query answer table: {answer_table_name}")
        logger.info(f"Query answer attribute: {answer_attribute_name}")

    logger.info("=" * 80)


def log_search_results(logger: logging.Logger, candidate_table_name: str, candidate_attribute_name: str, score: float):
    logger.info(f"Candidate table: {candidate_table_name}")
    logger.info(f"Candidate attribute: {candidate_attribute_name}")
    logger.info(f"Candidate score: {score}")
    logger.info("-" * 50)


def log_args_and_metrics(logger: logging.Logger, args: Namespace, metrics: Tuple[float, float]):
    logger.info(args)
    logger.info("=" * 50)
    logger.info(f"Top-{args.top_k} search")
    logger.info(f"  Number of queries: {metrics[3]}")
    logger.info(f"  Precision: {metrics[0] : .2f}")
    logger.info(f"  Recall: {metrics[1] : .2f}")
    logger.info(f"  Lookup time: {metrics[2] : .2f} s/query")