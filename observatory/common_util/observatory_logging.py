import logging

# import warnings
# warnings.simplefilter(action="ignore", category=FutureWarning)


def custom_logger(logger_name: str, level=logging.INFO):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    format_string = "[%(levelname)s] %(message)s"
    log_format = logging.Formatter(format_string)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode="w")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger
