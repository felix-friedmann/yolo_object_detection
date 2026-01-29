import logging
import os
from src import config

def get_logger(name: str, file: str=None) -> logging.Logger:
    """
    Creates a logger with the given name or returns the existing logger.
    :param name: The name of the logger.
    :param file: The file to write the logger to. Default: None.
    :return: The logger object.
    """

    level = config.LOG_LEVEL

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Logger already exists
    if logger.hasHandlers():
        return logger

    os.makedirs(f"logs", exist_ok=True)

    # File handler
    if file is not None:
        filename = file if file.endswith(".log") else file + ".log"
    else:
        filename = name.split(".")[-1] + ".log"

    file_handler = logging.FileHandler(f"logs/{filename}")
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Format
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
