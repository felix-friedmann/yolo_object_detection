import logging

LOG_LEVEL: int = logging.INFO

def set_log_level(level: int):
    """
    Set the global logging level.
    :param level: The logging level.
    """
    global LOG_LEVEL
    LOG_LEVEL = level
