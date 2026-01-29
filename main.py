import argparse
import logging
from src import config
from src import utils
from src import yolo

def main():
    # CLI Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    # Logger
    config.set_log_level(level=getattr(logging, args.log_level))
    logger = utils.get_logger(__name__)
    logger.info("Starting main loop")

    # Model
    model = yolo.get_model()

if __name__ == '__main__':
    main()
