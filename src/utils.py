import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional
import mlflow
from src import config
from dataclasses import dataclass

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

    file_handler = RotatingFileHandler(f"logs/{filename}",
                                       maxBytes=10*1024*1024,
                                       backupCount=3)
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


@dataclass
class Config:
    """
    Model configuration.
    """
    model: str = "yolo26n.pt"
    data: str = "coco128.yaml"
    epochs: int = 20
    batch: int = 4
    imgsz: int = 640
    save: bool = True
    save_period: int = -1
    device: str = "0"
    name: str = None
    optimizer: str = "auto"
    seed: int = 42
    lr0: float = 0.01
    weight_decay: float = 0.0005
    dropout: float = 0.0

    def to_dict(self):
        """
        Converts the config to dict without model.
        :return: The config dict.
        """
        dic = self.__dict__.copy()
        dic.pop("model", None)
        return dic


def log_training(result, train_config: Config):
    """
    Logs the training result.
    :param result: The training result.
    :param train_config: The training config.
    """

    logger = get_logger(__name__)
    if result is None:
        logger.error("Training result is None")
        exit(1)

    mlflow.log_params(train_config.to_dict())

    metrics = result.results_dict()
    mlflow.log_metrics({
        "mAP50": metrics.get('metrics/mAP50(B)', 0),
        "mAP50-95": metrics.get('metrics/mAP50-95(B)', 0),
        "precision": metrics.get('metrics/precision(B)', 0),
        "recall": metrics.get('metrics/recall(B)', 0)
    })

    if "save_dir" in metrics:
        weights_path = Path(metrics["save_dir"]) / "weights" / "best.pt"
    else:
        weights_path = Path(f"runs/detect/{train_config.name}/weights/best.pt")

    if weights_path.exists():
        mlflow.log_artifact(str(weights_path), artifact_path="model")
    else:
        logger.error(f"Weights not found at {weights_path}")
