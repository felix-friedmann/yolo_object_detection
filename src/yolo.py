from typing import Optional
from ultralytics import YOLO
from src.utils import Config

def get_model(config: Config) -> tuple[YOLO, Optional[dict]]:
    """
    Load pretrained YOLO model and train it further with another dataset.
    :param config: The model configuration.
    :return: The trained model.
    """

    model = YOLO(config.model)
    # ** dictionary unpacking
    result = model.train(**config.to_dict())

    return model, result
