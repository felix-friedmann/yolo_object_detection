from ultralytics import YOLO

def get_model() -> YOLO:
    """
    Load pretrained YOLO model and train it further with another dataset.
    :return: The trained model.
    """

    # alternatively .yaml for just architecture
    model = YOLO("yolo26n.pt")
    # other dataset
    model.train(data="coco.yaml", epochs=50)

    return model
