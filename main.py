import argparse
import logging
from dataclasses import fields
import mlflow
from ultralytics import settings
from src import config
from src import utils
from src import yolo

def main():
    # CLI Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Log level")
    parser.add_argument("--model", type=str, default=None, help="Model name")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--name", type=str, required=True, help="Name of run")
    parser.add_argument("--optimizer", type=str, default=None, help="Optimizer")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout")
    args = parser.parse_args()

    # Logger
    config.set_log_level(level=getattr(logging, args.log_level))
    logger = utils.get_logger(__name__)

    # MLflow (no autolog possible)
    settings.update({"mlflow": False})
    mlflow.set_experiment("yolo-object-detection")

    # Load config
    config_fields = {f.name for f in fields(utils.Config)}
    overrides = {k: v for k, v in vars(args).items() if v is not None and k in config_fields}
    logger.info(f"Loading configurations: {overrides}")
    train_config = utils.Config(**overrides)

    with mlflow.start_run(run_name=args.name):
        logger.info(f"Starting model training, run {args.name}")

        # YOLO
        model, result = yolo.get_model(train_config)

        # log training
        logger.info("Logging training data...")
        utils.log_training(result, train_config)

if __name__ == '__main__':
    main()
