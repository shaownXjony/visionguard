import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import platform

import yaml
from ultralytics import YOLO


# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """
    Professional YOLOv8 training system with validation and monitoring.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = Path(config["data"])
        self.weights = config["weights"]
        self.device = self._determine_device(config.get("device", ""))
        self.model: Optional[YOLO] = None

    # -------------------- Device --------------------

    def _determine_device(self, user_device: str) -> str:
        if user_device:
            logger.info(f"Using user-specified device: {user_device}")
            return user_device

        try:
            import torch

            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"GPU detected: {gpu_name}")
                logger.info(f"GPU memory: {gpu_memory:.2f} GB")
                return "cuda"

            logger.warning("No GPU detected, using CPU")
            return "cpu"

        except Exception:
            logger.warning("PyTorch not available, defaulting to CPU")
            return "cpu"

    # -------------------- Dataset Validation --------------------

    def _validate_dataset_config(self) -> Dict[str, Any]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        for field in ["train", "val", "nc", "names"]:
            if field not in config:
                raise ValueError(f"Missing '{field}' in dataset config")

        logger.info("Dataset configuration validated")
        return config

    def _validate_dataset_paths(self, config: Dict[str, Any]) -> None:
        base = Path(config.get("path", self.data_path.parent))
        if not base.is_absolute():
            base = Path.cwd() / base

        for split in ["train", "val"]:
            path = base / config[split]
            if not path.exists():
                raise FileNotFoundError(f"{split} path not found: {path}")

        logger.info("Dataset paths validated")

    # -------------------- Model --------------------

    def _load_model(self) -> YOLO:
        logger.info(f"Loading model: {self.weights}")
        return YOLO(self.weights)

    # -------------------- Training --------------------

    def validate(self) -> bool:
        try:
            cfg = self._validate_dataset_config()
            self._validate_dataset_paths(cfg)
            return True
        except Exception as exc:
            logger.error(f"Validation failed: {exc}")
            return False

    def train(self) -> bool:
        try:
            self.model = self._load_model()

            Path(self.config["project"]).mkdir(parents=True, exist_ok=True)

            logger.info("=" * 60)
            logger.info("TRAINING CONFIGURATION")
            logger.info(f"Dataset: {self.config['data']}")
            logger.info(f"Weights: {self.weights}")
            logger.info(f"Epochs: {self.config['epochs']}")
            logger.info(f"Device: {self.device}")
            logger.info("=" * 60)

            self.model.train(
                data=str(self.data_path),
                epochs=self.config["epochs"],
                imgsz=self.config["imgsz"],
                batch=self.config["batch"],
                project=self.config["project"],
                name=self.config["name"],
                device=self.device,
                workers=self.config["workers"],
                exist_ok=self.config.get("exist_ok", False),
                verbose=True,
            )

            logger.info("Training completed successfully")
            return True

        except Exception as exc:
            logger.error(f"Training failed: {exc}")
            return False

    def run(self) -> int:
        logger.info("VisionGuard Training")
        if not self.validate():
            return 1
        if not self.train():
            return 1
        return 0


# -------------------- CLI --------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionGuard - Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--data", required=True, help="Dataset YAML file")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument(
        "--project",
        default="runs/train",
        help="Project directory for results",
    )
    parser.add_argument(
        "--output",
        help="Alias for --project (CLI consistency)",
    )
    parser.add_argument("--name", default="exp")
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # CLI STEP 2: map --output to --project
    if args.output:
        args.project = args.output

    config = {
        "data": args.data,
        "weights": args.weights,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": args.exist_ok,
    }

    trainer = YOLOTrainer(config)
    sys.exit(trainer.run())


if __name__ == "__main__":
    main()
