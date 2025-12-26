import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import platform

import yaml
from ultralytics import YOLO


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """
    Professional YOLOv8 training system with validation and monitoring.
    
    Handles complete training pipeline from dataset validation to
    model training with comprehensive error handling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration dictionary containing:
                - data: Path to data.yaml
                - weights: Initial model weights
                - epochs: Number of training epochs
                - imgsz: Image size for training
                - batch: Batch size
                - project: Project directory
                - name: Experiment name
                - device: Device to use (cpu/cuda/auto)
                - workers: Number of dataloader workers
                - exist_ok: Whether to overwrite existing experiments
        """
        self.config = config
        self.data_path = Path(config['data'])
        self.weights = config['weights']
        self.device = self._determine_device(config.get('device', ''))
        self.model: Optional[YOLO] = None
        
    def _determine_device(self, user_device: str) -> str:
        """
        Determine the best device to use for training.
        
        Args:
            user_device: User-specified device ('cpu', 'cuda', or '' for auto)
            
        Returns:
            str: Device string to use
        """
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
            logger.warning("Training will be significantly slower")
            return "cpu"
            
        except ImportError:
            logger.warning("PyTorch not found, defaulting to CPU")
            return "cpu"
        except Exception as exc:
            logger.error(f"Error detecting device: {exc}")
            return "cpu"
    
    def _validate_dataset_config(self) -> Dict[str, Any]:
        """
        Validate dataset YAML configuration file.
        
        Returns:
            Dict: Parsed dataset configuration
            
        Raises:
            FileNotFoundError: If data.yaml doesn't exist
            ValueError: If configuration is invalid
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found: {self.data_path.resolve()}\n"
                f"Please create {self.data_path} as described in the documentation."
            )
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML format in {self.data_path}: {exc}") from exc
        except Exception as exc:
            raise ValueError(f"Failed to read {self.data_path}: {exc}") from exc
        
        # Validate required fields
        required_fields = ['train', 'val', 'nc', 'names']
        missing_fields = [field for field in required_fields if field not in config]
        
        if missing_fields:
            raise ValueError(
                f"Missing required fields in {self.data_path}: {', '.join(missing_fields)}\n"
                f"Required fields: {', '.join(required_fields)}"
            )
        
        logger.info("Dataset configuration validated successfully")
        return config
    
    def _validate_dataset_paths(self, config: Dict[str, Any]) -> None:
        """
        Validate that dataset paths exist.
        
        Args:
            config: Dataset configuration dictionary
            
        Raises:
            FileNotFoundError: If train or val paths don't exist
        """
        # Get the base directory from 'path' field in config
        dataset_base = config.get('path', '')
        
        # Resolve dataset base path
        if dataset_base:
            base_path = Path(dataset_base)
            if not base_path.is_absolute():
                # Resolve relative to the data.yaml file's parent or current working directory
                base_path = Path.cwd() / base_path
        else:
            base_path = self.data_path.parent
        
        for split in ['train', 'val']:
            raw_path = config[split]
            path = Path(raw_path)
            
            # Resolve relative paths
            if not path.is_absolute():
                path = base_path / path
            
            if not path.exists():
                raise FileNotFoundError(
                    f"Dataset path does not exist:\n"
                    f"  Split: {split}\n"
                    f"  Specified: {raw_path}\n"
                    f"  Resolved: {path.resolve()}\n"
                    f"Please check your {self.data_path} configuration."
                )
            
            # Check if path contains images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            images = [f for f in path.rglob('*') if f.suffix.lower() in image_extensions]
            
            if not images:
                logger.warning(f"No images found in {split} directory: {path}")
            else:
                logger.info(f"Found {len(images)} images in {split} set")
        
        logger.info("Dataset paths validated successfully")
    
    def _load_model(self) -> YOLO:
        """
        Load or create YOLO model.
        
        Returns:
            YOLO: Loaded model instance
            
        Raises:
            RuntimeError: If model fails to load
        """
        logger.info(f"Loading model: {self.weights}")
        
        try:
            model = YOLO(self.weights)
            logger.info("Model loaded successfully")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{self.weights}': {exc}\n"
                f"Make sure the file exists or use a pretrained model like 'yolov8n.pt'"
            ) from exc
    
    def _print_training_info(self) -> None:
        """Print comprehensive training configuration."""
        logger.info("="*60)
        logger.info("TRAINING CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Dataset:        {self.config['data']}")
        logger.info(f"Initial weights: {self.config['weights']}")
        logger.info(f"Epochs:         {self.config['epochs']}")
        logger.info(f"Image size:     {self.config['imgsz']}x{self.config['imgsz']}")
        logger.info(f"Batch size:     {self.config['batch']}")
        logger.info(f"Device:         {self.device}")
        logger.info(f"Workers:        {self.config['workers']}")
        logger.info(f"Save to:        {self.config['project']}/{self.config['name']}")
        logger.info(f"Platform:       {platform.system()} {platform.release()}")
        logger.info("="*60)
    
    def validate(self) -> bool:
        """
        Run complete validation before training.
        
        Returns:
            bool: True if validation passed
        """
        logger.info("Starting pre-training validation...")
        
        try:
            # Validate dataset configuration
            config = self._validate_dataset_config()
            
            # Validate dataset paths
            self._validate_dataset_paths(config)
            
            # Validate number of classes
            num_classes = config['nc']
            class_names = config['names']
            
            if len(class_names) != num_classes:
                raise ValueError(
                    f"Mismatch: 'nc' is {num_classes} but 'names' has {len(class_names)} items"
                )
            
            logger.info(f"Training on {num_classes} classes: {', '.join(class_names)}")
            
            logger.info("✓ All validation checks passed")
            return True
            
        except (FileNotFoundError, ValueError) as exc:
            logger.error(f"Validation failed: {exc}")
            return False
    
    def train(self) -> bool:
        """
        Execute model training.
        
        Returns:
            bool: True if training completed successfully
        """
        try:
            # Load model
            self.model = self._load_model()
            
            # Create project directory
            project_path = Path(self.config['project'])
            project_path.mkdir(parents=True, exist_ok=True)
            
            # Print configuration
            self._print_training_info()
            
            # Start training
            logger.info("\nStarting training...\n")
            
            results = self.model.train(
                data=str(self.data_path),
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                project=self.config['project'],
                name=self.config['name'],
                device=self.device,
                workers=self.config['workers'],
                exist_ok=self.config.get('exist_ok', False),
                verbose=True
            )
            
            # Training completed
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY")
            logger.info("="*60)
            
            # Show results location
            save_dir = Path(self.config['project']) / self.config['name']
            logger.info(f"Results saved to: {save_dir}")
            logger.info(f"Best model:      {save_dir / 'weights' / 'best.pt'}")
            logger.info(f"Last model:      {save_dir / 'weights' / 'last.pt'}")
            logger.info("="*60)
            
            return True
            
        except Exception as exc:
            logger.error(f"\nTraining failed: {exc}")
            
            if "CUDA" in str(exc) or "ROCm" in str(exc):
                logger.error("\nGPU Error detected. Try:")
                logger.error("  1. Use --device cpu to force CPU training")
                logger.error("  2. Reduce --batch size")
                logger.error("  3. Update GPU drivers")
            
            return False
    
    def run(self) -> int:
        """
        Run complete training pipeline.
        
        Returns:
            int: Exit code (0 for success, 1 for failure)
        """
        logger.info("VisionGuard Training System")
        logger.info("-" * 60)
        
        # Validation phase
        if not self.validate():
            logger.error("Pre-training validation failed. Aborting.")
            return 1
        
        # Training phase
        if not self.train():
            logger.error("Training failed. Check errors above.")
            return 1
        
        logger.info("\n✨ Training pipeline completed successfully!")
        return 0


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="VisionGuard - Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with default settings
  python train_custom.py --data data/data.yaml
  
  # Train with custom model and 100 epochs
  python train_custom.py --data data/data.yaml --weights yolov8s.pt --epochs 100
  
  # Train on CPU with small batch
  python train_custom.py --data data/data.yaml --device cpu --batch 8
  
  # Resume training with custom experiment name
  python train_custom.py --data data/data.yaml --name my_model_v2 --exist-ok
  
  # Full configuration
  python train_custom.py \\
      --data data/data.yaml \\
      --weights yolov8m.pt \\
      --epochs 200 \\
      --imgsz 640 \\
      --batch 16 \\
      --device cuda \\
      --workers 8 \\
      --project runs/custom \\
      --name experiment_01

Dataset Structure (data.yaml):
  path: ./data
  train: images/train
  val: images/val
  nc: 3
  names: ['class1', 'class2', 'class3']

Output Structure:
  runs/train/exp/
  ├── weights/
  │   ├── best.pt      # Best model checkpoint
  │   └── last.pt      # Last epoch checkpoint
  ├── results.png      # Training metrics plots
  ├── confusion_matrix.png
  └── ...
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset configuration YAML file'
    )
    
    # Model configuration
    parser.add_argument(
        '--weights',
        type=str,
        default='yolov8n.pt',
        help='Initial model weights (default: yolov8n.pt)'
    )
    
    # Training hyperparameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Training image size in pixels (default: 640)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size for training (default: 16)'
    )
    
    # Device configuration
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help="Device: 'cpu', 'cuda', '0,1,2,3', or '' for auto (default: auto)"
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of dataloader workers (default: 4)'
    )
    
    # Output configuration
    parser.add_argument(
        '--project',
        type=str,
        default='runs/train',
        help='Project directory for results (default: runs/train)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='exp',
        help='Experiment name (default: exp)'
    )
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='Overwrite existing experiment directory'
    )
    
    # Additional options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration dictionary
    config = {
        'data': args.data,
        'weights': args.weights,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': args.exist_ok,
    }
    
    # Create and run trainer
    trainer = YOLOTrainer(config)
    exit_code = trainer.run()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()