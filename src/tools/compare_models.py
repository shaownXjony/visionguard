"""
VisionGuard Model Comparison Tool.

Compare detection results from different YOLO models side-by-side.
Useful for evaluating trained models against pre-trained baselines.

Author: ShaownJony
Project: VisionGuard
License: MIT
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Compare detection results from multiple YOLO models."""
    
    def __init__(self, image_path: str, model_paths: List[str], conf: float = 0.5):
        """
        Initialize comparator.
        
        Args:
            image_path: Path to test image
            model_paths: List of model weight paths
            conf: Confidence threshold
        """
        self.image_path = Path(image_path)
        self.model_paths = model_paths
        self.conf = conf
        
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Comparing {len(model_paths)} models on: {image_path}")
    
    def _load_models(self) -> List[Tuple[str, YOLO]]:
        """Load all models."""
        models = []
        for path in self.model_paths:
            try:
                logger.info(f"Loading model: {path}")
                model = YOLO(path)
                model_name = Path(path).stem
                models.append((model_name, model))
            except Exception as exc:
                logger.error(f"Failed to load {path}: {exc}")
        return models
    
    def _run_inference(self, models: List[Tuple[str, YOLO]]) -> List[Tuple[str, object]]:
        """Run inference with all models."""
        image = cv2.imread(str(self.image_path))
        results = []
        
        for name, model in models:
            logger.info(f"Running inference with {name}...")
            result = model(image, conf=self.conf, verbose=False)[0]
            results.append((name, result))
            
            num_detections = len(result.boxes)
            logger.info(f"  {name}: {num_detections} objects detected")
        
        return results
    
    def _create_comparison_grid(self, results: List[Tuple[str, object]]) -> np.ndarray:
        """Create side-by-side comparison grid."""
        annotated_images = []
        
        for name, result in results:
            annotated = result.plot()
            
            # Add model name at top
            cv2.putText(
                annotated,
                name,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
            )
            
            # Add detection count
            num_objects = len(result.boxes)
            cv2.putText(
                annotated,
                f"Objects: {num_objects}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            
            annotated_images.append(annotated)
        
        # Get max dimensions
        max_height = max(img.shape[0] for img in annotated_images)
        max_width = max(img.shape[1] for img in annotated_images)
        
        # Resize all images to same dimensions
        resized_images = []
        for img in annotated_images:
            # Pad image to match max dimensions
            h, w = img.shape[:2]
            if h < max_height or w < max_width:
                padded = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                padded[:h, :w] = img
                resized_images.append(padded)
            else:
                resized_images.append(img)
        
        # Stack images side by side
        if len(resized_images) == 1:
            return resized_images[0]
        elif len(resized_images) == 2:
            return np.hstack(resized_images)
        else:
            # Create grid for 3+ images
            rows = []
            for i in range(0, len(resized_images), 2):
                if i + 1 < len(resized_images):
                    rows.append(np.hstack([resized_images[i], resized_images[i+1]]))
                else:
                    # Pad last image if odd number
                    last_img = resized_images[i]
                    empty = np.zeros_like(last_img)
                    rows.append(np.hstack([last_img, empty]))
            return np.vstack(rows)
    
    def _print_detailed_comparison(self, results: List[Tuple[str, object]]) -> None:
        """Print detailed comparison statistics."""
        logger.info("\n" + "="*70)
        logger.info("DETAILED COMPARISON")
        logger.info("="*70)
        
        for name, result in results:
            boxes = result.boxes
            logger.info(f"\nModel: {name}")
            logger.info("-" * 70)
            logger.info(f"Total detections: {len(boxes)}")
            
            if len(boxes) > 0:
                # Group by class
                class_counts = {}
                confidences = []
                
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    confidence = float(box.conf[0])
                    
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    confidences.append(confidence)
                
                # Print class distribution
                logger.info("\nClass Distribution:")
                for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    logger.info(f"  {class_name}: {count}")
                
                # Print confidence statistics
                avg_conf = sum(confidences) / len(confidences)
                min_conf = min(confidences)
                max_conf = max(confidences)
                
                logger.info(f"\nConfidence Statistics:")
                logger.info(f"  Average: {avg_conf:.3f}")
                logger.info(f"  Min: {min_conf:.3f}")
                logger.info(f"  Max: {max_conf:.3f}")
            else:
                logger.info("  No objects detected")
        
        logger.info("\n" + "="*70)
    
    def compare(self, output_dir: str = "outputs/comparisons") -> None:
        """
        Run complete comparison.
        
        Args:
            output_dir: Directory to save comparison results
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load models
        models = self._load_models()
        if not models:
            logger.error("No models loaded successfully")
            return
        
        # Run inference
        results = self._run_inference(models)
        
        # Create comparison grid
        comparison_grid = self._create_comparison_grid(results)
        
        # Save comparison
        output_file = output_path / f"comparison_{self.image_path.stem}.jpg"
        cv2.imwrite(str(output_file), comparison_grid)
        logger.info(f"\nComparison saved: {output_file}")
        
        # Print detailed statistics
        self._print_detailed_comparison(results)
        
        # Display (optional)
        logger.info("\nPress any key to close the comparison window...")
        cv2.imshow("Model Comparison", comparison_grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="VisionGuard Model Comparison Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare pre-trained vs trained model
  python compare_models.py --image test.jpg --models yolov8n.pt runs/train/exp/weights/best.pt
  
  # Compare multiple models
  python compare_models.py --image photo.jpg --models yolov8n.pt yolov8s.pt yolov8m.pt
  
  # Compare trained models
  python compare_models.py --image test.jpg --models runs/train/exp1/weights/best.pt runs/train/exp2/weights/best.pt
  
  # With custom confidence threshold
  python compare_models.py --image test.jpg --models model1.pt model2.pt --conf 0.6
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to test image'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to model weights (space-separated)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='Confidence threshold (default: 0.5)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/comparisons',
        help='Output directory (default: outputs/comparisons)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    comparator = ModelComparator(
        image_path=args.image,
        model_paths=args.models,
        conf=args.conf
    )
    
    comparator.compare(output_dir=args.output)
    
    logger.info("\n✨ Comparison complete!")


if __name__ == "__main__":
    main()