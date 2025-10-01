#!/usr/bin/env python3
"""
Independent script for testing trained UBTeacher models on external datasets.
"""

import os
import argparse
import json
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger

from ubteacher import add_ubteacher_config
from ubteacher.engine.trainer import UBTeacherTrainer, BaselineTrainer
from ubteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

# Import required modules for model registration
from ubteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from ubteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from ubteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import ubteacher.data.datasets.builtin


def setup_cfg(config_file, model_weights, opts=None):
    """
    Setup configuration for testing
    """
    cfg = get_cfg()
    add_ubteacher_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    
    # Set model weights
    cfg.MODEL.WEIGHTS = model_weights
    cfg.freeze()
    return cfg


def register_test_dataset(dataset_name, annotation_file, image_dir, thing_classes):
    """
    Register external test dataset
    """
    # Remove existing registration if present
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    
    # Register the dataset
    register_coco_instances(
        dataset_name,
        {},
        annotation_file,
        image_dir
    )
    
    # Set metadata
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    
    # Set class mapping (assuming COCO format with class id starting from 1)
    thing_dataset_id_to_contiguous_id = {i+1: i for i in range(len(thing_classes))}
    MetadataCatalog.get(dataset_name).thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
    
    print(f"Registered dataset: {dataset_name}")
    print(f"Classes: {thing_classes}")
    print(f"Annotation file: {annotation_file}")
    print(f"Image directory: {image_dir}")


def load_model(cfg, trainer_type="ubteacher"):
    """
    Load the trained model
    """
    if trainer_type == "ubteacher":
        # For UBTeacher, we need to load the ensemble model
        Trainer = UBTeacherTrainer
        model = Trainer.build_model(cfg)
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        
        checkpointer = DetectionCheckpointer(ensem_ts_model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        
        # Return the teacher model for inference
        return ensem_ts_model.modelTeacher
        
    elif trainer_type == "baseline":
        Trainer = BaselineTrainer
        model = Trainer.build_model(cfg)
        
        checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        
        return model
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")


def run_inference(cfg, model, dataset_name, output_dir=None):
    """
    Run inference on the test dataset
    """
    # Build test data loader
    test_loader = build_detection_test_loader(cfg, dataset_name)
    
    # Setup evaluator
    evaluator = COCOEvaluator(
        dataset_name,
        output_dir=output_dir,
        use_fast_impl=False
    )
    
    # Run inference
    print(f"Running inference on dataset: {dataset_name}")
    results = inference_on_dataset(model, test_loader, evaluator)
    
    return results


def save_results(results, output_file):
    """
    Save inference results to file
    """
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test UBTeacher model on external dataset")
    
    # Required arguments
    parser.add_argument("--config-file", required=True,
                       help="Path to config file used for training")
    parser.add_argument("--model-weights", required=True,
                       help="Path to trained model weights (.pth file)")
    parser.add_argument("--test-dataset-name", required=True,
                       help="Name for the test dataset")
    parser.add_argument("--test-annotations", required=True,
                       help="Path to test dataset COCO format annotations JSON file")
    parser.add_argument("--test-images", required=True,
                       help="Path to test dataset images directory")
    
    # Optional arguments
    parser.add_argument("--trainer-type", default="ubteacher", choices=["ubteacher", "baseline"],
                       help="Type of trainer used (default: ubteacher)")
    parser.add_argument("--thing-classes", nargs="+", default=["tumor"],
                       help="List of class names (default: ['tumor'])")
    parser.add_argument("--output-dir", default="./test_output",
                       help="Directory to save test results")
    parser.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                       help="Additional config options")
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger(name="ubteacher")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("UBTeacher External Dataset Testing")
    print("="*50)
    print(f"Config file: {args.config_file}")
    print(f"Model weights: {args.model_weights}")
    print(f"Test dataset: {args.test_dataset_name}")
    print(f"Trainer type: {args.trainer_type}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    # Setup configuration
    cfg = setup_cfg(args.config_file, args.model_weights, args.opts)
    default_setup(cfg, args)
    
    # Register test dataset
    register_test_dataset(
        args.test_dataset_name,
        args.test_annotations,
        args.test_images,
        args.thing_classes
    )
    
    # Load model
    print("Loading model...")
    model = load_model(cfg, args.trainer_type)
    model.eval()
    
    # Run inference
    print("Starting inference...")
    results = run_inference(cfg, model, args.test_dataset_name, args.output_dir)
    
    # Save results
    results_file = os.path.join(args.output_dir, f"{args.test_dataset_name}_results.json")
    save_results(results, results_file)
    
    # Print summary
    print("\n" + "="*50)
    print("INFERENCE COMPLETED")
    print("="*50)
    print("Results Summary:")
    for key, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()