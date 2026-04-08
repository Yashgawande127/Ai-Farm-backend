#!/usr/bin/env python3
"""
Training script for all crop prediction models including deep learning models
Run this script to train DeepAgYieldNet, ShuffleNetV2, EfficientCapsNet, LAD-Net, and RegNet
"""

import os
import sys
import logging

# Add current directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from model_trainer import train_all_models_comprehensive

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    try:
        logger.info("=== Starting Comprehensive Model Training ===")
        logger.info("Training traditional ML and deep learning models...")
        
        # Train all models
        results = train_all_models_comprehensive()
        
        logger.info("=== Training Complete ===")
        logger.info(f"Best model: {results.get('best_model', 'Unknown')}")
        logger.info(f"Best accuracy: {results.get('best_accuracy', 'Unknown'):.4f}")
        
        # Print model accuracies
        logger.info("\n=== Model Accuracies ===")
        for model_name, model_data in results.get('all_models', {}).items():
            accuracy = model_data.get('accuracy', 0)
            model_type = model_data.get('model_type', 'Unknown')
            logger.info(f"{model_name}: {accuracy:.4f} ({model_type})")
        
        logger.info("\n=== Deep Learning Results ===")
        dl_results = results.get('deep_learning_results', {})
        for model_name, result in dl_results.items():
            if 'accuracy' in result:
                logger.info(f"{model_name}: {result['accuracy']:.4f}")
            else:
                logger.info(f"{model_name}: Training failed - {result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()
    
    print("\n" + "="*60)
    print("CROP YIELD PREDICTION MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Models trained: {len(results.get('all_models', {}))}")
    print(f"Best performing model: {results.get('best_model', 'Unknown')}")
    print(f"Best accuracy: {results.get('best_accuracy', 0):.1%}")
    print("\nModels available for crop yield prediction:")
    print("- Random Forest (Traditional ML)")
    print("- Decision Tree (Traditional ML)")
    print("- DeepAgYieldNet (Deep Learning)")
    print("- ShuffleNet V2 (Deep Learning)")
    print("- EfficientCapsNet (Deep Learning)")  
    print("- LAD-Net (Deep Learning)")
    print("- RegNet (Deep Learning)")
    print("\nRun the backend server to use these models for predictions!")