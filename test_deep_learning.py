#!/usr/bin/env python3
"""
Test script to verify deep learning model implementation
Run this to test the models with sample data
"""

import numpy as np
import os
import sys

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from deep_learning_models import (
            DeepAgYieldNet, ShuffleNetV2, EfficientCapsNet, 
            LADNet, RegNet, DeepLearningTrainer
        )
        print("✓ Deep learning models imported successfully")
    except ImportError as e:
        print(f"✗ Deep learning models import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        print("✓ Standard ML libraries imported successfully")
    except ImportError as e:
        print(f"✗ Standard ML libraries import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        from deep_learning_models import (
            DeepAgYieldNet, ShuffleNetV2, EfficientCapsNet, 
            LADNet, RegNet
        )
        
        input_size = 7  # Number of input features
        num_classes = 22  # Number of crop classes
        
        # Test each model
        models = {
            'DeepAgYieldNet': DeepAgYieldNet(input_size, num_classes),
            'ShuffleNetV2': ShuffleNetV2(input_size, num_classes),
            'EfficientCapsNet': EfficientCapsNet(input_size, num_classes),
            'LADNet': LADNet(input_size, num_classes),
            'RegNet': RegNet(input_size, num_classes)
        }
        
        for name, model in models.items():
            print(f"✓ {name} created successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test if models can perform forward pass"""
    print("\nTesting forward pass...")
    
    try:
        import torch
        from deep_learning_models import DeepLearningTrainer
        
        # Create sample data
        batch_size = 10
        input_size = 7
        X_sample = torch.randn(batch_size, input_size)
        
        # Test DeepAgYieldNet
        trainer = DeepLearningTrainer('deepagyieldnet')
        model = trainer.create_model(input_size, 22)
        
        with torch.no_grad():
            output = model(X_sample)
            print(f"✓ Forward pass successful. Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_training_pipeline():
    """Test the training pipeline with sample data"""
    print("\nTesting training pipeline...")
    
    try:
        from deep_learning_models import DeepLearningTrainer
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 7
        X_sample = np.random.randn(n_samples, n_features)
        y_sample = np.random.choice(['rice', 'wheat', 'cotton', 'maize'], n_samples)
        
        # Test training with a small model
        trainer = DeepLearningTrainer('deepagyieldnet')
        
        # Quick training test (just 5 epochs)
        result = trainer.train_model(X_sample, y_sample, epochs=5, batch_size=16)
        
        print(f"✓ Training pipeline successful. Accuracy: {result['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        return False

def main():
    """Main test function"""
    print("="*60)
    print("DEEP LEARNING MODELS TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    tests = [
        test_imports,
        test_model_creation,
        test_forward_pass,
        test_training_pipeline
    ]
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("Deep learning models are ready for crop yield prediction.")
        print("\nTo train all models, run: python train_models.py")
    else:
        print("✗ SOME TESTS FAILED!")
        print("Please check the error messages above and install missing dependencies.")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)