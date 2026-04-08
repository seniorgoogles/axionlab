#!/usr/bin/env python
"""Example script for training JSC-2L model.

This demonstrates how to use the unified Model class with the JSC dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

from src.engine.config import ModelConfig
from src.engine.trainer import Model
from src.engine.config.config_reader import ConfigReader
from src.engine.config.config_validator import ConfigValidator
from src.utils.dataset import JSCDataset


def load_jsc_data(config):
    """Load JSC dataset from config."""
    dataset_root = Path(config.dataset_root_path)

    if not dataset_root.exists():
        print(f"Dataset not found at {dataset_root}")
        print("Creating dummy data for demonstration...")

        # Create dummy data for demonstration
        num_samples = 10000
        features = np.random.randn(num_samples, 16).astype(np.float32)
        labels = np.random.randint(0, config.nc, size=num_samples).astype(np.int64)

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(labels)
        )
    else:
        print(f"Loading dataset from {dataset_root}")
        dataset = JSCDataset(dataset_root)

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size[0],
        shuffle=True,
        num_workers=config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size[1],
        shuffle=False,
        num_workers=config.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size[1],
        shuffle=False,
        num_workers=config.num_workers
    )

    return train_loader, val_loader, test_loader


def main():
    """Main training loop."""
    print("=" * 60)
    print("JSC-2L Model Training Example")
    print("=" * 60)

    # Load and validate config
    config_path = Path("configs/experiments/jsc/train_jsc_2l.yaml")
    config_dict = ConfigReader.read_config(config_path)

    # Validate config
    config = ConfigValidator.validate_and_create(config_dict, ModelConfig)

    print(f"Model name: {config.model_name}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Dataset root: {config.dataset_root_path}")
    print(f"Number of classes: {config.nc}")

    # Create model
    print("\nBuilding model...")
    model = Model(config)

    # Build the model (create instance of model_type with input_shape)
    # For JSC, we use the JSCDataset to get input shape
    try:
        from src.utils.dataset import JSCDataset
        dummy_dataset = JSCDataset(Path(config.dataset_root_path))
        input_shape = dummy_dataset[0][0].shape
        print(f"Input shape: {input_shape}")
    except:
        input_shape = (16,)  # Default for JSC
        print(f"Using default input shape: {input_shape}")

    model.build(model_type=nn.Sequential, input_shape=input_shape)

    print(f"Model: {model}")

    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = load_jsc_data(config)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Define optimizer and criterion
    optimizer = torch.optim.Adam(model.model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train
    print("\nStarting training...")
    try:
        results = model.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            num_epochs=config.epochs,
            validate_every=10
        )

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation accuracy: {results.get('best_val_accuracy', 'N/A'):.4f}")
        print("=" * 60)

        # Save checkpoint
        print("\nSaving checkpoint...")
        checkpoint_path = model.save_checkpoint()
        print(f"Checkpoint saved to: {checkpoint_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()