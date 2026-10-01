import torch
from torch.utils.data import DataLoader
from typing import Dict
from src.training.dataset import FIFAPlayerDataset
import pandas as pd
import numpy as np
from typing import Union

def create_dataloaders(
    X_dict: Dict[str, Union[pd.DataFrame, np.ndarray]],
    y_dict: Dict[str, Union[pd.Series, pd.DataFrame, np.ndarray]],
    batch_size: int = 32,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for training, validation, and testing.

    Args:
        X_dict: Dictionary containing 'train', 'val', and 'test' features.
        y_dict: Dictionary containing 'train', 'val', and 'test' targets.
        batch_size: Configurable batch size.
        num_workers: Number of workers for data loading (0 means main process).

    Returns:
        Dict[str, DataLoader]: Dictionary containing DataLoaders mapped by 'train', 'val', 'test'.
    """
    dataloaders = {}

    # Train DataLoader (shuffled)
    if 'train' in X_dict and 'train' in y_dict:
        train_dataset = FIFAPlayerDataset(X_dict['train'], y_dict['train'])
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    # Validation DataLoader (not shuffled)
    if 'val' in X_dict and 'val' in y_dict:
        val_dataset = FIFAPlayerDataset(X_dict['val'], y_dict['val'])
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    # Test DataLoader (not shuffled)
    if 'test' in X_dict and 'test' in y_dict:
        test_dataset = FIFAPlayerDataset(X_dict['test'], y_dict['test'])
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return dataloaders
