import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Union, Tuple

class FIFAPlayerDataset(Dataset):
    """
    PyTorch Dataset for FIFA Player Overall Rating Prediction.
    Expects preprocessed numerical features and targets.
    """
    def __init__(self, features: Union[pd.DataFrame, np.ndarray], targets: Union[pd.Series, pd.DataFrame, np.ndarray]):
        """
        Args:
            features: Preprocessed numerical feature data.
            targets: Corresponding target values (overall ratings).
        """
        # Convert to numpy arrays if they are pandas structures
        if isinstance(features, pd.DataFrame):
            features = features.to_numpy()
        if isinstance(targets, (pd.Series, pd.DataFrame)):
            targets = targets.to_numpy()

        # Ensure dimensions match
        if len(features) != len(targets):
            raise ValueError(f"Number of feature rows ({len(features)}) does not match target rows ({len(targets)})")

        # Convert to torch float32 tensors
        self.X = torch.tensor(features, dtype=torch.float32)

        # Ensure targets are shaped properly (N, 1) for regression
        self.y = torch.tensor(targets, dtype=torch.float32)
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample and its target.

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (features, target)
        """
        return self.X[idx], self.y[idx]
