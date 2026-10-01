import torch
from torch import nn


class FIFAOverallModel(nn.Module):
    """Feedforward regression model for predicting FIFA overall ratings."""

    def __init__(
        self,
        input_size: int,
        hidden_size1: int = 64,
        hidden_size2: int = 32,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be greater than zero.")
        if hidden_size1 <= 0 or hidden_size2 <= 0:
            raise ValueError("Hidden layer sizes must be greater than zero.")

        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return one predicted overall rating per input row."""
        if x.ndim != 2:
            raise ValueError(
                "Expected input tensor with shape (batch_size, input_size), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input tensor with {self.input_size} features, "
                f"got {x.shape[1]}."
            )

        return self.network(x)
