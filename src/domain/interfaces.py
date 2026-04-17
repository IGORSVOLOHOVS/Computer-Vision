from pathlib import Path
from typing import Protocol, runtime_checkable

import torch

from .models import ClassificationResult


@runtime_checkable
class ImageClassifier(Protocol):
    """Port for image classification engines."""

    def classify(self, input_tensor: torch.Tensor) -> ClassificationResult:
        """Classifies an image tensor."""
        ...

    def get_model_name(self) -> str:
        """Returns the name of the underlying model."""
        ...

    def get_input_size(self) -> int:
        """Returns the expected input size (width/height) for the model."""
        ...


@runtime_checkable
class ImageLoader(Protocol):
    """Port for image loading and preprocessing."""

    def load_and_transform(self, image_path: Path | str) -> torch.Tensor:
        """Loads an image and returns a preprocessed tensor."""
        ...
