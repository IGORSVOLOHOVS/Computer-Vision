from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """Represents a single class prediction."""

    label: str
    confidence: float


@dataclass(frozen=True)
class ClassificationResult:
    """Represents the full result of an image classification."""

    top_prediction: Prediction
    all_predictions: list[Prediction]
    model_name: str
