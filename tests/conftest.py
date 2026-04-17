from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from src.domain.models import ClassificationResult, Prediction


@pytest.fixture
def sample_input_tensor() -> torch.Tensor:
    """Returns a basic input tensor (1, 3, 224, 224)."""
    return torch.zeros((1, 3, 224, 224))


@pytest.fixture
def mock_timm_model(mocker: Any) -> MagicMock:
    """Provides a mocked timm model with default_cfg."""
    mock_model = MagicMock()
    mock_model.default_cfg = {
        "input_size": (3, 224, 224),
        "label_names": ["cat", "dog", "bird"],
    }
    # Mocking the forward pass to return a tensor of logits
    mock_model.return_value = torch.tensor([[10.0, 1.0, 0.5]])
    return mock_model


@pytest.fixture
def mock_classifier(mocker: Any) -> Any:
    """Fixture for ImageClassifier port."""
    return mocker.MagicMock()


@pytest.fixture
def mock_loader(mocker: Any) -> Any:
    """Fixture for ImageLoader port."""
    return mocker.MagicMock()


@pytest.fixture
def expected_classification_result() -> ClassificationResult:
    """Returns a standard classification result for testing."""
    return ClassificationResult(
        top_prediction=Prediction(label="cat", confidence=0.99),
        all_predictions=[Prediction(label="cat", confidence=0.99)],
        model_name="test_model",
    )
