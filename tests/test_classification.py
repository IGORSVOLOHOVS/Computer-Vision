from typing import Any

import pytest
import torch

from src.application.service import ClassifierService
from src.infrastructure.image_adapter import PILImageLoader
from src.infrastructure.timm_adapter import TimmClassifier


def test_service_orchestration(
    mock_classifier: Any, mock_loader: Any, expected_classification_result: Any
) -> None:
    """Verifies that the service correctly interacts with ports using fixtures."""
    # Arrange
    mock_loader.load_and_transform.return_value = torch.zeros((1, 3, 224, 224))
    mock_classifier.classify.return_value = expected_classification_result
    mock_classifier.get_input_size.return_value = 224

    service = ClassifierService(classifier=mock_classifier, loader=mock_loader)

    # Act
    result = service.classify_image("dummy.png")

    # Assert
    assert result.top_prediction.label == "cat"
    assert result.top_prediction.confidence == 0.99
    mock_loader.load_and_transform.assert_called_once_with("dummy.png")
    mock_classifier.classify.assert_called_once()


def test_image_loader_not_found() -> None:
    """Verifies that the image loader raises FileNotFoundError if image is missing."""
    # Arrange
    loader = PILImageLoader()

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        loader.load_and_transform("non_existent_image.png")


def test_timm_adapter_init(mocker: Any, mock_timm_model: Any) -> None:
    """Verifies that the TimmClassifier calls timm correctly using mocker."""
    # Arrange
    mock_create = mocker.patch(
        "src.infrastructure.timm_adapter.timm.create_model",
        return_value=mock_timm_model,
    )

    # Act
    TimmClassifier(model_name="vit_base")

    # Assert
    mock_create.assert_called_once_with("vit_base", pretrained=True)
    mock_timm_model.eval.assert_called_once()


@pytest.mark.parametrize("has_labels", [True, False])
def test_timm_classifier_classify(
    mocker: Any, mock_timm_model: Any, sample_input_tensor: Any, has_labels: bool
) -> None:
    """Verifies classification logic with and without label names."""
    # Arrange
    if not has_labels:
        mock_timm_model.default_cfg["label_names"] = []

    mocker.patch(
        "src.infrastructure.timm_adapter.timm.create_model",
        return_value=mock_timm_model,
    )
    classifier = TimmClassifier(model_name="test_model")

    # Act
    result = classifier.classify(sample_input_tensor)

    # Assert
    if has_labels:
        assert result.top_prediction.label == "cat"
    else:
        assert result.top_prediction.label == "class_0"
    assert result.top_prediction.confidence == pytest.approx(1.0, rel=1e-2)


def test_timm_classifier_get_input_size(mocker: Any, mock_timm_model: Any) -> None:
    """Verifies extraction of input size from model config."""
    # Arrange
    mock_timm_model.default_cfg["input_size"] = (3, 299, 299)
    mocker.patch(
        "src.infrastructure.timm_adapter.timm.create_model",
        return_value=mock_timm_model,
    )
    classifier = TimmClassifier(model_name="test_model")

    # Act
    size = classifier.get_input_size()

    # Assert
    assert size == 299
