"""Point 7: benchmarks, so a performance regression shows up as a number.

Inference is the model's cost. What a change to this repository can make slower
is everything around it: decoding a file, resizing, cropping, normalising, and
the service that wires loading to classification. That preprocessing runs on the
CPU for every image, so on a machine without a GPU to spare it is the ceiling on
throughput.

The model is deliberately excluded - `timm.create_model(pretrained=True)`
downloads weights, and a figure including that measures the network.

    pytest benchmarks --benchmark-only
    pytest benchmarks --benchmark-only --benchmark-compare
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

from src.application.service import ClassifierService
from src.domain.models import ClassificationResult, Prediction
from src.infrastructure.image_adapter import PILImageLoader


class StubClassifier:
    """Answers instantly, so only the pipeline around it is measured."""

    _answer = Prediction(label="stub", confidence=1.0)

    def classify(self, input_tensor: torch.Tensor) -> ClassificationResult:
        return ClassificationResult(
            top_prediction=self._answer,
            all_predictions=[self._answer],
            model_name="stub",
        )

    def get_model_name(self) -> str:
        return "stub"

    def get_input_size(self) -> int:
        return 224


@pytest.fixture(scope="module")
def images(tmp_path_factory: pytest.TempPathFactory) -> dict[int, Path]:
    """One JPEG per size, written once and shared by every benchmark."""
    directory = tmp_path_factory.mktemp("benchmark-images")
    written: dict[int, Path] = {}
    for side in (224, 512, 2048):
        pixels = np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
        path = directory / f"{side}.jpg"
        Image.fromarray(pixels).save(path, quality=90)
        written[side] = path
    return written


@pytest.mark.parametrize("side", [224, 512, 2048])
def test_preprocessing_scales_with_image_size(
    benchmark: Any, images: dict[int, Path], side: int
) -> None:
    """Decode, resize, crop, normalise - paid once per image, before the model."""
    loader = PILImageLoader()

    tensor = benchmark(loader.load_and_transform, images[side])

    assert tensor.shape == (1, 3, 224, 224)


def test_a_larger_target_size_costs_more(
    benchmark: Any, images: dict[int, Path]
) -> None:
    """384 is the other common ViT input; the resize dominates, so it shows."""
    loader = PILImageLoader(size=384)

    tensor = benchmark(loader.load_and_transform, images[512])

    assert tensor.shape == (1, 3, 384, 384)


def test_service_overhead(benchmark: Any, images: dict[int, Path]) -> None:
    """Load plus orchestration, with the model stubbed out."""
    service = ClassifierService(StubClassifier(), PILImageLoader())

    result = benchmark(service.classify_image, images[512])

    assert result.top_prediction.label == "stub"


def test_missing_file_fails_fast(benchmark: Any, tmp_path: Path) -> None:
    """The error path should cost a stat call, not a decode attempt."""
    loader = PILImageLoader()
    missing = tmp_path / "does-not-exist.jpg"

    def attempt() -> bool:
        try:
            loader.load_and_transform(missing)
        except FileNotFoundError:
            return True
        return False

    assert benchmark(attempt)
