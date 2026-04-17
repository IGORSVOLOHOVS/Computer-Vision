from pathlib import Path

from ..domain.interfaces import ImageClassifier, ImageLoader
from ..domain.models import ClassificationResult


class ClassifierService:
    """Orchestrator for image classification workflows."""

    def __init__(self, classifier: ImageClassifier, loader: ImageLoader) -> None:
        self.classifier = classifier
        self.loader = loader

    def classify_image(self, image_path: Path | str) -> ClassificationResult:
        """
        Loads, transforms, and classifies an image.

        Args:
            image_path: Path to the image file.

        Returns:
            ClassificationResult containing the prediction.
        """
        input_tensor = self.loader.load_and_transform(image_path)
        return self.classifier.classify(input_tensor)
