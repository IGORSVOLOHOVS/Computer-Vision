from typing import cast

import timm
import torch

from ..domain.interfaces import ImageClassifier
from ..domain.models import ClassificationResult, Prediction


class TimmClassifier(ImageClassifier):
    """Adapter for timm library classifiers."""

    def __init__(
        self, model_name: str, pretrained: bool = True, use_fp16: bool = True
    ) -> None:
        self.model_name = model_name
        # Detect device: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()

        self.use_fp16 = use_fp16 and self.device.type != "cpu"
        if self.use_fp16:
            self.model = self.model.half()

        # default_cfg is a dynamic attribute on timm models
        default_cfg = getattr(self.model, "default_cfg", {})
        self.labels = cast(list[str], default_cfg.get("label_names", []))

    def classify(self, input_tensor: torch.Tensor) -> ClassificationResult:
        input_tensor = input_tensor.to(self.device)
        if self.use_fp16:
            input_tensor = input_tensor.half()

        with torch.no_grad():
            output = self.model(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, dim=0)

        # Get labels if available, otherwise use index
        top_idx_item = int(top_idx.item())
        top_label = (
            self.labels[top_idx_item]
            if self.labels and top_idx_item < len(self.labels)
            else f"class_{top_idx_item}"
        )

        top_prediction = Prediction(label=top_label, confidence=top_prob.item())

        # Optionally populate all_predictions (here just taking top for simplicity)
        all_predictions = [top_prediction]

        return ClassificationResult(
            top_prediction=top_prediction,
            all_predictions=all_predictions,
            model_name=self.model_name,
        )

    def get_model_name(self) -> str:
        return self.model_name

    def get_input_size(self) -> int:
        """Extracts input size from model default_cfg."""
        default_cfg = getattr(self.model, "default_cfg", {})
        input_size = default_cfg.get("input_size", (3, 224, 224))
        # Usually (C, H, W), we return H (assuming square)
        return int(input_size[-1])
