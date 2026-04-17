from pathlib import Path
from typing import cast

import torch
from PIL import Image
from torchvision import transforms

from ..domain.interfaces import ImageLoader


class PILImageLoader(ImageLoader):
    """Adapter for PIL and torchvision image loading."""

    def __init__(self, size: int = 224) -> None:
        self.transform = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def load_and_transform(self, image_path: Path | str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found at {path}")

        image = Image.open(path).convert("RGB")
        # Add batch dimension [1, C, H, W]
        input_tensor = cast(torch.Tensor, self.transform(image).unsqueeze(0))
        return input_tensor
