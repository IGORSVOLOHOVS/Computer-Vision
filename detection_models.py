import torch
import cv2
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Dict, List, Tuple
from torchvision.transforms.functional import InterpolationMode
from torchvision.models.segmentation import FCN, FCN_ResNet50_Weights


CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
COLORMAP = np.array(
    [
        [0, 0, 0],  # background (black)
        [128, 0, 0],  # aeroplane (dark red)
        [0, 128, 0],  # bicycle (green)
        [128, 128, 0],  # bird (yellow-green)
        [0, 0, 128],  # boat (blue)
        [128, 0, 128],  # bottle (purple)
        [0, 128, 128],  # bus (teal)
        [128, 128, 128],  # car (gray)
        [64, 0, 0],  # cat (brown)
        [192, 0, 0],  # chair (red)
        [64, 128, 0],  # cow (olive)
        [192, 128, 0],  # diningtable (orange)
        [64, 0, 128],  # dog (indigo)
        [192, 0, 128],  # horse (pink)
        [64, 128, 128],  # motorbike (cyan)
        [192, 128, 128],  # person (light pink)
        [0, 64, 0],  # pottedplant (dark green)
        [128, 64, 0],  # sheep (mustard)
        [0, 192, 0],  # sofa (bright green)
        [128, 192, 0],  # train (lime)
        [0, 64, 128],  # tvmonitor (navy)
    ],
    dtype=np.uint8,
)


def load_model_for_inference() -> FCN:
    model = fcn_resnet50(
        weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
        progress=True,
    )
    model.eval()
    return model


def model_input_pipeline(image_path: str) -> torch.Tensor:
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    INPUT_SIZE = (520, 520)

    img = Image.open(image_path).convert("RGB")
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Resize(INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
            T.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ]
    )
    img = transform(img).unsqueeze(0)
    return img


def raw_tensor_to_class_mask(raw_CHW_logits: torch.Tensor) -> np.ndarray:
    single_pixel_logits = raw_CHW_logits[:, 100, 100]
    CHW_probs: torch.Tensor = torch.softmax(raw_CHW_logits, 0)
    single_pixel_probs = CHW_probs[:, 100, 100]
    tnz_class_mask: torch.Tensor = CHW_probs.argmax(0, keepdim=False)
    class_mask: np.ndarray = tnz_class_mask.numpy()
    return class_mask


def make_segmentation_results_plot(
    orig_image: np.ndarray,
    classes_mask: np.ndarray,
    colormap: np.ndarray,
    classes_names: List[str],
) -> None:
    fig, (image_ax, segmentation_map_ax) = plt.subplots(1, 2, figsize=(12, 6))
    image_ax.imshow(orig_image)
    image_ax.set_title("Исходное изображение")
    colored_mask = colormap[classes_mask]
    segmentation_map_ax.imshow(colored_mask)
    segmentation_map_ax.set_title("Маска классов")

    unique_classes = np.unique(classes_mask)
    patches = [
        segmentation_map_ax.scatter(
            0, 0, color=colormap[i] / 255, label=classes_names[i]
        )
        for i in unique_classes
    ]
    segmentation_map_ax.legend(
        handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    fig.tight_layout()
    plt.show()


def inference_pipeline(img_path: str) -> None:
    model = load_model_for_inference()
    input_image = model_input_pipeline(img_path)
    with torch.no_grad():
        output = model(input_image)["out"]
        output = output[0]
    output_mask = raw_tensor_to_class_mask(output)
    orig_img = Image.open(img_path)
    make_segmentation_results_plot(orig_img, output_mask, COLORMAP, CLASSES)


inference_pipeline("sparrow.png")