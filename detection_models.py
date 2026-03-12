COLORMAP = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)


@dataclass
class SegmentationSample:
    img: np.ndarray
    classes_map: np.ndarray

    def create_viz(self) -> np.ndarray:
        alpha = 0.6
        colored_mask = COLORMAP[self.classes_map]
        viz = self.img.copy()
        h, w = viz.shape[:2]
        colored_mask = cv2.resize(colored_mask, (w, h))
        mask_over_image_viz = cv2.addWeighted(
            viz,
            alpha,
            colored_mask,
            1 - alpha,
            0,
        )
        return mask_over_image_viz
    
COLORMAP = np.array(
    [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 215, 0]], dtype=np.uint8
)


def coco_export_usage_sample():
    file_folder = os.path.dirname(__file__)
    img_folder = os.path.join(file_folder, "bird_sample", "images")
    polygons_json_file = os.path.join(
        file_folder, "bird_sample", "annotations", "instances_default.json"
    )

    polygon_dataset = COCOSematicSegmentationDataset(img_folder, polygons_json_file)
    poly_sample = polygon_dataset[0]
    viz = poly_sample.create_viz()
    plt.imshow(viz)
    plt.show()


coco_export_usage_sample()