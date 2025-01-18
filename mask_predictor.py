import torch

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from mobile_sam import sam_model_registry, SamPredictor
from typing import Tuple, Dict


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def show_mask(mask):
    masked_regions = mask > 0
    mask_array = np.ma.masked_where(~masked_regions, mask)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.imshow(mask_array, alpha=0.5, cmap="rainbow")
    plt.axis("off")
    plt.show()


class MaskPredictor:
    BOX_THRESHOLD = 0.4
    TEXT_THRESHOLD = 0.3

    def __init__(self):
        ## Grounding Dino Object Detector
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.object_detector_processor = AutoProcessor.from_pretrained(model_id)
        self.object_detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id
        ).to(DEVICE)
        self.object_detector.eval()

        ## Mobile Segment Anything
        sam_checkpoint = "models/mobile_sam.pt"
        model_type = "vit_t"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=DEVICE)
        sam.eval()
        self.sam_predictor = SamPredictor(sam)

    def __get_bounding_boxes(self, image: Image.Image, text_prompt: str):
        inputs = self.object_detector_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.object_detector(**inputs)
            results = (
                self.object_detector_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=self.BOX_THRESHOLD,
                    text_threshold=self.TEXT_THRESHOLD,
                    target_sizes=[image.size[::-1]],
                )
            )
            bounding_boxes = results[0]["boxes"].numpy()
            labels = results[0]["labels"]
        return bounding_boxes, labels

    def __get_segmentation_mask(self, image: np.ndarray, bounding_boxes: np.ndarray):
        image = np.asarray(image)
        self.sam_predictor.set_image(image)
        transformed_boxes = np.array(
            [
                self.sam_predictor.transform.apply_boxes(
                    box, self.sam_predictor.original_size
                )
                for box in bounding_boxes
            ]
        )
        boxes_torch = torch.as_tensor(
            transformed_boxes, dtype=torch.float, device=self.sam_predictor.device
        )

        all_masks, iou_predictions, low_res_masks = self.sam_predictor.predict_torch(
            None,
            None,
            boxes_torch,
            None,
            True,
            return_logits=False,
        )
        return all_masks, iou_predictions, low_res_masks

    def inference(
        self, image: Image.Image, text_prompt: str
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """Get segmentation mask of image given a text prompt

        Args:
            image (Image.Image): Input image
            text_prompt (str): Text prompt formatted for input to hugging face Grounding Dino model.
            See https://huggingface.co/docs/transformers/en/model_doc/grounding-dino#usage-tips

        Returns:
            tuple(np.ndarray, dict):
            np.ndarray: Segmentation mask M where M[i,j]!=0 represent masked pixels of prompted clases
            and M[i,j] is the class label
            dict: Label ID to label name mapping
        """
        final_mask = torch.zeros(image.size[::-1], dtype=torch.int32)
        bounding_boxes, labels = self.__get_bounding_boxes(image, text_prompt)
        if not bounding_boxes.shape[0]:
            return final_mask.numpy(), {}
        label_id_mapping = {
            label: idx for idx, label in enumerate(set(labels), start=1)
        }
        id_label_mapping = {idx: label for label, idx in label_id_mapping.items()}
        ## SAM inference
        np_image = np.asarray(image)
        all_masks, iou_predictions, low_res_masks = self.__get_segmentation_mask(
            np_image, bounding_boxes
        )
        max_conf_idx = iou_predictions.argmax(dim=1)
        all_masks = all_masks[torch.arange(all_masks.shape[0]), max_conf_idx]
        for mask_index, mask in enumerate(all_masks):
            label_id = label_id_mapping[labels[mask_index]]
            final_mask[mask] = label_id
        return final_mask.numpy(), id_label_mapping


if __name__ == "__main__":
    predictor = MaskPredictor()
    image = Image.open("images/kitti_1.png").convert("RGB")
    text = "car.pole."
    mask, id_label_mapping = predictor.inference(image, text)
    show_mask(mask)
