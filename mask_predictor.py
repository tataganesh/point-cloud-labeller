import torch
import sys
import requests

import numpy as np
import matplotlib.pyplot as plt
import cv2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image
from mobile_sam import sam_model_registry, SamPredictor
import time


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class MaskPredictor:

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

    def inference(self, image: np.ndarray, prompt: str):
        inputs = self.object_detector_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            outputs = self.object_detector(**inputs)
            results = (
                self.object_detector_processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.3,
                    target_sizes=[image.shape[::-1]],
                )
            )
            bounding_boxes = results[0]["boxes"].numpy()

            ## SAM inference
            self.sam_predictor.set_image(image)
            box = [
                self.sam_predictor.transform.apply_boxes(
                    box, self.sam_predictor.original_size
                )
                for box in bounding_boxes
            ]
            box_torch = torch.as_tensor(
                box, dtype=torch.float, device=self.sam_predictor.device
            )

            all_masks, iou_predictions, low_res_masks = (
                self.sam_predictor.predict_torch(
                    None,
                    None,
                    box_torch,
                    None,
                    True,
                    return_logits=False,
                )
            )
            print(all_masks.shape)


if __name__ == "__main__":
    predictor = MaskPredictor()
    image = cv2.imread("images/kitti_2.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = "car.street light.traffic light."
    predictor.inference(image, text)
