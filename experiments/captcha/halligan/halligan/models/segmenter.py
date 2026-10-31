from __future__ import annotations

import os
from typing import TypeAlias

import torch
import cv2
import numpy as np
from PIL import Image
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
from ultralytics.engine.results import Results

from .model import Model, WEIGHTS_DIR


Bbox: TypeAlias = list[int, int, int, int]
Mask: TypeAlias = np.ndarray
Segment: TypeAlias = Image.Image


class Segmenter(metaclass=Model):

    model = FastSAM(str(WEIGHTS_DIR / "FastSAM-x.pt"))  # downloaded there by Ultralytics if missing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @classmethod
    def segment(cls, image: Image.Image) -> tuple[list[Bbox], list[Mask], list[Segment]]:
        """Use FastSAM for image segmentation

        Returns:
            A list of segment bounding boxes (xmin, ymin, xmax, ymax)
            A list of binary segmentation masks (np.ndarray)
            A list of segmented images (np.ndarray)
        """
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        everything_results = cls.model(image, device=cls.device, retina_masks=True, imgsz=768, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image, everything_results, device=cls.device)
        annotations = prompt_process.everything_prompt()
        annotation: Results = annotations[0]

        bboxes, masks, segments = [], [], []
        for a in annotation:
            img = np.copy(a.orig_img)
            for c in a:
                # Get mask from segment contour
                mask = np.zeros(a.orig_img.shape[:2], np.uint8)
                contour: np.ndarray = c.masks.xy.pop()
                contour = contour.astype(np.int32).reshape(-1, 1, 2)
                if contour.size == 0: continue
                _ = cv2.drawContours(mask, [contour], -1, (255, 255, 255), cv2.FILLED)

                # Get segmented image and its bbox from mask
                isolated = np.dstack([img, mask])
                bbox: np.ndarray = c.boxes.xyxy.cpu().numpy()
                bbox = bbox.squeeze().astype(np.int32)
                x1, y1, x2, y2 = bbox
                iso_segment = isolated[y1:y2, x1:x2]
                iso_segment = cv2.cvtColor(iso_segment, cv2.COLOR_BGR2RGB)
                iso_segment = Image.fromarray(iso_segment)
                
                bboxes.append([x1, y1, x2, y2])
                masks.append(mask)
                segments.append(iso_segment)

        return bboxes, masks, segments