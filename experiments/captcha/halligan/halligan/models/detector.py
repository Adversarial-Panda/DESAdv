from __future__ import annotations

import os
from typing import TypeAlias

import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    GroundingDinoProcessor,
    GroundingDinoModel
)

from .model import Model, HF_CACHE


Bbox: TypeAlias = list[int, int, int, int]


class Detector(metaclass=Model):
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor: GroundingDinoProcessor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE)
    model: GroundingDinoModel = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_id, cache_dir=HF_CACHE).to(device)
    model.eval()

    @classmethod
    def detect(cls, images: list[Image.Image], text: str) -> list[Bbox]:
        """Use GroundingDINO for open-set object detection

        Returns:
            A list of object bounding boxes (xmin, ymin, xmax, ymax)
        """
        images = [image.convert("RGB") for image in images]
        all_boxes, _, _ = cls._get_grounding_output(images, text, box_threshold=0.25, text_threshold=0.25)
        return all_boxes
    
    @classmethod
    def _get_grounding_output(
        cls, 
        images: list[Image.Image],
        caption: str,
        box_threshold: float,
        text_threshold: float
    ) -> torch.Tensor:
        if not caption.endswith("."):
            caption = caption + "."

        captions = [caption.lower().strip()] * len(images)
        inputs = cls.processor(images=images, text=captions, return_tensors="pt").to(cls.device)
        
        with torch.no_grad():
            outputs = cls.model(**inputs)
        
        results: list[dict] = cls.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold,
            text_threshold,
            target_sizes=[images[-1].size[::-1]] * len(images),
        )

        all_boxes, all_scores, all_labels = [], [], []

        for result in results:
            logits: torch.Tensor = result["scores"]
            boxes: torch.Tensor = result["boxes"]
            labels = result["labels"]

            all_boxes.append(boxes.cpu().int().tolist())
            all_scores.append(torch.Tensor([logit.item() for logit in logits.cpu()]))
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels