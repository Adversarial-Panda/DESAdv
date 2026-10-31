from __future__ import annotations

import os 
from typing import TypeAlias

import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPModel
)

from .model import Model, HF_CACHE


Feature: TypeAlias = np.ndarray


class CLIP(metaclass=Model):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=HF_CACHE)
    preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=HF_CACHE)
    
    @classmethod
    def get_text_features(cls, text: str) -> list[Feature]: 
        text = cls.preprocessor(text=text, return_tensors='pt')
        features: torch.FloatTensor = cls.model.get_text_features(**text)
        features = features.detach().cpu().numpy()
        return features
    
    @classmethod
    def get_image_features(cls, images: list[Image.Image]) -> list[Feature]: 
        images = cls.preprocessor(images=images, return_tensors='pt')
        features: torch.FloatTensor = cls.model.get_image_features(**images)
        features = features.detach().cpu().numpy()
        return features