from abc import ABC, abstractmethod
from pathlib import Path

import torch.nn as nn
from PIL import Image
from fastai.data.transforms import Category
from fastai.torch_core import TensorImage
from fastai.vision.core import PILImage
from numpy import ndarray
from torch import Tensor

from utils import device


class Detector(ABC):
    def __init__(self, model: nn.Module):
        self.model = model
        self.model = self.model.eval()  # Set model to evaluation mode
        self.model = self.model.to(device)
        self.explainers = {}

        for p in self.model.parameters():
            p.requires_grad = False

    def get_explanation(self, img: PILImage) -> tuple[Category | str, float, ndarray]:
        prediction, probability, index = self.get_prediction(img)
        return prediction, probability, self.get_saliency(img)[index]

    def add_explainer(self, name, explainer):
        self.explainers[name] = explainer

    def explain(self, img: Image, explainer_names: list) -> dict:
        results = {}
        img_tensor = self.get_processed_tensor(img)
        for name in explainer_names:
            explainer = self.explainers.get(name)
            if explainer:
                results[name] = explainer.get_saliency(img_tensor)
                # RISE Calculates explanations for all classes, so it needs the index to provide the correct map
                if name == 'RISE' and len(results[name].shape) > 2:
                    # TODO: Clean up that the index will not be returned by get prediction but only for detectors which really need it, such as DiffusionDetective
                    _, _, index = self.get_prediction(img)
                    results[name] = results[name][index]
        return results

    @abstractmethod
    def get_prediction(self, img: PILImage) -> tuple[Category | str, float, int]:
        pass

    @abstractmethod
    def get_processed_tensor(self, img: PILImage) -> TensorImage | Tensor:
        pass

    @abstractmethod
    def get_processed_image(self, img: PILImage) -> TensorImage | Tensor:
        pass

    @abstractmethod
    def get_metrics(self, path: Path) -> tuple[dict, float]:
        pass
