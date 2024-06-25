from abc import ABC, abstractmethod

import numpy as np
import torch

from utils import device


class Explainer(ABC):
    def __init__(self, model):
        self.model = model.to(device)

    @abstractmethod
    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        pass
