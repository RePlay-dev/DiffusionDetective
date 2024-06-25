import numpy as np
import shap
import torch

from explainers.base import Explainer
from utils import device


class SHAPExplainer(Explainer):
    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        print("Generating SHAP Explanation")
        # print(self.model)
        background = torch.randn((1, 3, 224, 224)).to(device)
        explainer = shap.GradientExplainer(self.model, background)

        # Generate shap values
        shap_values, indexes = explainer.shap_values(img_tensor, ranked_outputs=1)

        # Sum over the absolute values of SHAP values for each channel to get a single saliency map
        # Assuming the class of interest is the first one
        shap_values_for_class = shap_values[..., 0]  # Select SHAP values for the class of interest
        shap_saliency = np.sum(np.abs(shap_values_for_class), axis=1).squeeze(0)
        return shap_saliency
