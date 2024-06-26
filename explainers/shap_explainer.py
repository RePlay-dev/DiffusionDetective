import numpy as np
import torch
from captum.attr import GradientShap
from captum.attr._utils.visualization import _normalize_attr
from torchvision.transforms.functional import gaussian_blur

from explainers.base import Explainer


class SHAPExplainer(Explainer):

    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        print("Generating Advanced SHAP Explanation")

        # Create adaptive baselines (blurred versions of the input)
        baselines = []
        for sigma in [2.0, 5.0, 10.0, 20.0]:
            blurred = gaussian_blur(img_tensor, kernel_size=[15, 15], sigma=[sigma, sigma])
            baselines.append(blurred)
        baselines = torch.cat(baselines, dim=0)

        print("img_tensor shape:", img_tensor.shape)
        print("baselines shape:", baselines.shape)

        # Get the prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_label_idx = output.argmax().item()

        # Compute SHAP values
        gradient_shap = GradientShap(self.model)
        shap_attrs = gradient_shap.attribute(img_tensor,
                                             n_samples=50,
                                             stdevs=0.01,
                                             baselines=baselines,
                                             target=pred_label_idx)

        # Convert to numpy array and move channels to the last dimension
        shap_attrs = shap_attrs.squeeze().permute(1, 2, 0).cpu().numpy()

        saliency_map = _normalize_attr(shap_attrs, "absolute_value", 2, reduction_axis=2)

        return saliency_map
