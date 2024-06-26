import numpy as np
import torch
from captum.attr import GradientShap
from captum.attr._utils.visualization import _normalize_attr
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

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

        # Enable gradients for the CLIP model
        self.model.requires_grad = True

        # Get the prediction
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred_label_idx = output.argmax().item()

        # Compute SHAP values using batch processing
        gradient_shap = GradientShap(self.model)

        # Define batch size and total number of samples
        batch_size = 10
        n_samples = 50
        shap_attrs = []

        # Use tqdm for the progress bar
        for i in tqdm(range(0, n_samples, batch_size), desc='Computing SHAP values'):
            batch_samples = min(batch_size, n_samples - i)
            batch_attrs = gradient_shap.attribute(img_tensor,
                                                  n_samples=batch_samples,
                                                  stdevs=0.01,
                                                  baselines=baselines,
                                                  target=pred_label_idx)
            shap_attrs.append(batch_attrs)

        # Combine and average the batched results
        shap_attrs = torch.cat(shap_attrs, dim=0).mean(dim=0, keepdim=True)

        # Disable gradients for the CLIP model
        self.model.requires_grad = False

        # Convert to numpy array and move channels to the last dimension
        shap_attrs = shap_attrs.squeeze().permute(1, 2, 0).cpu().numpy()
        saliency_map = _normalize_attr(shap_attrs, "absolute_value", 2, reduction_axis=2)
        return saliency_map
