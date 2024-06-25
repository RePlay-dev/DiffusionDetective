import numpy as np
import torch
from pytorch_grad_cam import AblationCAM, GradCAM, GradCAMPlusPlus, EigenGradCAM, RandomCAM

from explainers.base import Explainer


class AblationCAMExplainer(Explainer):
    def get_saliency(self, img_tensor: torch.Tensor) -> np.ndarray:
        print("Generating AblationCAM Explanation")
        # GradCam requires gradients to be calculated
        for p in self.model.parameters():
            p.requires_grad = True

        img_tensor.requires_grad = True

        # TODO: Implement for CLIP
        # Correctly access the ConvNeXt model within the Sequential container
        convnext_model = self.model[0].model
        last_stage = convnext_model.stages[-1]
        last_block = last_stage.blocks[-1]
        target_layer = last_block.conv_dw

        # Create a AblationCAM object using the correct target layer
        ablation_cam = AblationCAM(model=self.model, target_layers=[target_layer])

        # Generate the CAM mask
        ablation_cam_mask = ablation_cam(input_tensor=img_tensor, eigen_smooth=True, aug_smooth=True)[0, :]

        for p in self.model.parameters():
            p.requires_grad = False

        return ablation_cam_mask

    def get_all_gradcam_saliency_maps(self, img_tensor: torch.Tensor) -> list[tuple[str, np.ndarray]]:
        print("Generating Explanations from all GradCAM Variants")
        # Enable gradients for parameters for GradCAM processing
        for p in self.model.parameters():
            p.requires_grad = True

        # Enable gradient
        img_tensor.requires_grad = True

        # Extract the target layer for CAM methods
        convnext_model = self.model[0].model
        last_stage = convnext_model.stages[-1]
        last_block = last_stage.blocks[-1]
        target_layer = last_block.conv_dw

        # Dictionary to hold GradCAM variants and their computed masks
        cams = {
            "GradCAM": GradCAM,
            "GradCAMPlusPlus": GradCAMPlusPlus,
            "EigenGradCAM": EigenGradCAM,
            "AblationCAM": AblationCAM,
            "RandomCAM": RandomCAM
        }

        # Using list comprehension to initialize CAM objects and compute masks along with names
        cam_results = [
            (cam_name, cam_class(model=self.model, target_layers=[target_layer])
                       (input_tensor=img_tensor, eigen_smooth=True, aug_smooth=True)[0, :])
            for cam_name, cam_class in cams.items()
        ]

        # Disable gradients for parameters after processing
        for p in self.model.parameters():
            p.requires_grad = False

        return cam_results
